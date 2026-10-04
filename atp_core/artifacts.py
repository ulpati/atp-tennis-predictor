"""Saving and loading trained runs without pickle (CWE-502).

A run lives in ``<models_dir>/<run_id>/``::

    manifest.json     metadata, metrics and the SHA-256 of every other file
    model_<key>.ubj   XGBoost boosters in the native UBJSON format
    state.json.gz     player state used for prediction
    test_sample.npz   test rows for SHAP, read with ``allow_pickle=False``

``<models_dir>/LATEST`` holds the id of the most recent run. Every file is
verified against the manifest before it is parsed, file names and run ids
are validated, and symbolic links are refused.
"""

from __future__ import annotations

import gzip
import hmac
import io
import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xgboost as xgb

from . import __version__
from .config import FeatureConfig, TrainConfig
from .exceptions import ArtifactError, IntegrityError
from .features import FEATURE_NAMES
from .model import Calibrator, TrainResult
from .state import LeagueState
from .utils import atomic_write_bytes, atomic_write_text, sha256_bytes

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
LATEST_FILE = "LATEST"
MANIFEST_FILE = "manifest.json"
STATE_FILE = "state.json.gz"
SAMPLE_FILE = "test_sample.npz"
RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z(?:-\d{1,3})?$")
FILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
MODEL_KEYS = ("all", "Hard", "Clay", "Grass", "Carpet")
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_STATE_BYTES = 1024 * 1024 * 1024


def _json_safe(value):
    """Replace NaN and infinities with ``None`` so the JSON stays standard."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    return value


def _new_run_dir(models_dir: Path) -> tuple[str, Path]:
    base = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for suffix in range(1000):
        run_id = base if suffix == 0 else f"{base}-{suffix}"
        run_dir = models_dir / run_id
        try:
            run_dir.mkdir(mode=0o750)
        except FileExistsError:
            continue
        return run_id, run_dir
    raise ArtifactError(f"Cannot allocate a run directory in {models_dir}.")


def save_run(
    models_dir: str | Path,
    result: TrainResult,
    state: LeagueState,
    *,
    feature_config: FeatureConfig,
    train_config: TrainConfig,
    data_summary: dict,
) -> Path:
    """Write a complete run and point ``LATEST`` at it. Returns the run directory."""
    root = Path(models_dir)
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    run_id, run_dir = _new_run_dir(root)
    files: dict[str, str] = {}
    models: dict[str, dict] = {}

    for key, booster in result.boosters.items():
        if key not in MODEL_KEYS:
            raise ArtifactError(f"Unexpected model key {key!r}.")
        name = f"model_{key.lower()}.ubj"
        raw = bytes(booster.save_raw(raw_format="ubj"))
        atomic_write_bytes(run_dir / name, raw)
        files[name] = sha256_bytes(raw)
        models[key] = {"file": name, "best_iteration": int(result.best_iterations[key])}

    state_json = json.dumps(_json_safe(state.to_dict()), separators=(",", ":"), allow_nan=False)
    state_raw = gzip.compress(state_json.encode("utf-8"), compresslevel=6, mtime=0)
    atomic_write_bytes(run_dir / STATE_FILE, state_raw)
    files[STATE_FILE] = sha256_bytes(state_raw)

    if result.test_sample:
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            X=np.asarray(result.test_sample["X"], dtype=np.float32),
            y=np.asarray(result.test_sample["y"], dtype=np.int8),
            surface=np.asarray(result.test_sample["surface"], dtype="U8"),
        )
        atomic_write_bytes(run_dir / SAMPLE_FILE, buffer.getvalue())
        files[SAMPLE_FILE] = sha256_bytes(buffer.getvalue())

    manifest = {
        "format_version": FORMAT_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "atp_core_version": __version__,
        "xgboost_version": xgb.__version__,
        "feature_names": list(FEATURE_NAMES),
        "feature_config": feature_config.to_dict(),
        "train_config": train_config.to_dict(),
        "params": result.params,
        "models": models,
        "routing": result.routing,
        "calibrator": result.calibrator.to_dict() if result.calibrator else None,
        "metrics": result.metrics,
        "cv": result.cv,
        "search": result.search,
        "split": result.split,
        "refit": result.refit,
        "data": data_summary,
        "files": files,
    }
    atomic_write_text(run_dir / MANIFEST_FILE, json.dumps(_json_safe(manifest), indent=2, allow_nan=False) + "\n")
    atomic_write_text(root / LATEST_FILE, run_id + "\n")
    logger.info("Saved run %s to %s", run_id, run_dir)
    return run_dir


def list_runs(models_dir: str | Path) -> list[str]:
    root = Path(models_dir)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if RUN_ID_RE.match(p.name) and p.is_dir() and not p.is_symlink())


def resolve_run_dir(models_dir: str | Path, run_id: str | None = None) -> Path:
    """Return the directory of ``run_id`` (or of ``LATEST``), refusing anything suspicious."""
    root = Path(models_dir)
    if run_id is None:
        latest = root / LATEST_FILE
        if not latest.is_file() or latest.is_symlink():
            raise ArtifactError(f"No trained model in {root}. Run 'atp train' first.")
        with open(latest, encoding="utf-8") as handle:
            run_id = handle.read(64).strip()
    if not RUN_ID_RE.match(run_id or ""):
        raise ArtifactError(f"Invalid run id {run_id!r}.")
    run_dir = root / run_id
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ArtifactError(f"Run {run_id} not found in {root}.")
    return run_dir


def _read_file(run_dir: Path, name: str, limit: int) -> bytes:
    if not FILE_RE.match(name):
        raise ArtifactError(f"Invalid file name {name!r} in manifest.")
    path = run_dir / name
    if path.is_symlink():
        raise ArtifactError(f"Refusing the symbolic link {name} in {run_dir}.")
    if not path.is_file():
        raise ArtifactError(f"Missing file {name} in {run_dir}.")
    if path.stat().st_size > limit:
        raise ArtifactError(f"{name} is larger than allowed ({limit} bytes).")
    return path.read_bytes()


def _read_verified(run_dir: Path, files: dict[str, str], name: str, limit: int | None = None) -> bytes:
    expected = files.get(name)
    if expected is None or not SHA_RE.match(expected):
        raise ArtifactError(f"Manifest has no valid checksum for {name}.")
    data = _read_file(run_dir, name, MAX_FILE_BYTES if limit is None else limit)
    if not hmac.compare_digest(sha256_bytes(data), expected):
        raise IntegrityError(f"{name} in {run_dir} does not match its recorded SHA-256. Refusing to load it.")
    return data


def _decompress_limited(data: bytes, limit: int) -> bytes:
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as handle:
        out = handle.read(limit + 1)
    if len(out) > limit:
        raise ArtifactError("Decompressed state exceeds the size limit.")
    return out


@dataclass
class Run:
    """A loaded, verified training run."""

    run_id: str
    run_dir: Path
    manifest: dict
    boosters: dict[str, xgb.Booster]
    best_iterations: dict[str, int]
    routing: dict[str, str]
    calibrator: Calibrator | None
    state: LeagueState

    def model_key(self, surface: str) -> str:
        return self.routing.get(surface, "all")

    def test_sample(self) -> dict[str, np.ndarray]:
        files = self.manifest["files"]
        if SAMPLE_FILE not in files:
            raise ArtifactError("This run has no stored test sample.")
        data = _read_verified(self.run_dir, files, SAMPLE_FILE)
        with np.load(io.BytesIO(data), allow_pickle=False) as npz:
            return {k: npz[k] for k in ("X", "y", "surface")}


def load_manifest(run_dir: Path) -> dict:
    raw = _read_file(run_dir, MANIFEST_FILE, MAX_MANIFEST_BYTES)
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Corrupted manifest in {run_dir}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("format_version") != FORMAT_VERSION:
        raise ArtifactError(f"Run {run_dir.name} was saved with an unsupported format. Retrain with this version.")
    if not isinstance(manifest.get("files"), dict):
        raise ArtifactError("Manifest has no file list.")
    return manifest


def load_run(models_dir: str | Path, run_id: str | None = None) -> Run:
    """Load and verify a run. Nothing is deserialised before its checksum matches."""
    run_dir = resolve_run_dir(models_dir, run_id)
    manifest = load_manifest(run_dir)
    if manifest.get("feature_names") != list(FEATURE_NAMES):
        raise ArtifactError("This run uses a different feature set. Retrain it with this version of the code.")
    files = manifest["files"]

    boosters: dict[str, xgb.Booster] = {}
    best_iterations: dict[str, int] = {}
    for key, info in manifest.get("models", {}).items():
        if key not in MODEL_KEYS or not isinstance(info, dict):
            raise ArtifactError(f"Unexpected model entry {key!r} in manifest.")
        booster = xgb.Booster()
        try:
            booster.load_model(bytearray(_read_verified(run_dir, files, str(info.get("file", "")))))
        except xgb.core.XGBoostError as exc:
            raise ArtifactError(f"Cannot load model {key}: {exc}") from exc
        try:
            best_iteration = int(info.get("best_iteration", 0))
        except (TypeError, ValueError) as exc:
            raise ArtifactError(f"Invalid best_iteration for model {key}.") from exc
        if not 0 <= best_iteration < booster.num_boosted_rounds():
            raise ArtifactError(f"best_iteration of model {key} does not match its trees.")
        boosters[key] = booster
        best_iterations[key] = best_iteration
    if "all" not in boosters:
        raise ArtifactError("Run has no global model.")

    routing = {str(s): str(k) for s, k in manifest.get("routing", {}).items()}
    if any(k not in boosters for k in routing.values()):
        raise ArtifactError("Routing refers to a model that is not in the run.")

    state_raw = _decompress_limited(_read_verified(run_dir, files, STATE_FILE), MAX_STATE_BYTES)
    try:
        state = LeagueState.from_dict(json.loads(state_raw.decode("utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ArtifactError(f"Corrupted player state: {exc}") from exc

    calibrator = None
    if manifest.get("calibrator"):
        try:
            calibrator = Calibrator.from_dict(manifest["calibrator"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactError(f"Invalid calibrator: {exc}") from exc

    return Run(
        run_id=run_dir.name,
        run_dir=run_dir,
        manifest=manifest,
        boosters=boosters,
        best_iterations=best_iterations,
        routing=routing,
        calibrator=calibrator,
        state=state,
    )
