import json
import shutil

import numpy as np
import pytest

from atp_core import artifacts
from atp_core.artifacts import list_runs, load_manifest, load_run, resolve_run_dir
from atp_core.exceptions import ArtifactError, IntegrityError
from atp_core.model import predict_rows


@pytest.fixture
def run_copy(trained_models_dir, tmp_path):
    """A private copy of the trained run that a test may tamper with."""
    target = tmp_path / "models"
    shutil.copytree(trained_models_dir, target)
    return target, resolve_run_dir(target)


def _rewrite_manifest(run_dir, change):
    path = run_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    change(manifest)
    path.write_text(json.dumps(manifest))


def test_run_contains_only_non_executable_formats(trained_models_dir):
    run_dir = resolve_run_dir(trained_models_dir)
    names = {p.name for p in run_dir.iterdir()}
    assert names == {"manifest.json", "model_all.ubj", "state.json.gz", "test_sample.npz"}
    assert (trained_models_dir / "LATEST").read_text().strip() == run_dir.name
    assert list_runs(trained_models_dir) == [run_dir.name]


def test_round_trip_gives_identical_predictions(trained_models_dir):
    run = load_run(trained_models_dir)
    sample = run.test_sample()
    assert sample["X"].dtype == np.float32
    first = predict_rows(run.boosters["all"], sample["X"][:20], run.best_iterations["all"])
    again = predict_rows(
        load_run(trained_models_dir, run.run_id).boosters["all"], sample["X"][:20], run.best_iterations["all"]
    )
    assert np.array_equal(first, again)


def test_tampered_model_is_refused(run_copy):
    models, run_dir = run_copy
    with open(run_dir / "model_all.ubj", "ab") as handle:
        handle.write(b"\x00")
    with pytest.raises(IntegrityError):
        load_run(models)


def test_tampered_state_is_refused(run_copy):
    models, run_dir = run_copy
    (run_dir / "state.json.gz").write_bytes(b"not the original")
    with pytest.raises(IntegrityError):
        load_run(models)


@pytest.mark.parametrize("name", ["../../etc/passwd", ".hidden", "a/b.ubj", ""])
def test_manifest_file_names_are_validated(run_copy, name):
    models, run_dir = run_copy
    _rewrite_manifest(run_dir, lambda m: m["models"]["all"].update(file=name) or m["files"].update({name: "0" * 64}))
    with pytest.raises(ArtifactError):
        load_run(models)


@pytest.mark.parametrize("run_id", ["../outside", "20240101T000000Z/../x", "latest", "2024"])
def test_run_ids_are_validated(trained_models_dir, run_id):
    with pytest.raises(ArtifactError):
        resolve_run_dir(trained_models_dir, run_id)


def test_symlinked_run_directory_is_refused(run_copy, tmp_path):
    models, run_dir = run_copy
    elsewhere = tmp_path / "elsewhere"
    run_dir.rename(elsewhere)
    run_dir.symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ArtifactError):
        load_run(models)


def test_symlinked_file_is_refused(run_copy, tmp_path):
    models, run_dir = run_copy
    elsewhere = tmp_path / "state.json.gz"
    (run_dir / "state.json.gz").rename(elsewhere)
    (run_dir / "state.json.gz").symlink_to(elsewhere)
    with pytest.raises(ArtifactError, match="symbolic link"):
        load_run(models)


def test_missing_latest_pointer(tmp_path):
    with pytest.raises(ArtifactError, match="No trained model"):
        load_run(tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m.update(format_version=2),
        lambda m: m.update(feature_names=["x"]),
        lambda m: m["models"].pop("all"),
        lambda m: m.update(routing={"Clay": "Clay"}),
        lambda m: m["models"].update(Weird={"file": "model_all.ubj"}),
        lambda m: m.update(calibrator={"x": [1, 0], "y": [0, 1]}),
        lambda m: m["files"].update({"state.json.gz": "zz"}),
        lambda m: m["models"]["all"].update(best_iteration="many"),
        lambda m: m["models"]["all"].update(best_iteration=10**9),
        lambda m: m["models"]["all"].update(best_iteration=-1),
    ],
)
def test_inconsistent_manifests_are_refused(run_copy, change):
    models, run_dir = run_copy
    _rewrite_manifest(run_dir, change)
    with pytest.raises(ArtifactError):
        load_run(models)


def test_corrupted_manifest(run_copy):
    _, run_dir = run_copy
    (run_dir / "manifest.json").write_text("{not json")
    with pytest.raises(ArtifactError, match="Corrupted"):
        load_manifest(run_dir)


def test_decompression_is_bounded(run_copy, monkeypatch):
    models, _ = run_copy
    monkeypatch.setattr(artifacts, "MAX_STATE_BYTES", 100)
    with pytest.raises(ArtifactError, match="size limit"):
        load_run(models)


def test_oversized_files_are_refused(run_copy, monkeypatch):
    models, _ = run_copy
    monkeypatch.setattr(artifacts, "MAX_FILE_BYTES", 10)
    with pytest.raises(ArtifactError, match="larger than allowed"):
        load_run(models)
