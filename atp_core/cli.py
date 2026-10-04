"""Command-line interface: ``atp <command> [options]``."""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import io
import json
import logging
import math
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from . import __version__
from .config import (
    DATA_SOURCE_COMMIT,
    DATA_SOURCE_REPO,
    DEFAULT_DATA_DIR,
    DEFAULT_MODELS_DIR,
    DEFAULT_SEED,
    SURFACES,
    FeatureConfig,
    TrainConfig,
    env_path,
)
from .exceptions import ATPError, ConfigurationError
from .utils import printable

logger = logging.getLogger("atp")

MAX_BATCH_ROWS = 10_000
MAX_BATCH_BYTES = 16 * 1024 * 1024
BATCH_HEADER_KEYS = {"player1", "playera", "player_a", "player 1", "player"}
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
MAX_SEED = 2**31 - 1


# --------------------------------------------------------------------------- argument types
def _surface(value: str) -> str:
    surface = value.strip().capitalize()
    if surface not in SURFACES:
        raise argparse.ArgumentTypeError(f"invalid surface {value!r} (choose from {', '.join(SURFACES)})")
    return surface


def _date(value: str) -> str:
    try:
        return dt.date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}, expected YYYY-MM-DD") from exc


def _int_range(low: int, high: int) -> Callable[[str], int]:
    def parse(value: str) -> int:
        try:
            number = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid integer {value!r}") from exc
        if not low <= number <= high:
            raise argparse.ArgumentTypeError(f"{number} is outside [{low}, {high}]")
        return number

    return parse


def _float_range(low: float, high: float) -> Callable[[str], float]:
    def parse(value: str) -> float:
        try:
            number = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid number {value!r}") from exc
        if not (low <= number <= high) or math.isnan(number):
            raise argparse.ArgumentTypeError(f"{number} is outside [{low}, {high}]")
        return number

    return parse


# --------------------------------------------------------------------------- output helpers
def _pct(value: float | None) -> str:
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value * 100:.1f}%"


def _num(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]], right: Sequence[int] = ()) -> str:
    cells = [[str(c) for c in headers]] + [[str(c) for c in row] for row in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]

    def fmt(row: list[str]) -> str:
        padded = (c.rjust(w) if i in right else c.ljust(w) for i, (c, w) in enumerate(zip(row, widths, strict=True)))
        return "  ".join(padded).rstrip()

    lines = [fmt(cells[0]), "  ".join("-" * w for w in widths)]
    lines.extend(fmt(r) for r in cells[1:])
    return "\n".join(lines)


def _print_json(data: object) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def _models_dir(args: argparse.Namespace) -> Path:
    return args.models_dir or env_path("ATP_MODELS_DIR", DEFAULT_MODELS_DIR)


def _data_dir(args: argparse.Namespace) -> Path:
    return args.data_dir or env_path("ATP_DATA_DIR", DEFAULT_DATA_DIR)


def _load(args: argparse.Namespace):
    from .artifacts import load_run

    return load_run(_models_dir(args), args.run)


# --------------------------------------------------------------------------- commands
def cmd_download(args: argparse.Namespace) -> int:
    from .download import download

    written = download(
        _data_dir(args),
        args.from_year,
        args.to_year,
        repo=args.repo,
        commit=args.commit,
        force=args.force,
        timeout=args.timeout,
    )
    print(f"Downloaded {len(written)} file(s) into {_data_dir(args)}.")
    print("Data: Jeff Sackmann / Tennis Abstract, CC BY-NC-SA 4.0 (attribution required, non-commercial use only).")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .artifacts import load_manifest, save_run
    from .data import clean_matches, load_matches, summarize
    from .features import build_training_set
    from .model import train

    feature_cfg = FeatureConfig(
        recent_n=args.recent_n, stats_window=args.stats_window, time_decay=args.time_decay, elo_k=args.elo_k
    ).validate()
    train_cfg = TrainConfig(
        train_start=args.train_start,
        test_start=args.test_start,
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
        search_iter=args.search,
        cv_folds=args.cv_folds,
        per_surface=args.per_surface,
        calibration=args.calibration,
        refit=args.refit,
        use_gpu=args.use_gpu,
        threads=args.threads,
        seed=args.seed,
        max_rounds=args.max_rounds,
    ).validate()
    models_dir = _models_dir(args)
    models_dir.mkdir(parents=True, exist_ok=True, mode=0o750)
    if not os.access(models_dir, os.W_OK | os.X_OK):  # fail now rather than after training
        raise ConfigurationError(f"Cannot write to {models_dir}.")
    years = (args.from_year or 1968, args.to_year or 2100) if args.from_year or args.to_year else None
    df = clean_matches(load_matches(_data_dir(args), years))
    training_set = build_training_set(df, feature_cfg, args.train_start)
    result = train(training_set, train_cfg)
    run_dir = save_run(
        models_dir,
        result,
        training_set.state,
        feature_config=feature_cfg,
        train_config=train_cfg,
        data_summary=summarize(df),
    )
    _print_evaluation(load_manifest(run_dir))
    print(f"\nSaved run {run_dir.name} in {run_dir.parent}.")
    return 0


def _print_evaluation(manifest: dict) -> None:
    metrics = manifest["metrics"]
    data, split = manifest.get("data", {}), manifest.get("split", {})
    print(f"Run {manifest['run_id']}  (created {manifest['created_at']}, atp_core {manifest['atp_core_version']})")
    print(f"Data: {data.get('matches', 'n/a')} matches, {data.get('first_date')} to {data.get('last_date')}")
    for name in ("train", "validation", "test"):
        period = split.get(name, {})
        print(f"  {name:<10} {period.get('matches', 0):>7} matches  {period.get('from')} to {period.get('to')}")

    test, base = metrics["test"], metrics["baselines"]
    rows = [
        ["Model (XGBoost)", _pct(test["accuracy"]), _num(test["log_loss"]), _num(test["brier"]), _num(test["auc"])],
        [
            "Elo only",
            _pct(base["elo"]["accuracy"]),
            _num(base["elo"]["log_loss"]),
            _num(base["elo"]["brier"]),
            _num(base["elo"]["auc"]),
        ],
        [
            "Surface Elo only",
            _pct(base["surface_elo"]["accuracy"]),
            _num(base["surface_elo"]["log_loss"]),
            _num(base["surface_elo"]["brier"]),
            _num(base["surface_elo"]["auc"]),
        ],
        ["Better ranking wins", _pct(base["ranking"]["accuracy"]), "", "", ""],
    ]
    print(f"\nTest period ({test['matches']} matches, never used for training or tuning):")
    print(_table(["Predictor", "Accuracy", "Log loss", "Brier", "AUC"], rows, right=(1, 2, 3, 4)))

    by_surface = metrics.get("test_by_surface", {})
    if by_surface:
        print("\nModel on the test period, by surface:")
        print(
            _table(
                ["Surface", "Matches", "Accuracy", "Log loss"],
                [[s, m["matches"], _pct(m["accuracy"]), _num(m["log_loss"])] for s, m in by_surface.items()],
                right=(1, 2, 3),
            )
        )
    if manifest.get("calibrator"):
        raw = metrics.get("test_uncalibrated", {})
        print(f"\nIsotonic calibration: log loss {_num(raw.get('log_loss'))} before, {_num(test['log_loss'])} after.")
    if metrics.get("surface_models"):
        print("\nPer-surface models (chosen on the validation period):")
        for surface, info in metrics["surface_models"].items():
            chosen = "surface model" if info["chosen"] != "all" else "global model"
            print(
                f"  {surface}: {chosen} (validation log loss: surface {_num(info['validation_log_loss_surface'])}, "
                f"global {_num(info['validation_log_loss_global'])})"
            )
    cv = manifest.get("cv") or []
    if cv:
        acc = [f["accuracy"] for f in cv]
        loss = [f["log_loss"] for f in cv]
        mean_acc, mean_loss = sum(acc) / len(acc), sum(loss) / len(loss)
        sd_acc = math.sqrt(sum((a - mean_acc) ** 2 for a in acc) / len(acc))
        print(
            f"\nCross-validation ({len(cv)} expanding-window folds): accuracy {_pct(mean_acc)} ± {_pct(sd_acc)}, "
            f"log loss {_num(mean_loss)}"
        )
    refit = manifest.get("refit") or {}
    if refit:
        matches = refit["models"]["all"]["matches"]
        print(
            f"\nSaved model: retrained with the same settings on all {matches} matches "
            f"from {refit['from']} to {refit['to']}, test period included."
        )
    else:
        print("\nSaved model: the evaluated one, trained on the training period only.")


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .artifacts import load_manifest, resolve_run_dir

    manifest = load_manifest(resolve_run_dir(_models_dir(args), args.run))
    if args.json:
        _print_json(
            {
                k: manifest.get(k)
                for k in (
                    "run_id",
                    "created_at",
                    "data",
                    "split",
                    "metrics",
                    "cv",
                    "refit",
                    "routing",
                    "params",
                    "feature_config",
                    "train_config",
                )
            }
        )
    else:
        _print_evaluation(manifest)
    return 0


def _print_prediction(p) -> None:
    print(f"{p.player_a} vs {p.player_b}  |  {p.surface}, best of {p.best_of}  |  as of {p.as_of}")
    print(
        _table(
            ["Player", "Model", "Elo only"],
            [
                [p.player_a, _pct(p.probability_a), _pct(p.elo_probability_a)],
                [p.player_b, _pct(p.probability_b), _pct(1 - p.elo_probability_a)],
            ],
            right=(1, 2),
        )
    )
    print(f"Predicted winner: {p.winner}  (confidence {p.confidence}, model '{p.model}')")
    for warning in p.warnings:
        print(f"Warning: {warning}")
    if p.explanation:
        print("\nMain factors (log-odds for the first player):")
        print(
            _table(
                ["Feature", "Value", "Effect"],
                [[e["feature"], _num(e["value_a_vs_b"], 3), f"{e['log_odds']:+.3f}"] for e in p.explanation],
                right=(1, 2),
            )
        )


def cmd_predict(args: argparse.Namespace) -> int:
    from .predict import Predictor

    prediction = Predictor(_load(args)).predict(
        args.player_a, args.player_b, args.surface, best_of=args.best_of, as_of=args.date, explain=args.explain
    )
    if args.json:
        _print_json(prediction.to_dict())
    else:
        _print_prediction(prediction)
    return 0


def _read_batch(path: Path, max_rows: int) -> list[list[str]]:
    """Rows of a batch file. A UTF-8 byte order mark and ';' separators, as Excel writes them, are accepted."""
    with open(path, "rb") as handle:
        raw = handle.read(MAX_BATCH_BYTES + 1)
    if len(raw) > MAX_BATCH_BYTES:
        raise ConfigurationError(f"{path} is larger than {MAX_BATCH_BYTES // 2**20} MB.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ConfigurationError(f"{path} is not a UTF-8 text file.") from exc
    first = next((line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")), "")
    delimiter = ";" if first.count(";") > first.count(",") else ","
    rows: list[list[str]] = []
    try:
        for line_no, row in enumerate(csv.reader(io.StringIO(text, newline=""), delimiter=delimiter), 1):
            if not row or not any(cell.strip() for cell in row) or row[0].lstrip().startswith("#"):
                continue
            if not rows and row[0].strip().lower() in BATCH_HEADER_KEYS:
                continue
            if len(rows) >= max_rows:
                raise ConfigurationError(f"{path} has more than {max_rows} rows; use --max-rows to raise the limit.")
            if len(row) < 3:
                raise ConfigurationError(f"{path}:{line_no}: expected 'player_a,player_b,surface[,best_of]'.")
            rows.append([cell.strip() for cell in row[:4]])
    except csv.Error as exc:
        raise ConfigurationError(f"{path} is not a valid CSV file: {exc}") from exc
    return rows


def cmd_batch(args: argparse.Namespace) -> int:
    from .predict import Predictor
    from .utils import sanitize_csv_cell

    predictor = Predictor(_load(args))
    results = []
    for row in _read_batch(args.file, args.max_rows):
        a, b, surface = row[0], row[1], row[2]
        best_of_text = row[3] if len(row) > 3 else ""
        if best_of_text not in ("", "3", "5"):
            results.append({"player_a": a, "player_b": b, "surface": surface, "error": "best_of must be 3 or 5."})
            continue
        best_of = 5 if best_of_text == "5" else 3
        try:
            results.append(predictor.predict(a, b, surface, best_of=best_of, as_of=args.date).to_dict())
        except ATPError as exc:
            results.append({"player_a": a, "player_b": b, "surface": surface, "error": str(exc)})
    if args.output:
        fields = [
            "player_a",
            "player_b",
            "surface",
            "best_of",
            "probability_a",
            "probability_b",
            "predicted_winner",
            "confidence",
            "warnings",
            "error",
        ]
        with open(args.output, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            for r in results:
                row = {**r, "warnings": "; ".join(r.get("warnings") or [])}
                writer.writerow([sanitize_csv_cell(row.get(f, "")) for f in fields])
        print(f"Wrote {len(results)} predictions to {args.output}.")
    if args.json:
        _print_json(results)
    elif not args.output:
        print(
            _table(
                ["Player A", "Player B", "Surface", "P(A)", "Predicted winner"],
                [
                    [
                        r["player_a"],
                        r["player_b"],
                        r["surface"],
                        _pct(r.get("probability_a")),
                        r.get("predicted_winner") or f"error: {r['error']}",
                    ]
                    for r in results
                ],
                right=(3,),
            )
        )
        for warning in dict.fromkeys(w for r in results for w in r.get("warnings") or []):
            print(f"Warning: {warning}")
    return 0 if all("error" not in r for r in results) else 1


def cmd_list(args: argparse.Namespace) -> int:
    from .features import from_epoch_day

    state = _load(args).state
    cutoff = (state.last_day or 0) - args.active_days
    active = [(pid, r) for pid, r in state.players.items() if r.last_day is not None and r.last_day >= cutoff]
    surface = args.surface

    def key(item):
        pid, r = item
        if args.sort == "wins":
            return r.wins
        return state.elo.rating(pid, surface)

    ranked = sorted(active, key=key, reverse=True)[: args.top]
    players = [
        {
            "position": i,
            "player": r.name,
            "elo": state.elo.rating(pid, surface),
            "wins": r.wins,
            "losses": r.losses,
            "atp_rank": None if math.isnan(r.rank) else int(r.rank),
            "last_match": from_epoch_day(r.last_day).date().isoformat(),
        }
        for i, (pid, r) in enumerate(ranked, 1)
    ]
    if args.json:
        _print_json(players)
        return 0
    label = f"{surface} Elo" if surface else "Elo"
    print(
        f"Top {len(players)} players active in the last {args.active_days} days, sorted by "
        f"{'wins' if args.sort == 'wins' else label}:"
    )
    rows = [
        [
            p["position"],
            p["player"],
            f"{p['elo']:.0f}",
            f"{p['wins']}-{p['losses']}",
            p["atp_rank"] or "n/a",
            p["last_match"],
        ]
        for p in players
    ]
    print(_table(["#", "Player", label, "W-L", "ATP rank", "Last match"], rows, right=(0, 2, 3, 4)))
    return 0


def cmd_player(args: argparse.Namespace) -> int:
    from .features import from_epoch_day
    from .predict import PlayerIndex

    run = _load(args)
    state = run.state
    pid = PlayerIndex.from_run(run).resolve(args.player)
    r = state.players[pid]
    surfaces = []
    for s in SURFACES:
        w, lo = r.surface_wins.get(s, 0), r.surface_losses.get(s, 0)
        if w + lo:
            surfaces.append(
                {"surface": s, "wins": w, "losses": lo, "win_rate": w / (w + lo), "elo": state.elo.rating(pid, s)}
            )
    profile = {
        "player": r.name,
        "wins": r.wins,
        "losses": r.losses,
        "win_rate": r.wins / (r.wins + r.losses) if r.wins + r.losses else None,
        "elo": state.elo.rating(pid),
        "atp_rank": None if math.isnan(r.rank) else int(r.rank),
        "last_match": from_epoch_day(r.last_day).date().isoformat() if r.last_day is not None else None,
        "recent_results": "".join("W" if x else "L" for x in r.results),
        "surfaces": surfaces,
    }
    if args.json:
        _print_json(profile)
        return 0
    print(
        f"{r.name}: {r.wins}-{r.losses} ({_pct(profile['win_rate'])}), Elo {profile['elo']:.0f}, "
        f"last ATP rank {profile['atp_rank'] or 'n/a'}, last match {profile['last_match']}"
    )
    print(
        _table(
            ["Surface", "W-L", "Win rate", "Surface Elo"],
            [[s["surface"], f"{s['wins']}-{s['losses']}", _pct(s["win_rate"]), f"{s['elo']:.0f}"] for s in surfaces],
            right=(1, 2, 3),
        )
    )
    print(f"Recent results, oldest first: {profile['recent_results'] or 'none'}")
    return 0


def cmd_h2h(args: argparse.Namespace) -> int:
    from .predict import PlayerIndex

    run = _load(args)
    index = PlayerIndex.from_run(run)
    a, b = index.resolve(args.player_a), index.resolve(args.player_b)
    state = run.state
    name_a, name_b = state.players[a].name, state.players[b].name
    rows = []
    for s in ("all", *SURFACES):
        wa, wb = state.h2h_wins(a, b, s), state.h2h_wins(b, a, s)
        if wa + wb or s == "all":
            rows.append([s if s != "all" else "All surfaces", wa, wb])
    if args.json:
        _print_json(
            {
                "player_a": name_a,
                "player_b": name_b,
                "records": [{"surface": r[0], "wins_a": r[1], "wins_b": r[2]} for r in rows],
            }
        )
        return 0
    print(f"Head-to-head: {name_a} vs {name_b}")
    print(_table(["Surface", name_a, name_b], rows, right=(1, 2)))
    return 0


def cmd_importance(args: argparse.Namespace) -> int:
    from .explain import gain_importance

    run = _load(args)
    ranking = gain_importance(run.boosters["all"])[: args.top]
    if args.json:
        _print_json([{"feature": n, "gain_share": v} for n, v in ranking])
        return 0
    print("Share of total gain in the global model:")
    print(
        _table(["#", "Feature", "Gain share"], [[i, n, _pct(v)] for i, (n, v) in enumerate(ranking, 1)], right=(0, 2))
    )
    return 0


def cmd_shap(args: argparse.Namespace) -> int:
    from .explain import mean_abs_shap

    run = _load(args)
    sample = run.test_sample()
    ranking = mean_abs_shap(run.boosters["all"], sample["X"], run.best_iterations["all"], max_rows=args.max_samples)
    below = [n for n, v in ranking if v < args.threshold]
    if args.json:
        _print_json(
            {
                "mean_abs_shap": [{"feature": n, "value": v} for n, v in ranking],
                "below_threshold": below,
                "threshold": args.threshold,
            }
        )
        return 0
    print(f"Mean |SHAP| (log-odds) on {min(len(sample['X']), args.max_samples)} test rows:")
    print(
        _table(
            ["#", "Feature", "Mean |SHAP|"],
            [[i, n, _num(v)] for i, (n, v) in enumerate(ranking[: args.top], 1)],
            right=(0, 2),
        )
    )
    if below:
        print(f"\n{len(below)} features fall below {args.threshold} and are candidates for removal: {', '.join(below)}")
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    from .artifacts import list_runs, load_manifest

    root = _models_dir(args)
    rows = []
    for run_id in list_runs(root):
        try:
            m = load_manifest(root / run_id)
        except ATPError as exc:
            rows.append([run_id, "unreadable", str(exc), "", ""])
            continue
        test = m.get("metrics", {}).get("test", {})
        rows.append(
            [
                run_id,
                m.get("atp_core_version", "?"),
                m.get("data", {}).get("last_date", "?"),
                _pct(test.get("accuracy")),
                _num(test.get("log_loss")),
            ]
        )
    if args.json:
        keys = ["run", "version", "data_through", "test_accuracy", "test_log_loss"]
        _print_json([dict(zip(keys, r, strict=True)) for r in rows])
    elif rows:
        print(_table(["Run", "Version", "Data through", "Test accuracy", "Test log loss"], rows, right=(3, 4)))
    else:
        print(f"No runs in {root}.")
    return 0


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    run_opts = argparse.ArgumentParser(add_help=False)
    run_opts.add_argument(
        "--models-dir", type=Path, help="directory with trained runs (env ATP_MODELS_DIR, default 'models')"
    )
    run_opts.add_argument("--run", help="run id to use (default: the latest)")
    json_opt = argparse.ArgumentParser(add_help=False)
    json_opt.add_argument("--json", action="store_true", help="print JSON instead of a table")
    data_opt = argparse.ArgumentParser(add_help=False)
    data_opt.add_argument(
        "--data-dir", type=Path, help="directory with atp_matches_YYYY.csv (env ATP_DATA_DIR, default 'data')"
    )

    parser = argparse.ArgumentParser(
        prog="atp",
        description="ATP match predictor: leakage-free features, XGBoost and Elo ratings.",
        epilog="Data: Jeff Sackmann / Tennis Abstract, CC BY-NC-SA 4.0.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--log-level",
        default=os.environ.get("ATP_LOG_LEVEL", "").strip().upper() or "INFO",
        choices=LOG_LEVELS,
        help="logging verbosity (env ATP_LOG_LEVEL, default INFO)",
    )
    parser.add_argument("--log-file", type=Path, help="also write logs to this file")
    parser.add_argument("--debug", action="store_true", help="show full tracebacks on errors")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    this_year = dt.date.today().year

    p = sub.add_parser("download", parents=[data_opt], help="download the ATP match files")
    p.add_argument("--from-year", type=_int_range(1968, 2100), default=1968, help="first year (default 1968)")
    p.add_argument("--to-year", type=_int_range(1968, 2100), default=this_year, help="last year (default: this year)")
    p.add_argument("--repo", default=DATA_SOURCE_REPO, help=f"GitHub 'owner/name' (default {DATA_SOURCE_REPO})")
    p.add_argument("--commit", default=DATA_SOURCE_COMMIT, help="full commit SHA to download from (default: pinned)")
    p.add_argument("--force", action="store_true", help="replace files that already exist")
    p.add_argument("--timeout", type=_float_range(1, 300), default=30.0, help="seconds per request")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("train", parents=[data_opt], help="train and evaluate a model, then retrain it on every match")
    p.add_argument("--models-dir", type=Path, help="where to save the run (env ATP_MODELS_DIR, default 'models')")
    p.add_argument("--from-year", type=_int_range(1968, 2100), help="ignore files before this year")
    p.add_argument("--to-year", type=_int_range(1968, 2100), help="ignore files after this year")
    p.add_argument(
        "--train-start",
        type=_date,
        help="first match date that produces training rows "
        "(earlier matches only warm up ratings; default: 3 years after the first match)",
    )
    p.add_argument("--test-start", type=_date, help="first date of the test period (default: last 10%% of matches)")
    p.add_argument("--test-fraction", type=_float_range(0.01, 0.49), default=0.10)
    p.add_argument("--val-fraction", type=_float_range(0.01, 0.49), default=0.10)
    p.add_argument("--search", type=_int_range(0, 500), default=0, help="random-search iterations (default 0)")
    p.add_argument("--cv-folds", type=_int_range(0, 20), default=0, help="expanding-window CV folds (default 0)")
    p.add_argument(
        "--per-surface",
        action="store_true",
        help="also train Hard/Clay/Grass models; each is used only if it beats the global model on validation",
    )
    p.add_argument(
        "--calibration",
        choices=["none", "isotonic"],
        default="none",
        help="isotonic calibration fitted on the validation period (needs --no-refit)",
    )
    p.add_argument(
        "--no-refit",
        dest="refit",
        action="store_false",
        help="save the evaluated model instead of retraining it on all periods, test period included",
    )
    p.add_argument("--elo-k", type=_float_range(1, 200), help="fixed Elo K-factor (default: dynamic K)")
    p.add_argument("--recent-n", type=_int_range(10, 500), default=20, help="matches for recent form")
    p.add_argument("--stats-window", type=_int_range(1, 500), default=30, help="matches for serve/return stats")
    p.add_argument(
        "--time-decay",
        type=_float_range(0, 0.999),
        default=0.0,
        help="per-match decay for recent form, e.g. 0.9 (default 0 = off)",
    )
    p.add_argument("--use-gpu", action="store_true", help="train on CUDA when available")
    p.add_argument("--threads", type=_int_range(1, 256), help="CPU threads (default: up to 8)")
    p.add_argument("--seed", type=_int_range(0, MAX_SEED), default=DEFAULT_SEED)
    p.add_argument("--max-rounds", type=_int_range(10, 20000), default=3000)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", parents=[run_opts, json_opt], help="show the metrics of a run")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("predict", parents=[run_opts, json_opt], help="predict one match")
    p.add_argument("player_a")
    p.add_argument("player_b")
    p.add_argument("surface", type=_surface, help=f"one of {', '.join(SURFACES)}")
    p.add_argument("--best-of", type=int, choices=[3, 5], default=3)
    p.add_argument("--date", type=_date, help="match date (default: last date in the data)")
    p.add_argument("--explain", action="store_true", help="show the features that drive the prediction")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("batch", parents=[run_opts, json_opt], help="predict matches listed in a CSV file")
    p.add_argument("file", type=Path, help="CSV rows: player_a,player_b,surface[,best_of]")
    p.add_argument("--output", type=Path, help="write results to this CSV file")
    p.add_argument("--date", type=_date, help="match date (default: last date in the data)")
    p.add_argument("--max-rows", type=_int_range(1, 1_000_000), default=MAX_BATCH_ROWS)
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("list", parents=[run_opts, json_opt], help="list the strongest active players")
    p.add_argument("--top", type=_int_range(1, 1000), default=20)
    p.add_argument("--sort", choices=["elo", "wins"], default="elo")
    p.add_argument("--surface", type=_surface, help="sort by the Elo of this surface")
    p.add_argument("--active-days", type=_int_range(1, 36500), default=365)
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("player", parents=[run_opts, json_opt], help="show a player's profile")
    p.add_argument("player")
    p.set_defaults(func=cmd_player)

    p = sub.add_parser("h2h", parents=[run_opts, json_opt], help="head-to-head record of two players")
    p.add_argument("player_a")
    p.add_argument("player_b")
    p.set_defaults(func=cmd_h2h)

    p = sub.add_parser("importance", parents=[run_opts, json_opt], help="feature importance (total gain)")
    p.add_argument("--top", type=_int_range(1, 500), default=20)
    p.set_defaults(func=cmd_importance)

    p = sub.add_parser("shap", parents=[run_opts, json_opt], help="mean |SHAP| on the test sample")
    p.add_argument("--top", type=_int_range(1, 500), default=20)
    p.add_argument("--max-samples", type=_int_range(10, 100_000), default=2000)
    p.add_argument("--threshold", type=_float_range(0, 10), default=0.005)
    p.set_defaults(func=cmd_shap)

    p = sub.add_parser("runs", parents=[json_opt], help="list saved runs")
    p.add_argument("--models-dir", type=Path, help="directory with trained runs")
    p.set_defaults(func=cmd_runs)
    return parser


class _PrintableFormatter(logging.Formatter):
    """Log formatter that keeps control characters from data or batch files out of the terminal."""

    def format(self, record: logging.LogRecord) -> str:
        return printable(super().format(record))


class _PrintableWriter:
    """Standard output wrapper that replaces control characters before they reach the terminal."""

    def __init__(self, stream) -> None:
        self._stream = stream

    def write(self, text: str) -> int:
        return self._stream.write(printable(text))

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


def configure_logging(level: str, log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    formatter = _PrintableFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(level=getattr(logging, level), handlers=handlers, force=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.log_level not in LOG_LEVELS:  # argparse does not check defaults, here ATP_LOG_LEVEL
        parser.error(f"ATP_LOG_LEVEL must be one of {', '.join(LOG_LEVELS)}")
    try:
        configure_logging(args.log_level, args.log_file)
    except OSError as exc:
        parser.error(f"cannot open the log file: {exc.strerror}")
    try:
        with contextlib.redirect_stdout(_PrintableWriter(sys.stdout)):
            return int(args.func(args) or 0)
    except ATPError as exc:
        if args.debug:
            logger.exception("Command failed")
        else:
            logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logger.error("Interrupted.")
        return 130
    except OSError as exc:
        # Missing files, missing permissions or a full disk are the user's to fix: one clear line.
        if args.debug:
            raise
        logger.error("%s", f"{exc.strerror}: {exc.filename}" if exc.strerror and exc.filename else exc)
        return 1
    except Exception as exc:
        if args.debug:
            raise
        logger.error("Unexpected error (%s). Re-run with --debug for details.", type(exc).__name__)
        return 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
