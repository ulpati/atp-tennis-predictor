"""Model training and evaluation with a strictly chronological protocol.

* The matches are split by date into train, validation and test periods.
  The test period is never used for any decision: not for early stopping,
  not for hyper-parameter search, not for calibration, not for routing.
* Early stopping, random search, optional calibration and the choice between
  global and per-surface models all use the validation period only.
* Metrics are computed per match on the symmetric probability
  ``(p(A beats B) + 1 - p(B beats A)) / 2``, the same quantity ``predict`` returns.
* After the evaluation, the chosen models are retrained on every period with
  the same settings (``refit``), so predictions also learn from the most
  recent matches. The reported metrics always come from the evaluated models.
"""

from __future__ import annotations

import itertools
import logging
import math
import os
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from .config import TrainConfig
from .exceptions import ModelTrainingError
from .features import FEATURE_NAMES, TrainingSet

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: dict = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "learning_rate": 0.03,
    "max_depth": 5,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "gamma": 0.0,
}

SEARCH_SPACE: dict[str, list] = {
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "min_child_weight": [1, 3, 5, 10, 20],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
    "reg_alpha": [0.0, 0.1, 0.5, 1.0],
    "gamma": [0.0, 0.1, 0.5, 1.0],
}

MIN_MATCHES_PER_SPLIT = 50
MIN_MATCHES_PER_SURFACE_MODEL = 2000
TEST_SAMPLE_MATCHES = 3000
_EPS = 1e-6


# --------------------------------------------------------------------------- splitting
@dataclass(frozen=True)
class Split:
    """Positions (into ``TrainingSet.matches``) of the matches in each period."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def describe(self, matches: pd.DataFrame) -> dict:
        def period(pos: np.ndarray) -> dict:
            dates = matches["date"].to_numpy()[pos]
            return {
                "matches": len(pos),
                "from": pd.Timestamp(dates.min()).date().isoformat(),
                "to": pd.Timestamp(dates.max()).date().isoformat(),
            }

        return {"train": period(self.train), "validation": period(self.val), "test": period(self.test)}


def _boundary(sorted_dates: np.ndarray, fraction_before: float) -> pd.Timestamp:
    index = min(max(int(len(sorted_dates) * fraction_before), 0), len(sorted_dates) - 1)
    return pd.Timestamp(sorted_dates[index])


def chronological_split(matches: pd.DataFrame, cfg: TrainConfig) -> Split:
    """Split matches by date. A tournament start date never spans two periods."""
    dates = matches["date"].to_numpy(dtype="datetime64[ns]")
    sorted_dates = np.sort(dates)
    if cfg.test_start:
        test_start = pd.Timestamp(cfg.test_start)
        pre_test = sorted_dates[sorted_dates < np.datetime64(test_start)]
        if len(pre_test) == 0:
            raise ModelTrainingError(f"No matches before test start {test_start.date()}.")
        share = cfg.val_fraction / (1.0 - cfg.test_fraction)
        val_start = _boundary(pre_test, 1.0 - share)
    else:
        test_start = _boundary(sorted_dates, 1.0 - cfg.test_fraction)
        val_start = _boundary(sorted_dates, 1.0 - cfg.test_fraction - cfg.val_fraction)
    test_start_np, val_start_np = np.datetime64(test_start), np.datetime64(val_start)
    split = Split(
        train=np.flatnonzero(dates < val_start_np),
        val=np.flatnonzero((dates >= val_start_np) & (dates < test_start_np)),
        test=np.flatnonzero(dates >= test_start_np),
    )
    for name, positions in (("train", split.train), ("validation", split.val), ("test", split.test)):
        if len(positions) < MIN_MATCHES_PER_SPLIT:
            raise ModelTrainingError(
                f"The {name} period has only {len(positions)} matches; need at least {MIN_MATCHES_PER_SPLIT}. "
                "Load more years of data or change the split options."
            )
    return split


def rows_for(match_positions: np.ndarray) -> np.ndarray:
    """Row indices (winner row, then loser row) for the given match positions."""
    positions = np.asarray(match_positions, dtype=np.int64)
    return np.stack([2 * positions, 2 * positions + 1], axis=1).ravel()


# --------------------------------------------------------------------------- metrics
def winner_probabilities(p_rows: np.ndarray) -> np.ndarray:
    """Symmetric probability that the actual winner wins, one value per match."""
    return (p_rows[0::2] + 1.0 - p_rows[1::2]) / 2.0


def match_metrics(p_winner: np.ndarray) -> dict:
    """Accuracy, log loss, Brier score and AUC from per-match winner probabilities."""
    p = np.clip(np.asarray(p_winner, dtype=np.float64), _EPS, 1.0 - _EPS)
    n = len(p)
    if n == 0:
        return {"matches": 0}
    correct = np.where(np.isclose(p, 0.5), 0.5, (p > 0.5).astype(float))
    y_sym = np.concatenate([np.ones(n), np.zeros(n)])
    p_sym = np.concatenate([p, 1.0 - p])
    return {
        "matches": int(n),
        "accuracy": float(correct.mean()),
        "log_loss": float(-np.log(p).mean()),
        "brier": float(((1.0 - p) ** 2).mean()),
        "auc": float(roc_auc_score(y_sym, p_sym)),
    }


def baseline_metrics(X_rows: np.ndarray, names: tuple[str, ...] = FEATURE_NAMES) -> dict:
    """Simple reference predictors on the same matches: ranking and Elo alone."""
    index = {name: i for i, name in enumerate(names)}
    winner_rows = X_rows[0::2]
    r_w = winner_rows[:, index["p1_rank_log"]].astype(np.float64)
    r_l = winner_rows[:, index["p2_rank_log"]].astype(np.float64)
    known = np.isfinite(r_w) & np.isfinite(r_l)
    rank_correct = np.where(np.isclose(r_w, r_l), 0.5, (r_w < r_l).astype(float))
    return {
        "ranking": {
            "matches": int(known.sum()),
            "accuracy": float(rank_correct[known].mean()) if known.any() else math.nan,
        },
        "elo": match_metrics(winner_rows[:, index["elo_prob"]].astype(np.float64)),
        "surface_elo": match_metrics(winner_rows[:, index["elo_surface_prob"]].astype(np.float64)),
    }


# --------------------------------------------------------------------------- calibration
@dataclass(frozen=True)
class Calibrator:
    """Isotonic mapping stored as plain arrays (no pickling needed)."""

    x: tuple[float, ...]
    y: tuple[float, ...]

    def __call__(self, p: np.ndarray | float) -> np.ndarray:
        values = np.asarray(p, dtype=np.float64)
        forward = np.interp(values, self.x, self.y)
        backward = np.interp(1.0 - values, self.x, self.y)
        return (forward + 1.0 - backward) / 2.0  # keeps f(1 - p) = 1 - f(p)

    def to_dict(self) -> dict:
        return {"x": list(self.x), "y": list(self.y)}

    @classmethod
    def from_dict(cls, data: dict) -> Calibrator:
        x, y = tuple(float(v) for v in data["x"]), tuple(float(v) for v in data["y"])
        if len(x) != len(y) or len(x) < 2 or any(b < a for a, b in itertools.pairwise(x)):
            raise ValueError("Invalid calibrator thresholds.")
        return cls(x=x, y=y)


def fit_calibrator(p_winner: np.ndarray) -> Calibrator:
    p = np.asarray(p_winner, dtype=np.float64)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=_EPS, y_max=1.0 - _EPS)
    iso.fit(np.concatenate([p, 1.0 - p]), np.concatenate([np.ones(len(p)), np.zeros(len(p))]))
    return Calibrator(x=tuple(map(float, iso.X_thresholds_)), y=tuple(map(float, iso.y_thresholds_)))


# --------------------------------------------------------------------------- xgboost helpers
def default_threads() -> int:
    return max(1, min(os.cpu_count() or 1, 8))


def resolve_device(use_gpu: bool) -> str:
    """Return ``"cuda"`` when a working CUDA build and GPU are available, else ``"cpu"``."""
    if not use_gpu:
        return "cpu"
    if not xgb.build_info().get("USE_CUDA", False):
        logger.warning("This XGBoost build has no CUDA support; training on CPU.")
        return "cpu"
    try:
        probe = xgb.DMatrix(np.random.default_rng(0).random((32, 3)), label=np.arange(32) % 2)
        xgb.train({"device": "cuda", "tree_method": "hist", "verbosity": 0}, probe, num_boost_round=1)
    except xgb.core.XGBoostError as exc:
        logger.warning("CUDA device not usable (%s); training on CPU.", exc)
        return "cpu"
    logger.info("Training on the CUDA device.")
    return "cuda"


def _dmatrix(X: np.ndarray, y: np.ndarray | None, names: tuple[str, ...], threads: int) -> xgb.DMatrix:
    return xgb.DMatrix(X, label=y, feature_names=list(names), missing=np.nan, nthread=threads)


def _full_params(params: dict, cfg: TrainConfig, device: str, threads: int) -> dict:
    return {**DEFAULT_PARAMS, **params, "seed": cfg.seed, "device": device, "nthread": threads, "verbosity": 0}


@dataclass
class Fit:
    booster: xgb.Booster
    best_iteration: int
    val_log_loss: float


def fit_booster(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict,
    cfg: TrainConfig,
    device: str,
    names: tuple[str, ...] = FEATURE_NAMES,
) -> Fit:
    threads = cfg.threads or default_threads()
    dtrain = _dmatrix(X_train, y_train, names, threads)
    dval = _dmatrix(X_val, y_val, names, threads)
    booster = xgb.train(
        _full_params(params, cfg, device, threads),
        dtrain,
        num_boost_round=cfg.max_rounds,
        evals=[(dval, "validation")],
        early_stopping_rounds=cfg.early_stopping_rounds,
        verbose_eval=False,
    )
    return Fit(booster=booster, best_iteration=int(booster.best_iteration), val_log_loss=float(booster.best_score))


def fit_rounds(
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    rounds: int,
    cfg: TrainConfig,
    device: str,
    names: tuple[str, ...] = FEATURE_NAMES,
) -> xgb.Booster:
    """Train exactly ``rounds`` trees without a validation set (used by the final refit)."""
    threads = cfg.threads or default_threads()
    return xgb.train(
        _full_params(params, cfg, device, threads),
        _dmatrix(X, y, names, threads),
        num_boost_round=rounds,
        verbose_eval=False,
    )


def predict_rows(
    booster: xgb.Booster,
    X: np.ndarray,
    best_iteration: int,
    names: tuple[str, ...] = FEATURE_NAMES,
    threads: int | None = None,
) -> np.ndarray:
    dmatrix = _dmatrix(np.asarray(X, dtype=np.float32), None, names, threads or default_threads())
    return booster.predict(dmatrix, iteration_range=(0, best_iteration + 1)).astype(np.float64)


# --------------------------------------------------------------------------- search and CV
def random_search(X: np.ndarray, y: np.ndarray, split: Split, cfg: TrainConfig, device: str) -> tuple[dict, list]:
    """Pick hyper-parameters by validation log loss. The test period is not touched."""
    rng = random.Random(cfg.seed)  # noqa: S311 - reproducible sampling, not security-related  # nosec B311
    tr, va = rows_for(split.train), rows_for(split.val)
    best_params, best_loss, history = dict(cfg.params), math.inf, []
    candidates = [dict(cfg.params)] + [
        {k: rng.choice(v) for k, v in SEARCH_SPACE.items()} for _ in range(cfg.search_iter)
    ]
    for i, params in enumerate(candidates):
        fit = fit_booster(X[tr], y[tr], X[va], y[va], params, cfg, device)
        history.append({"params": params, "val_log_loss": fit.val_log_loss, "rounds": fit.best_iteration + 1})
        logger.info("Search %d/%d: validation log loss %.5f", i, len(candidates) - 1, fit.val_log_loss)
        if fit.val_log_loss < best_loss:
            best_params, best_loss = params, fit.val_log_loss
    return best_params, history


def cross_validate(
    X: np.ndarray, y: np.ndarray, matches: pd.DataFrame, split: Split, params: dict, cfg: TrainConfig, device: str
) -> list[dict]:
    """Expanding-window CV on the pre-test period; each fold early-stops on its own tail."""
    pre_test = np.concatenate([split.train, split.val])
    dates = matches["date"].to_numpy(dtype="datetime64[ns]")[pre_test]
    order = pre_test[np.argsort(dates, kind="stable")]
    unique_dates = np.unique(dates)
    edges = [unique_dates[int(len(unique_dates) * k / (cfg.cv_folds + 1))] for k in range(1, cfg.cv_folds + 1)]
    sorted_dates = np.sort(dates)
    results = []
    for k, start in enumerate(edges, 1):
        end = edges[k] if k < len(edges) else None
        fold_train = order[sorted_dates < start]
        mask_eval = (sorted_dates >= start) if end is None else ((sorted_dates >= start) & (sorted_dates < end))
        fold_eval = order[mask_eval]
        cut = int(len(fold_train) * 0.9)
        inner_train, inner_val = fold_train[:cut], fold_train[cut:]
        if min(len(inner_train), len(inner_val), len(fold_eval)) < MIN_MATCHES_PER_SPLIT:
            continue
        fit = fit_booster(
            X[rows_for(inner_train)],
            y[rows_for(inner_train)],
            X[rows_for(inner_val)],
            y[rows_for(inner_val)],
            params,
            cfg,
            device,
        )
        p = winner_probabilities(predict_rows(fit.booster, X[rows_for(fold_eval)], fit.best_iteration))
        metrics = match_metrics(p)
        metrics.update({"fold": k, "from": pd.Timestamp(start).date().isoformat()})
        results.append(metrics)
        logger.info("CV fold %d: accuracy %.4f, log loss %.4f", k, metrics["accuracy"], metrics["log_loss"])
    return results


# --------------------------------------------------------------------------- training entry point
@dataclass
class TrainResult:
    boosters: dict[str, xgb.Booster]
    best_iterations: dict[str, int]
    routing: dict[str, str]
    calibrator: Calibrator | None
    params: dict
    metrics: dict
    split: dict
    cv: list[dict] = field(default_factory=list)
    search: list[dict] = field(default_factory=list)
    test_sample: dict[str, np.ndarray] = field(default_factory=dict)
    refit: dict = field(default_factory=dict)
    """Matches and rounds of each refitted model; empty when the evaluated models were kept."""


def _routed_probabilities(
    boosters: dict[str, xgb.Booster],
    best: dict[str, int],
    routing: dict[str, str],
    X_rows: np.ndarray,
    surfaces: np.ndarray,
) -> np.ndarray:
    """Row probabilities, each match scored by the model its surface is routed to."""
    row_keys = np.array([routing.get(s, "all") for s in np.repeat(surfaces, 2)])
    p = np.empty(len(X_rows), dtype=np.float64)
    for key in np.unique(row_keys):
        mask = row_keys == key
        p[mask] = predict_rows(boosters[key], X_rows[mask], best[key])
    return p


def refit_all_periods(
    X: np.ndarray,
    y: np.ndarray,
    matches: pd.DataFrame,
    best: dict[str, int],
    params: dict,
    cfg: TrainConfig,
    device: str,
) -> tuple[dict[str, xgb.Booster], dict[str, int], dict]:
    """Retrain each chosen model on all matches of its scope, with the rounds found by early stopping.

    Nothing is chosen here: parameters, rounds and routing were all fixed on the
    validation period, and the metrics were computed before this step.
    """
    surfaces = matches["surface"].to_numpy()
    dates = matches["date"].to_numpy(dtype="datetime64[ns]")
    boosters, new_best, models = {}, {}, {}
    for key, best_iteration in best.items():
        positions = np.arange(len(matches)) if key == "all" else np.flatnonzero(surfaces == key)
        rows = rows_for(positions)
        rounds = best_iteration + 1
        boosters[key] = fit_rounds(X[rows], y[rows], params, rounds, cfg, device)
        new_best[key] = rounds - 1
        models[key] = {"matches": len(positions), "rounds": rounds}
        logger.info("Refitted the %s model on %d matches with %d rounds", key, len(positions), rounds)
    period = {
        "from": pd.Timestamp(dates.min()).date().isoformat(),
        "to": pd.Timestamp(dates.max()).date().isoformat(),
    }
    return boosters, new_best, {**period, "models": models}


def train(ts: TrainingSet, cfg: TrainConfig) -> TrainResult:
    """Train, select and evaluate a model on a :class:`TrainingSet`."""
    cfg = cfg.validate()
    X, y, matches = ts.X, ts.y, ts.matches
    split = chronological_split(matches, cfg)
    logger.info("Split: %s", split.describe(matches))
    device = resolve_device(cfg.use_gpu)

    search_history: list[dict] = []
    params = dict(cfg.params)
    if cfg.search_iter:
        params, search_history = random_search(X, y, split, cfg, device)

    tr, va, te = rows_for(split.train), rows_for(split.val), rows_for(split.test)
    fit = fit_booster(X[tr], y[tr], X[va], y[va], params, cfg, device)
    boosters, best = {"all": fit.booster}, {"all": fit.best_iteration}
    logger.info("Global model: %d rounds, validation log loss %.5f", fit.best_iteration + 1, fit.val_log_loss)

    surfaces = matches["surface"].to_numpy()
    routing: dict[str, str] = {}
    surface_choice: dict[str, dict] = {}
    if cfg.per_surface:
        p_val_global = winner_probabilities(predict_rows(fit.booster, X[va], fit.best_iteration))
        for surface in ("Hard", "Clay", "Grass"):
            s_train = split.train[surfaces[split.train] == surface]
            s_val = split.val[surfaces[split.val] == surface]
            if len(s_train) < MIN_MATCHES_PER_SURFACE_MODEL or len(s_val) < MIN_MATCHES_PER_SPLIT:
                continue
            s_fit = fit_booster(
                X[rows_for(s_train)], y[rows_for(s_train)], X[rows_for(s_val)], y[rows_for(s_val)], params, cfg, device
            )
            p_val_surface = winner_probabilities(predict_rows(s_fit.booster, X[rows_for(s_val)], s_fit.best_iteration))
            global_loss = match_metrics(p_val_global[surfaces[split.val] == surface])["log_loss"]
            surface_loss = match_metrics(p_val_surface)["log_loss"]
            chosen = surface if surface_loss < global_loss else "all"
            surface_choice[surface] = {
                "validation_log_loss_global": global_loss,
                "validation_log_loss_surface": surface_loss,
                "chosen": chosen,
            }
            if chosen == surface:
                boosters[surface], best[surface] = s_fit.booster, s_fit.best_iteration
                routing[surface] = surface
            logger.info(
                "%s: surface model %.5f vs global %.5f on validation, using %s",
                surface,
                surface_loss,
                global_loss,
                chosen,
            )

    p_val = winner_probabilities(_routed_probabilities(boosters, best, routing, X[va], surfaces[split.val]))
    calibrator = fit_calibrator(p_val) if cfg.calibration == "isotonic" else None

    p_test_raw = winner_probabilities(_routed_probabilities(boosters, best, routing, X[te], surfaces[split.test]))
    p_test = calibrator(p_test_raw) if calibrator else p_test_raw
    test_surfaces = surfaces[split.test]
    metrics = {
        "validation": match_metrics(p_val),
        "test": match_metrics(p_test),
        "test_by_surface": {
            s: match_metrics(p_test[test_surfaces == s])
            for s in ("Hard", "Clay", "Grass", "Carpet")
            if (test_surfaces == s).sum() >= MIN_MATCHES_PER_SPLIT
        },
        "baselines": baseline_metrics(X[te]),
        "surface_models": surface_choice,
    }
    if calibrator:
        metrics["test_uncalibrated"] = match_metrics(p_test_raw)

    cv_results = cross_validate(X, y, matches, split, params, cfg, device) if cfg.cv_folds else []

    refit: dict = {}
    if cfg.refit:
        boosters, best, refit = refit_all_periods(X, y, matches, best, params, cfg, device)

    rng = np.random.default_rng(cfg.seed)
    sample = np.sort(rng.choice(split.test, size=min(TEST_SAMPLE_MATCHES, len(split.test)), replace=False))
    sample_rows = rows_for(sample)
    return TrainResult(
        boosters=boosters,
        best_iterations=best,
        routing=routing,
        calibrator=calibrator,
        params={**DEFAULT_PARAMS, **params},
        metrics=metrics,
        split=split.describe(matches),
        cv=cv_results,
        search=search_history,
        test_sample={"X": X[sample_rows], "y": y[sample_rows], "surface": surfaces[sample].astype(str)},
        refit=refit,
    )
