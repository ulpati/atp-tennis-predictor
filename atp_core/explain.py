"""Model explanations with XGBoost's built-in TreeSHAP (no extra dependency)."""

from __future__ import annotations

import numpy as np
import xgboost as xgb

from .features import FEATURE_NAMES
from .model import default_threads


def contributions(
    booster: xgb.Booster, X: np.ndarray, best_iteration: int, names: tuple[str, ...] = FEATURE_NAMES
) -> np.ndarray:
    """SHAP values in log-odds, shape ``(n_rows, n_features + 1)``; the last column is the bias."""
    dmatrix = xgb.DMatrix(
        np.asarray(X, dtype=np.float32), feature_names=list(names), missing=np.nan, nthread=default_threads()
    )
    return booster.predict(dmatrix, pred_contribs=True, iteration_range=(0, best_iteration + 1))


def mean_abs_shap(
    booster: xgb.Booster,
    X: np.ndarray,
    best_iteration: int,
    max_rows: int = 2000,
    seed: int = 0,
    names: tuple[str, ...] = FEATURE_NAMES,
) -> list[tuple[str, float]]:
    """Mean absolute SHAP value per feature, largest first."""
    rows = np.asarray(X)
    if len(rows) > max_rows:
        rows = rows[np.random.default_rng(seed).choice(len(rows), max_rows, replace=False)]
    values = np.abs(contributions(booster, rows, best_iteration, names)[:, :-1]).mean(axis=0)
    return sorted(zip(names, map(float, values), strict=True), key=lambda item: item[1], reverse=True)


def gain_importance(booster: xgb.Booster, names: tuple[str, ...] = FEATURE_NAMES) -> list[tuple[str, float]]:
    """Share of total gain per feature, largest first. Unused features get 0."""
    scores = booster.get_score(importance_type="total_gain")
    total = sum(scores.values()) or 1.0
    return sorted(((n, scores.get(n, 0.0) / total) for n in names), key=lambda item: item[1], reverse=True)
