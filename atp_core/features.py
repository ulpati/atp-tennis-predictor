"""Leakage-free feature construction.

Every match yields two rows: one from the winner's point of view (target 1)
and one from the loser's (target 0). Both rows come from the same
:func:`match_features` call with the players swapped, so every feature is
mirrored and none of them can reveal who won. All inputs are read from the
:class:`~atp_core.state.LeagueState` *before* the match result is applied.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import FeatureConfig
from .exceptions import DataValidationError
from .state import LeagueState
from .utils import safe_float

logger = logging.getLogger(__name__)

CONTEXT_FEATURES: tuple[str, ...] = ("surface_hard", "surface_clay", "surface_grass", "surface_carpet", "best_of_5")
PAIR_FEATURES: tuple[str, ...] = ("elo_prob", "elo_surface_prob", "h2h_p1", "h2h_p2", "h2h_diff", "h2h_surface_diff")
PLAYER_FEATURES: tuple[str, ...] = (
    "elo",
    "elo_surface",
    "rank_log",
    "points_log",
    "age",
    "height",
    "lefty",
    "experience_log",
    "winrate",
    "surface_winrate",
    "recent_form",
    "momentum",
    "form_std",
    "serve_won",
    "return_won",
    "ace_rate",
    "df_rate",
    "first_in",
    "bp_saved",
    "days_since_last",
    "matches_recent",
    "minutes_recent",
    "streak",
)
FEATURE_NAMES: tuple[str, ...] = (
    CONTEXT_FEATURES
    + PAIR_FEATURES
    + tuple(f"p1_{f}" for f in PLAYER_FEATURES)
    + tuple(f"p2_{f}" for f in PLAYER_FEATURES)
    + tuple(f"{f}_diff" for f in PLAYER_FEATURES)
)

_STAT_COLUMNS = ("svpt", "ace", "df", "1stIn", "1stWon", "2ndWon", "bpSaved", "bpFaced")


def context_features(surface: str, best_of: int) -> list[float]:
    return [
        float(surface == "Hard"),
        float(surface == "Clay"),
        float(surface == "Grass"),
        float(surface == "Carpet"),
        float(best_of == 5),
    ]


def match_features(
    state: LeagueState, player_1: str, player_2: str, surface: str, best_of: int, day: int
) -> list[float]:
    """Feature vector for ``player_1`` against ``player_2`` in :data:`FEATURE_NAMES` order."""
    s1 = state.snapshot(player_1, surface, day)
    s2 = state.snapshot(player_2, surface, day)
    pair = state.pair(player_1, player_2, surface)
    values = context_features(surface, best_of)
    values.extend(pair[name] for name in PAIR_FEATURES)
    values.extend(s1[name] for name in PLAYER_FEATURES)
    values.extend(s2[name] for name in PLAYER_FEATURES)
    values.extend(s1[name] - s2[name] for name in PLAYER_FEATURES)
    return values


@dataclass
class TrainingSet:
    """Feature matrix with one pair of mirrored rows per match."""

    X: np.ndarray
    """Shape ``(2 * n_matches, n_features)``; row ``2i`` is the winner's view, ``2i+1`` the loser's."""
    y: np.ndarray
    matches: pd.DataFrame
    """One row per emitted match: ``match_index``, ``date``, ``surface``, ``best_of``, ``winner_id``, ``loser_id``."""
    state: LeagueState


def to_epoch_day(dates: pd.Series | pd.Timestamp) -> np.ndarray | int:
    """Days since 1970-01-01 for a timestamp or a datetime series."""
    if isinstance(dates, pd.Series):
        return dates.to_numpy(dtype="datetime64[D]").astype(np.int64)
    return int(np.datetime64(pd.Timestamp(dates).date(), "D").astype(np.int64))


def from_epoch_day(day: int) -> pd.Timestamp:
    return pd.Timestamp(np.datetime64(int(day), "D"))


def default_train_start(df: pd.DataFrame, warmup_years: int = 3) -> pd.Timestamp:
    """First date that emits rows: a few years of warm-up when the data spans long enough."""
    first, last = df["tourney_date"].min(), df["tourney_date"].max()
    if (last - first).days >= 365 * (2 * warmup_years + 2):
        return pd.Timestamp(year=first.year + warmup_years, month=1, day=1)
    return first


def build_training_set(
    df: pd.DataFrame, config: FeatureConfig | None = None, train_start: str | pd.Timestamp | None = None
) -> TrainingSet:
    """Walk through the cleaned matches in order and build mirrored feature rows."""
    if df.empty:
        raise DataValidationError("No matches left after cleaning.")
    state = LeagueState(config)
    start = pd.Timestamp(train_start) if train_start is not None else default_train_start(df)
    emit = (df["tourney_date"] >= start).to_numpy()
    n_matches = int(emit.sum())
    if n_matches == 0:
        raise DataValidationError(f"No matches on or after {start.date()} to train on.")

    n_features = len(FEATURE_NAMES)
    X = np.empty((2 * n_matches, n_features), dtype=np.float32)
    y = np.zeros(2 * n_matches, dtype=np.int8)
    y[0::2] = 1

    days = to_epoch_day(df["tourney_date"])
    col = {
        c: df[c].to_numpy()
        for c in (
            "winner_id",
            "winner_name",
            "loser_id",
            "loser_name",
            "surface",
            "best_of",
            "minutes",
            "winner_hand",
            "loser_hand",
            "winner_rank",
            "winner_rank_points",
            "loser_rank",
            "loser_rank_points",
            "winner_age",
            "loser_age",
            "winner_ht",
            "loser_ht",
        )
    }
    stats_cols = {f"{side}_{s}": df[f"{side}_{s}"].to_numpy() for side in ("w", "l") for s in _STAT_COLUMNS}

    kept = np.flatnonzero(emit)
    row = 0
    for i in range(len(df)):
        w, lo = str(col["winner_id"][i]), str(col["loser_id"][i])
        surface, best_of, day = str(col["surface"][i]), int(col["best_of"][i]), int(days[i])
        for pid, side in ((w, "winner"), (lo, "loser")):
            state.observe_pre_match(
                pid,
                str(col[f"{side}_name"][i]),
                day=day,
                hand=str(col[f"{side}_hand"][i]),
                rank=safe_float(col[f"{side}_rank"][i]),
                points=safe_float(col[f"{side}_rank_points"][i]),
                age=safe_float(col[f"{side}_age"][i]),
                height=safe_float(col[f"{side}_ht"][i]),
            )
        if emit[i]:
            X[row] = match_features(state, w, lo, surface, best_of, day)
            X[row + 1] = match_features(state, lo, w, surface, best_of, day)
            row += 2
        stats = {name: safe_float(values[i]) for name, values in stats_cols.items()}
        state.record_result(w, lo, surface=surface, day=day, minutes=safe_float(col["minutes"][i]), stats=stats)

    matches = pd.DataFrame(
        {
            "match_index": df["match_index"].to_numpy()[kept],
            "date": df["tourney_date"].to_numpy()[kept],
            "surface": col["surface"][kept],
            "best_of": col["best_of"][kept],
            "winner_id": col["winner_id"][kept],
            "loser_id": col["loser_id"][kept],
        }
    )
    logger.info("Built %d rows from %d matches (training rows from %s)", len(X), n_matches, start.date())
    return TrainingSet(X=X, y=y, matches=matches, state=state)
