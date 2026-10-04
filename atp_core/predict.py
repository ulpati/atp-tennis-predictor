"""Match prediction from a loaded run, with strict player-name resolution."""

from __future__ import annotations

import difflib
import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .artifacts import Run
from .config import CONFIDENCE_THRESHOLDS, SURFACES
from .exceptions import ConfigurationError, InvalidSurfaceError, PlayerNotFoundError
from .explain import contributions
from .features import FEATURE_NAMES, from_epoch_day, match_features, to_epoch_day
from .model import predict_rows
from .utils import normalize_name

logger = logging.getLogger(__name__)

AUTO_ACCEPT_RATIO = 0.90
SUGGESTION_CUTOFF = 0.60
INACTIVE_DAYS = 365
ACTIVE_WINDOW_DAYS = 730
"""A partial name matching several players resolves to the only one active in the last two years of data."""


class PlayerIndex:
    """Resolves user-typed names to player ids without silently guessing."""

    def __init__(self, names: dict[str, str], last_days: dict[str, int | None]) -> None:
        self.names = names
        self.last_days = last_days
        known_days = [d for d in last_days.values() if d is not None]
        self.reference_day = max(known_days) if known_days else None
        self.by_key: dict[str, list[str]] = {}
        for pid, name in names.items():
            self.by_key.setdefault(normalize_name(name), []).append(pid)

    @classmethod
    def from_run(cls, run: Run) -> PlayerIndex:
        players = run.state.players
        return cls({pid: rec.name for pid, rec in players.items()}, {pid: rec.last_day for pid, rec in players.items()})

    def _most_recent(self, ids: list[str]) -> list[str]:
        return sorted(ids, key=lambda pid: self.last_days.get(pid) or -1, reverse=True)

    def _label(self, pid: str) -> str:
        return self.names[pid]

    def _is_active(self, pid: str) -> bool:
        last = self.last_days.get(pid)
        return last is not None and self.reference_day is not None and last >= self.reference_day - ACTIVE_WINDOW_DAYS

    def resolve(self, query: str) -> str:
        key = normalize_name(query)
        if not key:
            raise PlayerNotFoundError(query, reason="is empty")
        if key in self.by_key:
            return self._most_recent(self.by_key[key])[0]

        tokens = key.split()
        partial = [k for k in self.by_key if all(t in k.split() for t in tokens)]
        if len(partial) == 1:
            return self._most_recent(self.by_key[partial[0]])[0]
        if len(partial) > 1:
            ranked = self._most_recent([pid for k in partial for pid in self.by_key[k]])
            active = [pid for pid in ranked if self._is_active(pid)]
            if len(active) == 1:
                others = ", ".join(self._label(p) for p in ranked if p != active[0])
                logger.warning(
                    "Interpreting '%s' as '%s', the only active player among: %s.",
                    query,
                    self._label(active[0]),
                    others,
                )
                return active[0]
            raise PlayerNotFoundError(query, [self._label(p) for p in ranked[:5]], reason="is ambiguous")

        scored = sorted(((difflib.SequenceMatcher(None, key, k).ratio(), k) for k in self.by_key), reverse=True)[:5]
        if scored and scored[0][0] >= AUTO_ACCEPT_RATIO and (len(scored) == 1 or scored[1][0] < AUTO_ACCEPT_RATIO):
            pid = self._most_recent(self.by_key[scored[0][1]])[0]
            logger.warning("Interpreting '%s' as '%s'.", query, self._label(pid))
            return pid
        suggestions = [self._label(self._most_recent(self.by_key[k])[0]) for r, k in scored if r >= SUGGESTION_CUTOFF]
        raise PlayerNotFoundError(query, suggestions)


def confidence_label(probability: float) -> str:
    margin = abs(probability - 0.5)
    for label, threshold in CONFIDENCE_THRESHOLDS:
        if margin > threshold:
            return label
    return "low"


@dataclass
class Prediction:
    player_a: str
    player_b: str
    surface: str
    best_of: int
    as_of: str
    probability_a: float
    raw_a_vs_b: float
    raw_b_vs_a: float
    elo_probability_a: float
    model: str
    confidence: str
    warnings: list[str]
    explanation: list[dict] | None = None

    @property
    def probability_b(self) -> float:
        return 1.0 - self.probability_a

    @property
    def winner(self) -> str:
        return self.player_a if self.probability_a >= 0.5 else self.player_b

    def to_dict(self) -> dict:
        data = dict(self.__dict__)
        data["probability_b"] = self.probability_b
        data["predicted_winner"] = self.winner
        return data


class Predictor:
    """Predicts matches with the boosters and player state of a :class:`Run`."""

    def __init__(self, run: Run) -> None:
        self.run = run
        self.state = run.state
        self.index = PlayerIndex.from_run(run)

    def resolve(self, name: str) -> str:
        return self.index.resolve(name)

    def _row(self, a: str, b: str, surface: str, best_of: int, day: int) -> np.ndarray:
        return np.asarray([match_features(self.state, a, b, surface, best_of, day)], dtype=np.float32)

    def predict(
        self, name_a: str, name_b: str, surface: str, best_of: int = 3, as_of: str | None = None, explain: bool = False
    ) -> Prediction:
        surface = surface.strip().capitalize()
        if surface not in SURFACES:
            raise InvalidSurfaceError(f"Unknown surface {surface!r}. Choose one of: {', '.join(SURFACES)}.")
        if best_of not in (3, 5):
            raise ConfigurationError("best_of must be 3 or 5.")
        a, b = self.resolve(name_a), self.resolve(name_b)
        if a == b:
            raise PlayerNotFoundError(name_b, reason="is the same player as the first one")
        last_day = self.state.last_day or 0
        day = to_epoch_day(pd.Timestamp(as_of)) if as_of else last_day
        if day < last_day:
            raise ConfigurationError(
                f"The date must not be earlier than the end of the data ({from_epoch_day(last_day).date()}): "
                "the model only knows each player's latest state."
            )

        key = self.run.model_key(surface)
        booster, best = self.run.boosters[key], self.run.best_iterations[key]
        x_ab, x_ba = self._row(a, b, surface, best_of, day), self._row(b, a, surface, best_of, day)
        p_ab = float(predict_rows(booster, x_ab, best)[0])
        p_ba = float(predict_rows(booster, x_ba, best)[0])
        probability = (p_ab + 1.0 - p_ba) / 2.0
        if self.run.calibrator is not None:
            probability = float(self.run.calibrator(probability))

        warnings = []
        for pid in (a, b):
            record = self.state.players[pid]
            if record.last_day is not None and day - record.last_day > INACTIVE_DAYS:
                warnings.append(f"{record.name} has not played since {from_epoch_day(record.last_day).date()}.")
            if record.wins + record.losses < 20:
                warnings.append(f"{record.name} has only {record.wins + record.losses} matches in the data.")

        explanation = None
        if explain:
            contrib = contributions(booster, np.vstack([x_ab, x_ba]), best)
            symmetric = (contrib[0, :-1] - contrib[1, :-1]) / 2.0
            order = np.argsort(-np.abs(symmetric))[:10]
            explanation = [
                {"feature": FEATURE_NAMES[i], "value_a_vs_b": float(x_ab[0, i]), "log_odds": float(symmetric[i])}
                for i in order
                if not math.isclose(symmetric[i], 0.0)
            ]

        return Prediction(
            player_a=self.state.players[a].name,
            player_b=self.state.players[b].name,
            surface=surface,
            best_of=best_of,
            as_of=from_epoch_day(day).date().isoformat(),
            probability_a=probability,
            raw_a_vs_b=p_ab,
            raw_b_vs_a=p_ba,
            elo_probability_a=float(self.state.pair(a, b, surface)["elo_prob"]),
            model=key,
            confidence=confidence_label(probability),
            warnings=warnings,
            explanation=explanation,
        )
