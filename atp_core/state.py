"""Running per-player state shared by training and prediction.

The feature builder walks through the matches in chronological order. For
each match it first records what is known *before* the match (ranking,
age, height), then reads a snapshot of both players, and only afterwards
applies the result. Prediction reads snapshots from the final state with the
same code, so training and serving features cannot diverge.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from .config import SURFACES, FeatureConfig
from .elo import EloEngine
from .utils import smoothed_rate, std_or_nan, weighted_mean

#: Serve tuple layout stored for every match with statistics.
_SVPT, _ACE, _DF, _FIRST_IN, _FIRST_WON, _SECOND_WON, _BP_SAVED, _BP_FACED, _OPP_SVPT, _OPP_WON = range(10)
SCHEDULE_LEN = 64
MAX_DAYS_SINCE_LAST = 730.0


def _nan_to_none(value: float) -> float | None:
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else value


def _none_to_nan(value: float | None) -> float:
    return math.nan if value is None else float(value)


@dataclass
class PlayerRecord:
    """Everything the model needs to know about one player."""

    name: str
    recent_n: int
    stats_window: int
    hand: str = "U"
    wins: int = 0
    losses: int = 0
    surface_wins: dict[str, int] = field(default_factory=dict)
    surface_losses: dict[str, int] = field(default_factory=dict)
    results: deque = field(init=False)
    serve: deque = field(init=False)
    schedule: deque = field(init=False)
    last_day: int | None = None
    streak: int = 0
    rank: float = math.nan
    points: float = math.nan
    age: float = math.nan
    age_day: int | None = None
    height: float = math.nan

    def __post_init__(self) -> None:
        self.results = deque(maxlen=self.recent_n)
        self.serve = deque(maxlen=self.stats_window)
        self.schedule = deque(maxlen=SCHEDULE_LEN)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "hand": self.hand,
            "wins": self.wins,
            "losses": self.losses,
            "surface_wins": self.surface_wins,
            "surface_losses": self.surface_losses,
            "results": list(self.results),
            "serve": [[_nan_to_none(v) for v in row] for row in self.serve],
            "schedule": [[d, _nan_to_none(m)] for d, m in self.schedule],
            "last_day": self.last_day,
            "streak": self.streak,
            "rank": _nan_to_none(self.rank),
            "points": _nan_to_none(self.points),
            "age": _nan_to_none(self.age),
            "age_day": self.age_day,
            "height": _nan_to_none(self.height),
        }

    @classmethod
    def from_dict(cls, data: dict, recent_n: int, stats_window: int) -> PlayerRecord:
        record = cls(name=str(data["name"]), recent_n=recent_n, stats_window=stats_window)
        record.hand = str(data.get("hand", "U"))
        record.wins = int(data.get("wins", 0))
        record.losses = int(data.get("losses", 0))
        record.surface_wins = {str(k): int(v) for k, v in data.get("surface_wins", {}).items()}
        record.surface_losses = {str(k): int(v) for k, v in data.get("surface_losses", {}).items()}
        record.results.extend(int(r) for r in data.get("results", []))
        record.serve.extend(tuple(_none_to_nan(v) for v in row) for row in data.get("serve", []))
        record.schedule.extend((int(d), _none_to_nan(m)) for d, m in data.get("schedule", []))
        record.last_day = None if data.get("last_day") is None else int(data["last_day"])
        record.streak = int(data.get("streak", 0))
        record.rank = _none_to_nan(data.get("rank"))
        record.points = _none_to_nan(data.get("points"))
        record.age = _none_to_nan(data.get("age"))
        record.age_day = None if data.get("age_day") is None else int(data["age_day"])
        record.height = _none_to_nan(data.get("height"))
        return record


def _serve_tuple(stats: dict, prefix: str, opponent: str) -> tuple | None:
    svpt = stats.get(f"{prefix}_svpt", math.nan)
    if not svpt or math.isnan(svpt) or svpt <= 0:
        return None
    opp_svpt = stats.get(f"{opponent}_svpt", math.nan)
    opp_won = stats.get(f"{opponent}_1stWon", math.nan) + stats.get(f"{opponent}_2ndWon", math.nan)
    return (
        svpt,
        stats.get(f"{prefix}_ace", math.nan),
        stats.get(f"{prefix}_df", math.nan),
        stats.get(f"{prefix}_1stIn", math.nan),
        stats.get(f"{prefix}_1stWon", math.nan),
        stats.get(f"{prefix}_2ndWon", math.nan),
        stats.get(f"{prefix}_bpSaved", math.nan),
        stats.get(f"{prefix}_bpFaced", math.nan),
        opp_svpt,
        opp_won,
    )


def _ratio(rows: list[tuple], num: tuple[int, ...], den: int) -> float:
    total_num = 0.0
    total_den = 0.0
    for row in rows:
        values = [row[i] for i in num]
        if any(math.isnan(v) for v in values) or math.isnan(row[den]) or row[den] <= 0:
            continue
        total_num += sum(values)
        total_den += row[den]
    return total_num / total_den if total_den > 0 else math.nan


class LeagueState:
    """Ratings, records, head-to-heads and schedules of every player seen so far."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = (config or FeatureConfig()).validate()
        self.players: dict[str, PlayerRecord] = {}
        self.elo = EloEngine(k=self.config.elo_k)
        self.h2h: dict[str, dict[str, int]] = {}
        self.last_day: int | None = None

    # ------------------------------------------------------------------ updates
    def player(self, pid: str, name: str) -> PlayerRecord:
        record = self.players.get(pid)
        if record is None:
            record = PlayerRecord(name=name, recent_n=self.config.recent_n, stats_window=self.config.stats_window)
            self.players[pid] = record
        return record

    def observe_pre_match(
        self,
        pid: str,
        name: str,
        *,
        day: int,
        hand: str = "U",
        rank: float = math.nan,
        points: float = math.nan,
        age: float = math.nan,
        height: float = math.nan,
    ) -> None:
        """Record information that is public before the match starts."""
        record = self.player(pid, name)
        record.name = name or record.name
        if hand in ("L", "R"):
            record.hand = hand
        record.rank = rank
        record.points = points
        if not math.isnan(age):
            record.age = age
            record.age_day = day
        if not math.isnan(height) and height > 0:
            record.height = height

    def record_result(
        self, winner: str, loser: str, *, surface: str, day: int, minutes: float = math.nan, stats: dict | None = None
    ) -> None:
        """Apply a match result. Call only after the features of the match were read."""
        stats = stats or {}
        w = self.players[winner]
        lo = self.players[loser]
        w.wins += 1
        lo.losses += 1
        w.surface_wins[surface] = w.surface_wins.get(surface, 0) + 1
        lo.surface_losses[surface] = lo.surface_losses.get(surface, 0) + 1
        w.results.append(1)
        lo.results.append(0)
        w.streak = w.streak + 1 if w.streak > 0 else 1
        lo.streak = lo.streak - 1 if lo.streak < 0 else -1
        for record, prefix, opponent in ((w, "w", "l"), (lo, "l", "w")):
            serve = _serve_tuple(stats, prefix, opponent)
            if serve is not None:
                record.serve.append(serve)
            record.schedule.append((day, minutes))
            record.last_day = day
        self.elo.update(winner, loser, surface)
        key = f"{winner}|{loser}"
        counts = self.h2h.setdefault(key, {})
        counts["all"] = counts.get("all", 0) + 1
        counts[surface] = counts.get(surface, 0) + 1
        self.last_day = day if self.last_day is None else max(self.last_day, day)

    # ------------------------------------------------------------------ reads
    def h2h_wins(self, player: str, opponent: str, surface: str = "all") -> int:
        return self.h2h.get(f"{player}|{opponent}", {}).get(surface, 0)

    def snapshot(self, pid: str, surface: str, day: int) -> dict[str, float]:
        """Pre-match description of a player as of ``day`` (days since 1970-01-01)."""
        cfg = self.config
        r = self.players[pid]
        played = r.wins + r.losses
        s_wins = r.surface_wins.get(surface, 0)
        s_played = s_wins + r.surface_losses.get(surface, 0)
        results = list(r.results)
        serve = list(r.serve)
        recent_schedule = [(d, m) for d, m in r.schedule if 0 <= day - d <= cfg.fatigue_days]
        minutes = [m for _, m in recent_schedule if not math.isnan(m)]
        surface_known = surface in SURFACES
        days_since = math.nan if r.last_day is None else float(min(max(day - r.last_day, 0), MAX_DAYS_SINCE_LAST))
        age = r.age + (day - r.age_day) / 365.25 if r.age_day is not None and not math.isnan(r.age) else math.nan
        return {
            "elo": self.elo.rating(pid),
            "elo_surface": self.elo.rating(pid, surface if surface_known else None),
            "rank_log": math.log(r.rank) if r.rank > 0 else math.nan,
            "points_log": math.log1p(r.points) if r.points >= 0 else math.nan,
            "age": age,
            "height": r.height,
            "lefty": 1.0 if r.hand == "L" else (0.0 if r.hand == "R" else math.nan),
            "experience_log": math.log1p(played),
            "winrate": smoothed_rate(r.wins, played),
            "surface_winrate": smoothed_rate(s_wins, s_played) if surface_known else math.nan,
            "recent_form": weighted_mean(results, cfg.time_decay),
            "momentum": weighted_mean(results[-cfg.momentum_n :]),
            "form_std": std_or_nan(results),
            "serve_won": _ratio(serve, (_FIRST_WON, _SECOND_WON), _SVPT),
            "return_won": 1.0 - _ratio(serve, (_OPP_WON,), _OPP_SVPT),
            "ace_rate": _ratio(serve, (_ACE,), _SVPT),
            "df_rate": _ratio(serve, (_DF,), _SVPT),
            "first_in": _ratio(serve, (_FIRST_IN,), _SVPT),
            "bp_saved": _ratio(serve, (_BP_SAVED,), _BP_FACED),
            "days_since_last": days_since,
            "matches_recent": float(len(recent_schedule)),
            "minutes_recent": float(sum(minutes)) if minutes else (0.0 if not recent_schedule else math.nan),
            "streak": float(r.streak),
        }

    def pair(self, player_a: str, player_b: str, surface: str) -> dict[str, float]:
        """Features that describe the matchup rather than a single player."""
        surface_known = surface in SURFACES
        elo_prob = self.elo.expected(player_a, player_b)
        elo_surface_prob = self.elo.expected(player_a, player_b, surface) if surface_known else elo_prob
        h_ab = self.h2h_wins(player_a, player_b)
        h_ba = self.h2h_wins(player_b, player_a)
        hs_ab = self.h2h_wins(player_a, player_b, surface)
        hs_ba = self.h2h_wins(player_b, player_a, surface)
        return {
            "elo_prob": elo_prob,
            "elo_surface_prob": elo_surface_prob,
            "h2h_p1": float(h_ab),
            "h2h_p2": float(h_ba),
            "h2h_diff": float(h_ab - h_ba),
            "h2h_surface_diff": float(hs_ab - hs_ba),
        }

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "players": {pid: rec.to_dict() for pid, rec in self.players.items()},
            "elo": self.elo.to_dict(),
            "h2h": self.h2h,
            "last_day": self.last_day,
        }

    @classmethod
    def from_dict(cls, data: dict) -> LeagueState:
        state = cls(FeatureConfig(**data.get("config", {})))
        cfg = state.config
        state.players = {
            str(pid): PlayerRecord.from_dict(rec, cfg.recent_n, cfg.stats_window)
            for pid, rec in data.get("players", {}).items()
        }
        state.elo = EloEngine.from_dict(data.get("elo", {}))
        state.h2h = {str(k): {str(s): int(n) for s, n in v.items()} for k, v in data.get("h2h", {}).items()}
        state.last_day = None if data.get("last_day") is None else int(data["last_day"])
        return state
