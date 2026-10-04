"""Elo ratings for tennis players, overall and per surface.

With ``k=None`` the K-factor shrinks as a player gains experience,
``K = 250 / (matches + 5) ** 0.4``, the variant popularised by
FiveThirtyEight's tennis model. A fixed ``k`` reproduces classic Elo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

BASE_RATING = 1500.0
_DYNAMIC_NUMERATOR = 250.0
_DYNAMIC_OFFSET = 5.0
_DYNAMIC_SHAPE = 0.4


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability that a player rated ``rating_a`` beats one rated ``rating_b``."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


@dataclass
class EloEngine:
    """Tracks overall and per-surface Elo ratings."""

    k: float | None = None
    base: float = BASE_RATING
    ratings: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    surface_ratings: dict[str, dict[str, float]] = field(default_factory=dict)
    surface_counts: dict[str, dict[str, int]] = field(default_factory=dict)

    def k_factor(self, matches_played: int) -> float:
        """K-factor for a player with ``matches_played`` previous matches."""
        if self.k is not None:
            return float(self.k)
        return _DYNAMIC_NUMERATOR / (matches_played + _DYNAMIC_OFFSET) ** _DYNAMIC_SHAPE

    def rating(self, player: str, surface: str | None = None) -> float:
        """Overall rating, or the rating on ``surface`` (the base rating if never played there)."""
        if surface is not None:
            return self.surface_ratings.get(surface, {}).get(player, self.base)
        return self.ratings.get(player, self.base)

    def matches(self, player: str, surface: str | None = None) -> int:
        """Matches counted for the overall rating, or on ``surface``."""
        if surface is not None:
            return self.surface_counts.get(surface, {}).get(player, 0)
        return self.counts.get(player, 0)

    def expected(self, player_a: str, player_b: str, surface: str | None = None) -> float:
        """Probability that ``player_a`` beats ``player_b``."""
        return expected_score(self.rating(player_a, surface), self.rating(player_b, surface))

    def update(self, winner: str, loser: str, surface: str | None = None) -> None:
        """Apply the result of one match to the overall and, if known, the surface ratings."""
        self._update_pool(self.ratings, self.counts, winner, loser)
        if surface is not None and surface != "Unknown":
            ratings = self.surface_ratings.setdefault(surface, {})
            counts = self.surface_counts.setdefault(surface, {})
            self._update_pool(ratings, counts, winner, loser)

    def _update_pool(self, ratings: dict[str, float], counts: dict[str, int], winner: str, loser: str) -> None:
        r_w = ratings.get(winner, self.base)
        r_l = ratings.get(loser, self.base)
        expected_w = expected_score(r_w, r_l)
        ratings[winner] = r_w + self.k_factor(counts.get(winner, 0)) * (1.0 - expected_w)
        ratings[loser] = r_l - self.k_factor(counts.get(loser, 0)) * (1.0 - expected_w)
        counts[winner] = counts.get(winner, 0) + 1
        counts[loser] = counts.get(loser, 0) + 1

    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "base": self.base,
            "ratings": self.ratings,
            "counts": self.counts,
            "surface_ratings": self.surface_ratings,
            "surface_counts": self.surface_counts,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EloEngine:
        k = data.get("k")
        return cls(
            k=None if k is None else float(k),
            base=float(data.get("base", BASE_RATING)),
            ratings={str(p): float(r) for p, r in data.get("ratings", {}).items()},
            counts={str(p): int(n) for p, n in data.get("counts", {}).items()},
            surface_ratings={
                str(s): {str(p): float(r) for p, r in pool.items()}
                for s, pool in data.get("surface_ratings", {}).items()
            },
            surface_counts={
                str(s): {str(p): int(n) for p, n in pool.items()} for s, pool in data.get("surface_counts", {}).items()
            },
        )
