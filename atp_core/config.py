"""Configuration constants and option objects.

Importing this module has no side effects: it does not create directories,
open log files or read environment variables. Environment overrides are
read explicitly by the CLI through :func:`env_path`.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .exceptions import ConfigurationError

#: Surfaces present in the ATP data. Carpet has not been used on tour since 2009.
SURFACES: tuple[str, ...] = ("Hard", "Clay", "Grass", "Carpet")

DEFAULT_DATA_DIR = "data"
DEFAULT_MODELS_DIR = "models"
DEFAULT_SEED = 42

#: Jeff Sackmann's tennis_atp data, pinned to an immutable commit.
#: The original repository is no longer online; forks keep his full history,
#: and a commit SHA identifies the same content in every fork.
DATA_SOURCE_REPO = "Kadantte/tennis_atp"
DATA_SOURCE_COMMIT = "712be0c5ade693cdab9e69c23a71a0edf5a23c44"
DATA_SOURCE_HOST = "raw.githubusercontent.com"

#: Confidence labels for |p - 0.5|.
CONFIDENCE_THRESHOLDS = (("high", 0.20), ("medium", 0.10), ("low", 0.0))


def env_path(name: str, default: str) -> Path:
    """Return a directory path from an environment variable, or ``default`` when it is unset or empty."""
    return Path(os.environ.get(name, "").strip() or default)


@dataclass(frozen=True)
class FeatureConfig:
    """Options for the incremental feature builder."""

    recent_n: int = 20
    """Matches used for the recent-form features."""
    momentum_n: int = 10
    """Matches used for the short-term momentum feature."""
    stats_window: int = 30
    """Matches used for serve and return statistics."""
    time_decay: float = 0.0
    """Per-match exponential decay applied to recent form (0 disables it)."""
    fatigue_days: int = 14
    """Window, in days, for the workload features."""
    elo_k: float | None = None
    """Fixed Elo K-factor. ``None`` uses a dynamic K that shrinks with experience."""

    def validate(self) -> FeatureConfig:
        if not 1 <= self.recent_n <= 500:
            raise ConfigurationError("recent_n must be between 1 and 500.")
        if not 1 <= self.momentum_n <= self.recent_n:
            raise ConfigurationError("momentum_n must be between 1 and recent_n.")
        if not 1 <= self.stats_window <= 500:
            raise ConfigurationError("stats_window must be between 1 and 500.")
        if not 0.0 <= self.time_decay < 1.0:
            raise ConfigurationError("time_decay must be in [0, 1). Use 0 to disable it.")
        if not 1 <= self.fatigue_days <= 365:
            raise ConfigurationError("fatigue_days must be between 1 and 365.")
        if self.elo_k is not None and not 0.0 < self.elo_k <= 200.0:
            raise ConfigurationError("elo_k must be in (0, 200], or omitted for dynamic K.")
        return self

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrainConfig:
    """Options for model training and evaluation."""

    train_start: str | None = None
    """First match date that produces training rows; earlier matches only warm up the state."""
    test_start: str | None = None
    """First date of the test period. When omitted, ``test_fraction`` is used."""
    test_fraction: float = 0.10
    val_fraction: float = 0.10
    search_iter: int = 0
    """Random-search iterations over XGBoost hyper-parameters (0 keeps the defaults)."""
    cv_folds: int = 0
    """Expanding-window cross-validation folds on the pre-test period (0 disables it)."""
    per_surface: bool = False
    calibration: str = "none"
    refit: bool = True
    """After the evaluation, retrain the chosen models on every period (train, validation and test)."""
    use_gpu: bool = False
    threads: int | None = None
    seed: int = DEFAULT_SEED
    max_rounds: int = 3000
    early_stopping_rounds: int = 100
    params: dict = field(default_factory=dict)
    """Extra XGBoost parameters that override the defaults."""

    def validate(self) -> TrainConfig:
        if not 0.0 < self.test_fraction < 0.5:
            raise ConfigurationError("test_fraction must be in (0, 0.5).")
        if not 0.0 < self.val_fraction < 0.5:
            raise ConfigurationError("val_fraction must be in (0, 0.5).")
        if self.search_iter < 0 or self.search_iter > 500:
            raise ConfigurationError("search_iter must be between 0 and 500.")
        if self.cv_folds < 0 or self.cv_folds == 1 or self.cv_folds > 20:
            raise ConfigurationError("cv_folds must be 0 or between 2 and 20.")
        if self.calibration not in ("none", "isotonic"):
            raise ConfigurationError("calibration must be 'none' or 'isotonic'.")
        if self.calibration != "none" and self.refit:
            raise ConfigurationError(
                "Isotonic calibration is fitted on the validation period, which a refitted model trains on. "
                "Use --no-refit together with --calibration isotonic."
            )
        if self.threads is not None and not 1 <= self.threads <= 256:
            raise ConfigurationError("threads must be between 1 and 256.")
        if self.max_rounds < 10 or self.early_stopping_rounds < 1:
            raise ConfigurationError("max_rounds must be >= 10 and early_stopping_rounds >= 1.")
        return self

    def to_dict(self) -> dict:
        return asdict(self)
