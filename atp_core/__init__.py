"""ATP tennis match predictor: leakage-free features, XGBoost and Elo ratings."""

from __future__ import annotations

__version__ = "1.0.0"

from .exceptions import (
    ArtifactError,
    ATPError,
    ConfigurationError,
    DataLoadingError,
    DataValidationError,
    DownloadError,
    IntegrityError,
    InvalidSurfaceError,
    ModelTrainingError,
    PlayerNotFoundError,
)

__all__ = [
    "ATPError",
    "ArtifactError",
    "ConfigurationError",
    "DataLoadingError",
    "DataValidationError",
    "DownloadError",
    "IntegrityError",
    "InvalidSurfaceError",
    "ModelTrainingError",
    "PlayerNotFoundError",
    "__version__",
]
