"""Exception hierarchy for the ATP predictor.

Every error that is the user's to fix (bad input, missing data, tampered
artifacts) derives from :class:`ATPError`, so the CLI can print a clean
message instead of a stack trace.
"""

from __future__ import annotations


class ATPError(Exception):
    """Base class for all expected, user-facing errors."""


class ConfigurationError(ATPError):
    """An option or environment variable has an invalid value."""


class DataLoadingError(ATPError):
    """Match files are missing or cannot be read."""


class DataValidationError(ATPError):
    """Match files do not have the expected structure."""


class DownloadError(ATPError):
    """Downloading the match data failed or returned unexpected content."""


class ModelTrainingError(ATPError):
    """Training could not be completed with the given data and options."""


class ArtifactError(ATPError):
    """A saved model run is missing, malformed or incompatible."""


class IntegrityError(ArtifactError):
    """A saved file does not match the SHA-256 recorded in the manifest."""


class InvalidSurfaceError(ATPError):
    """The requested court surface is not supported."""


class PlayerNotFoundError(ATPError):
    """A player name could not be resolved unambiguously."""

    def __init__(self, query: str, suggestions: list[str] | None = None, reason: str = "not found") -> None:
        self.query = query
        self.suggestions = list(suggestions or [])
        self.reason = reason
        message = f"Player '{query}' {reason}."
        if self.suggestions:
            message += " Did you mean: " + ", ".join(self.suggestions) + "?"
        super().__init__(message)
