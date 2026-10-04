"""Small, dependency-free helpers shared by the other modules."""

from __future__ import annotations

import hashlib
import math
import os
import re
import tempfile
import unicodedata
from collections.abc import Iterable
from pathlib import Path

_SPACES = re.compile(r"\s+")
_NAME_JUNK = re.compile(r"[^\w\s]")
_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
# C0 and C1 control characters except tab and newline.
_CONTROL_CHARS = {c: "?" for c in (*range(0x20), *range(0x7F, 0xA0)) if c not in (0x09, 0x0A)}


def normalize_name(name: str) -> str:
    """Return a comparison key for a player name.

    Accents and punctuation, hyphens included, are removed and case is folded, so that
    ``"Stan Wawrinka"``, ``"stan  wawrinka"`` and ``"Stan Wáwrinka"`` match, and so do
    ``"Félix Auger-Aliassime"`` and ``"Felix Auger Aliassime"``.
    """
    decomposed = unicodedata.normalize("NFKD", str(name))
    ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    cleaned = _NAME_JUNK.sub(" ", ascii_only.casefold())
    return _SPACES.sub(" ", cleaned).strip()


def safe_float(value: object, default: float = math.nan) -> float:
    """Convert ``value`` to ``float``; return ``default`` for missing or invalid input."""
    if value is None:
        return default
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return default if math.isnan(result) else result


def smoothed_rate(successes: float, total: float, prior: float = 1.0) -> float:
    """Laplace-smoothed rate ``(s + prior) / (n + 2 * prior)``; 0.5 with no data."""
    return (successes + prior) / (total + 2.0 * prior)


def weighted_mean(values: Iterable[float], decay: float = 0.0) -> float:
    """Mean of ``values`` (oldest first). With ``decay`` in (0, 1) newer values weigh more.

    The weight of the i-th most recent value is ``decay ** i``. Returns NaN
    for an empty input.
    """
    items = list(values)
    if not items:
        return math.nan
    if decay <= 0.0:
        return sum(items) / len(items)
    n = len(items)
    weights = [decay ** (n - 1 - i) for i in range(n)]
    return sum(w * v for w, v in zip(weights, items, strict=True)) / sum(weights)


def std_or_nan(values: Iterable[float]) -> float:
    """Population standard deviation, or NaN with fewer than two values."""
    items = list(values)
    if len(items) < 2:
        return math.nan
    mean = sum(items) / len(items)
    return math.sqrt(sum((v - mean) ** 2 for v in items) / len(items))


def sanitize_csv_cell(value: object) -> str:
    """Neutralise spreadsheet formula injection (CWE-1236) in CSV output."""
    text = "" if value is None else str(value)
    if text.startswith(_CSV_FORMULA_PREFIXES):
        return "'" + text
    return text


def printable(text: str) -> str:
    """Replace control characters with ``?``, so names read from data or batch files cannot send
    escape sequences to the terminal."""
    return text.translate(_CONTROL_CHARS)


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    """Write ``data`` to ``path`` atomically (temporary file in the same directory, then rename)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def atomic_write_text(path: str | os.PathLike[str], text: str) -> None:
    """Text variant of :func:`atomic_write_bytes` (UTF-8)."""
    atomic_write_bytes(path, text.encode("utf-8"))
