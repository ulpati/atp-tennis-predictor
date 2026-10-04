"""Loading, validation and cleaning of Jeff Sackmann's ``atp_matches_YYYY.csv`` files."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SURFACES
from .exceptions import DataLoadingError, DataValidationError
from .utils import normalize_name

logger = logging.getLogger(__name__)

MATCH_FILE_RE = re.compile(r"^atp_matches_(\d{4})\.csv$")

REQUIRED_COLUMNS: tuple[str, ...] = (
    "tourney_id",
    "tourney_name",
    "surface",
    "tourney_level",
    "tourney_date",
    "match_num",
    "winner_id",
    "winner_name",
    "loser_id",
    "loser_name",
    "score",
    "best_of",
    "round",
)

NUMERIC_COLUMNS: tuple[str, ...] = (
    "match_num",
    "best_of",
    "minutes",
    "winner_rank",
    "winner_rank_points",
    "loser_rank",
    "loser_rank_points",
    "winner_age",
    "loser_age",
    "winner_ht",
    "loser_ht",
    "w_ace",
    "w_df",
    "w_svpt",
    "w_1stIn",
    "w_1stWon",
    "w_2ndWon",
    "w_SvGms",
    "w_bpSaved",
    "w_bpFaced",
    "l_ace",
    "l_df",
    "l_svpt",
    "l_1stIn",
    "l_1stWon",
    "l_2ndWon",
    "l_SvGms",
    "l_bpSaved",
    "l_bpFaced",
)

COUNT_COLUMNS: tuple[str, ...] = tuple(c for c in NUMERIC_COLUMNS if c.startswith(("w_", "l_")))

#: Order of rounds inside a tournament: every tournament shares one start date,
#: so this order is what keeps the processing strictly chronological.
ROUND_ORDER: dict[str, int] = {
    "Q1": 0,
    "Q2": 1,
    "Q3": 2,
    "Q4": 3,
    "ER": 4,
    "R128": 5,
    "R64": 6,
    "R32": 7,
    "R16": 8,
    "RR": 9,
    "QF": 10,
    "SF": 11,
    "BR": 12,
    "F": 13,
}

_NOT_PLAYED = re.compile(r"W/O|WALKOVER|\bDEF\b|DEF\.", re.IGNORECASE)
MAX_FILE_BYTES = 64 * 1024 * 1024


def find_match_files(data_dir: str | Path, years: tuple[int, int] | None = None) -> list[Path]:
    """Return the ``atp_matches_YYYY.csv`` files in ``data_dir``, oldest first."""
    directory = Path(data_dir)
    if not directory.is_dir():
        raise DataLoadingError(f"Data directory not found: {directory}. Run 'atp download' first.")
    files = []
    for path in directory.iterdir():
        match = MATCH_FILE_RE.match(path.name)
        if not match or not path.is_file() or path.is_symlink():
            continue
        year = int(match.group(1))
        if years and not years[0] <= year <= years[1]:
            continue
        files.append((year, path))
    if not files:
        raise DataLoadingError(f"No atp_matches_YYYY.csv files in {directory}. Run 'atp download' first.")
    return [path for _, path in sorted(files)]


def load_matches(data_dir: str | Path, years: tuple[int, int] | None = None) -> pd.DataFrame:
    """Read and concatenate the match files, validating their header."""
    frames = []
    for path in find_match_files(data_dir, years):
        if path.stat().st_size > MAX_FILE_BYTES:
            raise DataValidationError(f"{path.name} is larger than {MAX_FILE_BYTES // 2**20} MB; refusing to load it.")
        try:
            frame = pd.read_csv(path, dtype=str, keep_default_na=True, encoding="utf-8", on_bad_lines="error")
        except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
            raise DataLoadingError(f"Cannot read {path.name}: {exc}") from exc
        frame.columns = [c.strip() for c in frame.columns]
        missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
        if missing:
            raise DataValidationError(f"{path.name} is missing required columns: {', '.join(missing)}")
        frames.append(frame)
        logger.debug("Read %s (%d rows)", path.name, len(frame))
    raw = pd.concat(frames, ignore_index=True, sort=False)
    logger.info("Loaded %d rows from %d files", len(raw), len(frames))
    return raw


def normalize_surface(values: pd.Series) -> pd.Series:
    """Map raw surface labels to one of :data:`SURFACES` or ``"Unknown"``."""
    lowered = values.fillna("").astype(str).str.strip().str.lower()
    result = pd.Series("Unknown", index=values.index, dtype=object)
    for surface in SURFACES:
        result[lowered.str.contains(surface.lower(), regex=False)] = surface
    return result


def clean_matches(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned, chronologically ordered copy of the raw matches.

    Walkovers and defaults are removed because no tennis was played, rows
    without a date or players are dropped, numeric columns are coerced and
    impossible negative counts become missing values.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {', '.join(missing)}")
    df = raw.copy()
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce") if column in df.columns else np.nan
    for column in COUNT_COLUMNS:
        df.loc[df[column] < 0, column] = np.nan

    df["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(str).str[:8], format="%Y%m%d", errors="coerce")
    for side in ("winner", "loser"):
        df[f"{side}_name"] = df[f"{side}_name"].fillna("").astype(str).str.strip()
        ids = df[f"{side}_id"].fillna("").astype(str).str.strip()
        fallback = "name:" + df[f"{side}_name"].map(normalize_name)
        df[f"{side}_id"] = ids.where(ids != "", fallback)
        df[f"{side}_hand"] = df.get(f"{side}_hand", pd.Series("", index=df.index)).fillna("U").astype(str)

    before = len(df)
    df = df[df["tourney_date"].notna() & (df["winner_name"] != "") & (df["loser_name"] != "")]
    df = df[df["winner_id"] != df["loser_id"]]
    not_played = df["score"].fillna("").astype(str).str.contains(_NOT_PLAYED)
    df = df[~not_played]
    logger.info("Dropped %d rows (missing data, walkovers, defaults)", before - len(df))

    df["surface"] = normalize_surface(df["surface"])
    df["best_of"] = df["best_of"].where(df["best_of"].isin([3, 5]), 3).astype(int)
    df["round_order"] = df["round"].map(ROUND_ORDER).fillna(len(ROUND_ORDER)).astype(int)
    df = df.sort_values(["tourney_date", "tourney_id", "round_order", "match_num"], kind="mergesort").reset_index(
        drop=True
    )
    df["match_index"] = np.arange(len(df), dtype=np.int64)
    return df


def summarize(df: pd.DataFrame) -> dict:
    """Describe the cleaned dataset for logs and manifests."""
    if df.empty:
        return {"matches": 0}
    players = pd.unique(pd.concat([df["winner_id"], df["loser_id"]], ignore_index=True))
    return {
        "matches": len(df),
        "players": len(players),
        "first_date": df["tourney_date"].min().date().isoformat(),
        "last_date": df["tourney_date"].max().date().isoformat(),
        "surfaces": {k: int(v) for k, v in df["surface"].value_counts().items()},
    }
