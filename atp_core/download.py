"""Download of the ATP match files with integrity checks.

Only HTTPS is allowed, redirects must stay on the same host, responses are
size-limited, the CSV header is validated, and files from the pinned default
commit are verified against the SHA-256 values shipped with the package.
Files are written atomically, so an interrupted download never leaves a
truncated CSV behind.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from importlib import resources
from pathlib import Path

from .config import DATA_SOURCE_COMMIT, DATA_SOURCE_HOST, DATA_SOURCE_REPO
from .data import REQUIRED_COLUMNS
from .exceptions import ConfigurationError, DownloadError
from .utils import atomic_write_bytes, sha256_bytes

logger = logging.getLogger(__name__)

MAX_DOWNLOAD_BYTES = 32 * 1024 * 1024
REPO_RE = re.compile(r"^[A-Za-z0-9-]{1,39}/[A-Za-z0-9._-]{1,100}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
USER_AGENT = "atp-tennis-predictor (+https://github.com/ulpati/atp-tennis-predictor)"


def known_checksums() -> dict:
    """SHA-256 of the files at the pinned commit, shipped as package data."""
    text = resources.files("atp_core").joinpath("data_checksums.json").read_text(encoding="utf-8")
    return json.loads(text)


def build_url(year: int, repo: str = DATA_SOURCE_REPO, commit: str = DATA_SOURCE_COMMIT) -> str:
    if not REPO_RE.match(repo):
        raise ConfigurationError(f"Invalid repository {repo!r}; expected 'owner/name'.")
    if not COMMIT_RE.match(commit):
        raise ConfigurationError("The commit must be a full 40-character SHA, so the content is immutable.")
    if not 1968 <= year <= 2100:
        raise ConfigurationError(f"Year out of range: {year}.")
    return f"https://{DATA_SOURCE_HOST}/{repo}/{commit}/atp_matches_{year}.csv"


class _SameHostRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).hostname != urllib.parse.urlsplit(req.full_url).hostname:
            raise DownloadError(f"Refusing redirect to another host: {newurl}")
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise DownloadError("Refusing redirect to a non-HTTPS URL.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch(url: str, timeout: float = 30.0, limit: int = MAX_DOWNLOAD_BYTES) -> bytes | None:
    """GET ``url`` over HTTPS. Returns ``None`` for 404, raises :class:`DownloadError` otherwise."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or not parts.hostname:
        raise DownloadError(f"Only HTTPS URLs are allowed: {url}")
    opener = urllib.request.build_opener(_SameHostRedirect())
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310 - HTTPS checked above
    try:
        with opener.open(request, timeout=timeout) as response:
            data = response.read(limit + 1)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise DownloadError(f"HTTP {exc.code} for {url}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DownloadError(f"Cannot download {url}: {exc}") from exc
    if len(data) > limit:
        raise DownloadError(f"{url} is larger than {limit // 2**20} MB.")
    return data


def validate_csv(data: bytes, name: str) -> None:
    try:
        header = next(csv.reader(io.StringIO(data[:4096].decode("utf-8", errors="strict").split("\n", 1)[0])))
    except (UnicodeDecodeError, StopIteration, csv.Error) as exc:
        raise DownloadError(f"{name} is not a valid UTF-8 CSV file.") from exc
    missing = [c for c in REQUIRED_COLUMNS if c not in {h.strip() for h in header}]
    if missing:
        raise DownloadError(f"{name} does not look like an ATP match file (missing {', '.join(missing)}).")


def download(
    data_dir: str | Path,
    from_year: int,
    to_year: int,
    *,
    repo: str = DATA_SOURCE_REPO,
    commit: str = DATA_SOURCE_COMMIT,
    force: bool = False,
    timeout: float = 30.0,
) -> list[Path]:
    """Download ``atp_matches_YYYY.csv`` for each year; returns the files written."""
    if from_year > to_year:
        raise ConfigurationError("from_year must not be greater than to_year.")
    for year in (from_year, to_year):
        build_url(year, repo, commit)  # validates repository, commit and years before any disk or network access
    target = Path(data_dir)
    target.mkdir(parents=True, exist_ok=True)
    checksums = known_checksums()
    verify = commit == checksums.get("commit")
    if not verify:
        logger.warning("Commit %s is not the pinned one: files cannot be checked against known SHA-256 values.", commit)
    written, kept = [], 0
    for year in range(from_year, to_year + 1):
        name = f"atp_matches_{year}.csv"
        path = target / name
        expected = checksums.get("files", {}).get(name) if verify else None
        if path.exists() and not force:
            # A file from the pinned commit is kept only if it still matches its known SHA-256.
            if expected is None or sha256_bytes(path.read_bytes()) == expected:
                logger.debug("%s already present, skipping", name)
                kept += 1
                continue
            logger.warning("%s does not match its known SHA-256; downloading it again.", name)
        data = fetch(build_url(year, repo, commit), timeout=timeout)
        if data is None:
            logger.warning("%s is not available at this commit, skipping", name)
            continue
        validate_csv(data, name)
        if verify and expected is None:
            raise DownloadError(f"No known checksum for {name} at the pinned commit.")
        if expected is not None and sha256_bytes(data) != expected:
            raise DownloadError(f"{name} does not match its known SHA-256. The download was discarded.")
        atomic_write_bytes(path, data)
        written.append(path)
        logger.info("Downloaded %s (%d KB)", name, len(data) // 1024)
    if kept:
        logger.info("Kept %d file(s) already present; use --force to download them again.", kept)
    return written
