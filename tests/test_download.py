import io
import urllib.error

import pytest

from atp_core import download as dl
from atp_core.config import DATA_SOURCE_COMMIT
from atp_core.exceptions import ConfigurationError, DownloadError

HEADER = (
    "tourney_id,tourney_name,surface,draw_size,tourney_level,tourney_date,match_num,winner_id,winner_name,"
    "loser_id,loser_name,score,best_of,round\n"
)
VALID = (HEADER + "2020-1,X,Hard,32,A,20200101,1,1,A B,2,C D,6-1 6-1,3,R32\n").encode()
OTHER_COMMIT = "a" * 40


def test_known_checksums_match_the_pinned_commit():
    known = dl.known_checksums()
    assert known["commit"] == DATA_SOURCE_COMMIT
    assert "atp_matches_2024.csv" in known["files"]
    assert all(len(v) == 64 for v in known["files"].values())


@pytest.mark.parametrize(
    ("year", "repo", "commit"),
    [
        (1900, "a/b", DATA_SOURCE_COMMIT),
        (2020, "bad repo", DATA_SOURCE_COMMIT),
        (2020, "a/b", "master"),
        (2020, "../../x", DATA_SOURCE_COMMIT),
    ],
)
def test_build_url_validation(year, repo, commit):
    with pytest.raises(ConfigurationError):
        dl.build_url(year, repo, commit)


def test_build_url_is_https_and_pinned():
    url = dl.build_url(2020)
    assert url.startswith("https://raw.githubusercontent.com/") and DATA_SOURCE_COMMIT in url


@pytest.mark.parametrize("url", ["http://example.com/x.csv", "file:///etc/passwd", "ftp://x/y"])
def test_fetch_requires_https(url):
    with pytest.raises(DownloadError, match="HTTPS"):
        dl.fetch(url)


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Opener:
    def __init__(self, outcome):
        self.outcome = outcome

    def open(self, request, timeout):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return _Response(self.outcome)


@pytest.fixture
def fake_network(monkeypatch):
    def install(outcome):
        monkeypatch.setattr(dl.urllib.request, "build_opener", lambda *handlers: _Opener(outcome))

    return install


def test_fetch_outcomes(fake_network):
    fake_network(b"data")
    assert dl.fetch("https://h/x") == b"data"
    fake_network(urllib.error.HTTPError("https://h/x", 404, "nf", {}, None))
    assert dl.fetch("https://h/x") is None
    fake_network(urllib.error.HTTPError("https://h/x", 500, "err", {}, None))
    with pytest.raises(DownloadError, match="HTTP 500"):
        dl.fetch("https://h/x")
    fake_network(urllib.error.URLError("no route"))
    with pytest.raises(DownloadError, match="Cannot download"):
        dl.fetch("https://h/x")
    fake_network(b"x" * 20)
    with pytest.raises(DownloadError, match="larger"):
        dl.fetch("https://h/x", limit=10)


def test_redirects_must_stay_on_https_and_same_host():
    handler = dl._SameHostRedirect()

    class Request:
        full_url = "https://raw.githubusercontent.com/a"

    with pytest.raises(DownloadError, match="another host"):
        handler.redirect_request(Request(), None, 302, "", {}, "https://evil.example/a")
    with pytest.raises(DownloadError, match="non-HTTPS"):
        handler.redirect_request(Request(), None, 302, "", {}, "http://raw.githubusercontent.com/a")


def test_download_writes_validated_files(fake_network, tmp_path):
    fake_network(VALID)
    written = dl.download(tmp_path, 2020, 2021, commit=OTHER_COMMIT)
    assert [p.name for p in written] == ["atp_matches_2020.csv", "atp_matches_2021.csv"]
    assert dl.download(tmp_path, 2020, 2020, commit=OTHER_COMMIT) == []  # already present
    assert dl.download(tmp_path, 2020, 2020, commit=OTHER_COMMIT, force=True)


@pytest.mark.parametrize("source", [{"repo": "evil/../x"}, {"commit": "main"}])
def test_invalid_source_is_refused_before_touching_the_disk(fake_network, tmp_path, source):
    fake_network(VALID)
    dl.download(tmp_path, 2020, 2020, commit=OTHER_COMMIT)
    with pytest.raises(ConfigurationError):
        dl.download(tmp_path, 2020, 2020, **{"commit": OTHER_COMMIT, **source})  # even if the file exists
    with pytest.raises(ConfigurationError):
        dl.download(tmp_path / "new", 2020, 2020, **{"commit": OTHER_COMMIT, **source})
    assert not (tmp_path / "new").exists()


def test_download_verifies_known_checksums(fake_network, tmp_path):
    fake_network(VALID)
    with pytest.raises(DownloadError, match="SHA-256"):
        dl.download(tmp_path, 2020, 2020)
    assert not (tmp_path / "atp_matches_2020.csv").exists()
    with pytest.raises(DownloadError, match="No known checksum"):
        dl.download(tmp_path, 2099, 2099)


def test_files_already_present_are_checked_against_known_checksums(fake_network, tmp_path, monkeypatch, caplog):
    name = "atp_matches_2020.csv"
    pinned = {"commit": DATA_SOURCE_COMMIT, "files": {name: dl.sha256_bytes(VALID)}}
    monkeypatch.setattr(dl, "known_checksums", lambda: pinned)
    fake_network(VALID)
    (tmp_path / name).write_bytes(b"tampered")
    assert [p.name for p in dl.download(tmp_path, 2020, 2020)] == [name]
    assert (tmp_path / name).read_bytes() == VALID
    assert "does not match its known SHA-256" in caplog.text
    assert dl.download(tmp_path, 2020, 2020) == []  # verified, so kept


def test_download_rejects_unexpected_content(fake_network, tmp_path):
    fake_network(b"<html>not a csv</html>")
    with pytest.raises(DownloadError, match="does not look like"):
        dl.download(tmp_path, 2020, 2020, commit=OTHER_COMMIT)
    fake_network(b"\xff\xfe\x00bad")
    with pytest.raises(DownloadError, match="UTF-8"):
        dl.download(tmp_path, 2020, 2020, commit=OTHER_COMMIT)


def test_download_skips_missing_years_and_validates_range(fake_network, tmp_path):
    fake_network(urllib.error.HTTPError("https://h/x", 404, "nf", {}, None))
    assert dl.download(tmp_path, 2020, 2020, commit=OTHER_COMMIT) == []
    with pytest.raises(ConfigurationError):
        dl.download(tmp_path, 2021, 2020)


def test_cli_download(fake_network, tmp_path, capsys):
    from atp_core.cli import main

    fake_network(VALID)
    code = main(
        ["download", "--data-dir", str(tmp_path), "--from-year", "2020", "--to-year", "2020", "--commit", OTHER_COMMIT]
    )
    assert code == 0 and "CC BY-NC-SA" in capsys.readouterr().out
