import math
import os

import pytest

from atp_core.utils import (
    atomic_write_text,
    normalize_name,
    safe_float,
    sanitize_csv_cell,
    sha256_bytes,
    smoothed_rate,
    std_or_nan,
    weighted_mean,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Stan Wawrinka", "stan wawrinka"),
        ("  STAN   wáwrinka ", "stan wawrinka"),
        ("J.-W. Tsonga", "j w tsonga"),
        ("Félix Auger-Aliassime", "felix auger aliassime"),
    ],
)
def test_normalize_name(raw, expected):
    assert normalize_name(raw) == expected


@pytest.mark.parametrize(("value", "expected"), [("3.5", 3.5), (2, 2.0), ("x", None), (None, None), (math.nan, None)])
def test_safe_float(value, expected):
    result = safe_float(value)
    assert math.isnan(result) if expected is None else result == expected


def test_smoothed_rate_is_neutral_without_data():
    assert smoothed_rate(0, 0) == 0.5
    assert smoothed_rate(9, 10) == pytest.approx(10 / 12)


def test_weighted_mean_favours_recent_values():
    assert math.isnan(weighted_mean([]))
    assert weighted_mean([0, 0, 1, 1]) == 0.5
    assert weighted_mean([0, 0, 1, 1], decay=0.5) > weighted_mean([1, 1, 0, 0], decay=0.5)


def test_std_or_nan():
    assert math.isnan(std_or_nan([1.0]))
    assert std_or_nan([0, 1]) == pytest.approx(0.5)


@pytest.mark.parametrize("cell", ["=HYPERLINK(1)", "+1", "-2", "@SUM(A1)", "\tx"])
def test_sanitize_csv_cell_neutralises_formulas(cell):
    assert sanitize_csv_cell(cell).startswith("'")


def test_sanitize_csv_cell_keeps_plain_text():
    assert sanitize_csv_cell("Jannik Sinner") == "Jannik Sinner"
    assert sanitize_csv_cell(None) == ""


def test_hashes_and_atomic_write(tmp_path):
    target = tmp_path / "sub" / "file.txt"
    atomic_write_text(target, "first")
    atomic_write_text(target, "second")
    assert target.read_text() == "second"
    assert sha256_bytes(target.read_bytes()) == sha256_bytes(b"second")
    assert os.listdir(target.parent) == ["file.txt"]
