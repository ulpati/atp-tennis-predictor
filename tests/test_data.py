import numpy as np
import pandas as pd
import pytest
from conftest import write_csvs

from atp_core import data as data_module
from atp_core.data import clean_matches, find_match_files, load_matches, normalize_surface, summarize
from atp_core.exceptions import DataLoadingError, DataValidationError


def test_find_match_files_filters_names_years_and_symlinks(data_dir, tmp_path):
    files = find_match_files(data_dir)
    assert [f.name for f in files] == [f"atp_matches_{y}.csv" for y in range(2014, 2020)]
    selected = find_match_files(data_dir, (2016, 2017))
    assert [f.name for f in selected] == ["atp_matches_2016.csv", "atp_matches_2017.csv"]
    (tmp_path / "atp_matches_2020.csv").symlink_to(files[0])
    (tmp_path / "notes.csv").write_text("x")
    with pytest.raises(DataLoadingError):
        find_match_files(tmp_path)


def test_find_match_files_missing_directory(tmp_path):
    with pytest.raises(DataLoadingError, match="not found"):
        find_match_files(tmp_path / "missing")


def test_load_matches_rejects_wrong_header(tmp_path):
    (tmp_path / "atp_matches_2020.csv").write_text("a,b,c\n1,2,3\n")
    with pytest.raises(DataValidationError, match="missing required columns"):
        load_matches(tmp_path)


def test_load_matches_rejects_oversized_files(data_dir, monkeypatch):
    monkeypatch.setattr(data_module, "MAX_FILE_BYTES", 10)
    with pytest.raises(DataValidationError, match="larger than"):
        load_matches(data_dir)


def test_clean_matches_drops_walkovers_and_normalises(raw_matches):
    df = clean_matches(raw_matches)
    assert len(df) == len(raw_matches) - 1  # the walkover
    assert set(df["surface"]) == {"Hard", "Clay", "Grass", "Unknown"}
    assert df["match_index"].tolist() == list(range(len(df)))
    assert df["tourney_date"].is_monotonic_increasing


def test_clean_matches_orders_rounds_inside_a_tournament(raw_matches):
    tournament = raw_matches[raw_matches["tourney_id"] == "2014-002"].copy()
    tournament["match_num"] = tournament["match_num"].to_numpy()[::-1]  # final gets the smallest number
    rounds = clean_matches(tournament)["round"].tolist()
    assert rounds == ["R16"] * 8 + ["QF"] * 4 + ["SF"] * 2 + ["F"]


def test_clean_matches_coerces_bad_values(raw_matches):
    raw = raw_matches.head(3).copy()
    raw.loc[raw.index[0], "w_ace"] = -4
    raw.loc[raw.index[1], "best_of"] = 7
    raw.loc[raw.index[2], "winner_id"] = np.nan
    raw.loc[raw.index[2], "surface"] = "Hardcourt"
    df = clean_matches(raw)
    assert np.isnan(df.loc[0, "w_ace"])
    assert set(df["best_of"]) <= {3, 5}
    assert df["winner_id"].str.startswith("name:").any()
    assert "Hard" in set(df["surface"])


def test_clean_matches_requires_columns():
    with pytest.raises(DataValidationError):
        clean_matches(pd.DataFrame({"winner_name": ["a"]}))


def test_normalize_surface():
    values = pd.Series(["Hard", "clay", "Grasscourt", "Carpet", None, "Sand"])
    assert normalize_surface(values).tolist() == ["Hard", "Clay", "Grass", "Carpet", "Unknown", "Unknown"]


def test_summarize(raw_matches, tmp_path):
    df = clean_matches(load_matches(write_csvs(raw_matches, tmp_path)))
    summary = summarize(df)
    assert summary["matches"] == len(df)
    assert summary["first_date"] == "2014-01-06"
    assert summarize(df.iloc[0:0]) == {"matches": 0}
