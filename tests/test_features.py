import numpy as np
import pandas as pd
import pytest

from atp_core.config import FeatureConfig
from atp_core.data import clean_matches
from atp_core.exceptions import ConfigurationError, DataValidationError
from atp_core.features import (
    CONTEXT_FEATURES,
    FEATURE_NAMES,
    PLAYER_FEATURES,
    build_training_set,
    context_features,
    default_train_start,
    from_epoch_day,
    match_features,
    to_epoch_day,
)
from atp_core.utils import safe_float

# How a feature row changes when the two players swap sides.
SWAPPED = [("h2h_p1", "h2h_p2"), *((f"p1_{f}", f"p2_{f}") for f in PLAYER_FEATURES)]
NEGATED = ["h2h_diff", "h2h_surface_diff", *(f"{f}_diff" for f in PLAYER_FEATURES)]
COMPLEMENTED = ["elo_prob", "elo_surface_prob"]

SWAP_COLUMNS = [
    ("winner_id", "loser_id"),
    ("winner_name", "loser_name"),
    ("winner_hand", "loser_hand"),
    ("winner_ht", "loser_ht"),
    ("winner_age", "loser_age"),
    ("winner_rank", "loser_rank"),
    ("winner_rank_points", "loser_rank_points"),
] + [(f"w_{s}", f"l_{s}") for s in ("ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced")]


@pytest.fixture(scope="module")
def cleaned(raw_matches):
    return clean_matches(raw_matches)


@pytest.fixture(scope="module")
def training_set(cleaned):
    return build_training_set(cleaned)


def _same(a, b):
    return np.allclose(np.nan_to_num(a, nan=-1e9), np.nan_to_num(b, nan=-1e9), atol=1e-4)


def mirror(row):
    """The feature row seen from the other player's side."""
    index = {name: i for i, name in enumerate(FEATURE_NAMES)}
    out = row.astype(np.float64).copy()
    for a, b in SWAPPED:
        out[index[a]], out[index[b]] = row[index[b]], row[index[a]]
    for name in NEGATED:
        out[index[name]] = -row[index[name]]
    for name in COMPLEMENTED:
        out[index[name]] = 1.0 - row[index[name]]
    return out


def test_every_feature_has_a_mirror_rule():
    """A new feature needs a rule above (context features stay unchanged), or the mirror test cannot check it."""
    covered = {n for pair in SWAPPED for n in pair} | set(NEGATED) | set(COMPLEMENTED) | set(CONTEXT_FEATURES)
    assert covered == set(FEATURE_NAMES)


def test_no_feature_describes_the_winner_or_the_match_outcome():
    # Workload features such as minutes_recent describe *previous* matches and are allowed.
    outcome = {"minutes", "score", "target", "rank_diff", "points_diff", "rank_ratio", "points_ratio"}
    assert not [n for n in FEATURE_NAMES if n.startswith(("winner", "loser")) or n in outcome]


def test_rows_come_in_mirrored_pairs(training_set):
    X, y = training_set.X, training_set.y
    assert X.shape == (2 * len(training_set.matches), len(FEATURE_NAMES))
    assert y[0::2].all() and not y[1::2].any()
    for i in range(0, len(X), 2):
        assert _same(mirror(X[i]), X[i + 1]), f"rows {i} and {i + 1} are not mirrored"


def test_features_do_not_depend_on_the_match_outcome(cleaned):
    """Swapping who won match k must only swap the order of its two rows."""
    k = len(cleaned) // 2
    flipped = cleaned.copy()
    for a, b in SWAP_COLUMNS:
        flipped.loc[k, [a, b]] = cleaned.loc[k, [b, a]].to_numpy()
    original = build_training_set(cleaned).X
    changed = build_training_set(flipped).X
    assert _same(original[: 2 * k], changed[: 2 * k])  # earlier matches are untouched
    assert _same(original[2 * k], changed[2 * k + 1])
    assert _same(original[2 * k + 1], changed[2 * k])


def test_prediction_features_equal_training_features(cleaned, training_set):
    """Train/serve parity: the state of matches < k plus pre-match info reproduces row k."""
    k = len(cleaned) - 40
    history = build_training_set(cleaned.iloc[:k]).state
    row = cleaned.iloc[k]
    day = int(to_epoch_day(pd.Timestamp(row["tourney_date"])))
    for side in ("winner", "loser"):
        history.observe_pre_match(
            str(row[f"{side}_id"]),
            row[f"{side}_name"],
            day=day,
            hand=row[f"{side}_hand"],
            rank=safe_float(row[f"{side}_rank"]),
            points=safe_float(row[f"{side}_rank_points"]),
            age=safe_float(row[f"{side}_age"]),
            height=safe_float(row[f"{side}_ht"]),
        )
    serving = match_features(history, row["winner_id"], row["loser_id"], row["surface"], int(row["best_of"]), day)
    assert _same(np.asarray(serving, dtype=np.float32), training_set.X[2 * k])


def test_train_start_controls_emitted_rows(cleaned):
    late = build_training_set(cleaned, train_start="2018-01-01")
    assert late.matches["date"].min() >= np.datetime64("2018-01-01")
    assert len(late.state.players) == len(build_training_set(cleaned).state.players)
    with pytest.raises(DataValidationError):
        build_training_set(cleaned, train_start="2030-01-01")
    with pytest.raises(DataValidationError):
        build_training_set(cleaned.iloc[0:0])


def test_default_train_start_keeps_a_warm_up_for_long_histories(cleaned):
    assert default_train_start(cleaned) == cleaned["tourney_date"].min()
    long = cleaned.copy()
    long.loc[long.index[-1], "tourney_date"] = pd.Timestamp("2030-01-01")
    assert default_train_start(long) == pd.Timestamp("2017-01-01")


def test_context_features_and_dates():
    assert context_features("Clay", 5) == [0.0, 1.0, 0.0, 0.0, 1.0]
    assert context_features("Unknown", 3) == [0.0, 0.0, 0.0, 0.0, 0.0]
    day = to_epoch_day(pd.Timestamp("2024-06-01"))
    assert from_epoch_day(day) == pd.Timestamp("2024-06-01")


def test_time_decay_changes_recent_form(cleaned):
    plain = build_training_set(cleaned, FeatureConfig(time_decay=0.0)).X
    decayed = build_training_set(cleaned, FeatureConfig(time_decay=0.5)).X
    column = FEATURE_NAMES.index("p1_recent_form")
    assert not _same(plain[:, column], decayed[:, column])


@pytest.mark.parametrize("kwargs", [{"recent_n": 0}, {"time_decay": 1.0}, {"momentum_n": 99}, {"elo_k": -1}])
def test_feature_config_validation(kwargs):
    with pytest.raises(ConfigurationError):
        FeatureConfig(**kwargs).validate()
