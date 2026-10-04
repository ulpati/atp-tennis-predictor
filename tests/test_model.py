import numpy as np
import pytest

from atp_core import model as model_module
from atp_core.config import TrainConfig
from atp_core.data import clean_matches
from atp_core.exceptions import ConfigurationError, ModelTrainingError
from atp_core.features import build_training_set
from atp_core.model import (
    Calibrator,
    baseline_metrics,
    chronological_split,
    fit_calibrator,
    match_metrics,
    resolve_device,
    rows_for,
    train,
    winner_probabilities,
)

FAST = {"max_rounds": 60, "early_stopping_rounds": 10, "threads": 2}


@pytest.fixture(scope="module")
def training_set(raw_matches):
    return build_training_set(clean_matches(raw_matches))


def test_split_is_chronological_and_disjoint(training_set):
    split = chronological_split(training_set.matches, TrainConfig())
    dates = training_set.matches["date"].to_numpy()
    assert dates[split.train].max() < dates[split.val].min()
    assert dates[split.val].max() < dates[split.test].min()
    assert len(split.train) + len(split.val) + len(split.test) == len(dates)
    described = split.describe(training_set.matches)
    assert described["test"]["matches"] == len(split.test)


def test_split_with_explicit_test_start(training_set):
    split = chronological_split(training_set.matches, TrainConfig(test_start="2019-01-01"))
    assert training_set.matches["date"].to_numpy()[split.test].min() >= np.datetime64("2019-01-01")
    with pytest.raises(ModelTrainingError):
        chronological_split(training_set.matches, TrainConfig(test_start="2000-01-01"))


def test_split_refuses_tiny_periods(training_set):
    with pytest.raises(ModelTrainingError, match="only"):
        chronological_split(training_set.matches.iloc[:120], TrainConfig())


def test_rows_for_keeps_pairs_together():
    assert rows_for(np.array([0, 3])).tolist() == [0, 1, 6, 7]


def test_metrics_on_known_values():
    perfect = match_metrics(np.array([0.99, 0.9, 0.8]))
    assert perfect["accuracy"] == 1.0 and perfect["auc"] == 1.0
    coin = match_metrics(np.array([0.5, 0.5]))
    assert coin["accuracy"] == 0.5 and coin["log_loss"] == pytest.approx(np.log(2))
    assert match_metrics(np.array([]))["matches"] == 0
    p_rows = np.array([0.7, 0.4, 0.2, 0.6])
    assert winner_probabilities(p_rows).tolist() == pytest.approx([0.65, 0.3])


def test_calibrator_is_symmetric_and_validated():
    rng = np.random.default_rng(0)
    p = np.clip(rng.normal(0.65, 0.15, 500), 0.01, 0.99)
    calibrator = fit_calibrator(p)
    grid = np.linspace(0.05, 0.95, 19)
    assert calibrator(grid) == pytest.approx(1 - calibrator(1 - grid))
    assert Calibrator.from_dict(calibrator.to_dict()) == calibrator
    with pytest.raises(ValueError):
        Calibrator.from_dict({"x": [0.5, 0.1], "y": [0.2, 0.3]})


def test_train_learns_without_leaking(training_set):
    result = train(training_set, TrainConfig(cv_folds=2, search_iter=1, calibration="isotonic", refit=False, **FAST))
    test = result.metrics["test"]
    # The synthetic winners depend on a hidden skill: a sound model clearly beats chance,
    # but a near-perfect score would mean the outcome leaked into the features.
    assert 0.62 < test["accuracy"] < 0.95
    assert test["log_loss"] < 0.69
    assert set(result.metrics["baselines"]) == {"ranking", "elo", "surface_elo"}
    assert "test_uncalibrated" in result.metrics
    assert len(result.search) == 2
    assert result.cv and all("accuracy" in fold for fold in result.cv)
    assert result.test_sample["X"].shape[1] == training_set.X.shape[1]
    assert result.refit == {}


def test_refit_uses_every_period_and_keeps_the_evaluation(training_set):
    evaluated = train(training_set, TrainConfig(refit=False, **FAST))
    refitted = train(training_set, TrainConfig(**FAST))
    # The metrics describe the evaluated model, so retraining afterwards cannot change them.
    assert refitted.metrics == evaluated.metrics
    info = refitted.refit["models"]["all"]
    assert info["matches"] == len(training_set.matches)
    assert info["rounds"] == evaluated.best_iterations["all"] + 1
    assert refitted.boosters["all"].num_boosted_rounds() == info["rounds"]
    assert refitted.best_iterations == evaluated.best_iterations
    assert refitted.refit["to"] == training_set.matches["date"].max().date().isoformat()


def test_per_surface_models_are_chosen_on_validation(training_set, monkeypatch):
    monkeypatch.setattr(model_module, "MIN_MATCHES_PER_SURFACE_MODEL", 100)
    result = train(training_set, TrainConfig(per_surface=True, **FAST))
    choices = result.metrics["surface_models"]
    assert set(choices) <= {"Hard", "Clay", "Grass"} and choices
    surfaces = training_set.matches["surface"].to_numpy()
    for surface, info in choices.items():
        assert (surface in result.boosters) == (info["chosen"] == surface)
        assert result.routing.get(surface, "all") == info["chosen"]
        if info["chosen"] == surface:
            assert result.refit["models"][surface]["matches"] == (surfaces == surface).sum()


def test_baselines_handle_missing_ranks(training_set):
    X = training_set.X[:40].copy()
    X[:, :] = np.nan
    X[:, list(model_module.FEATURE_NAMES).index("elo_prob")] = 0.5
    X[:, list(model_module.FEATURE_NAMES).index("elo_surface_prob")] = 0.5
    result = baseline_metrics(X)
    assert result["ranking"]["matches"] == 0
    assert result["elo"]["accuracy"] == 0.5


def test_resolve_device(monkeypatch):
    assert resolve_device(False) == "cpu"
    monkeypatch.setattr(model_module.xgb, "build_info", lambda: {"USE_CUDA": False})
    assert resolve_device(True) == "cpu"
    monkeypatch.setattr(model_module.xgb, "build_info", lambda: {"USE_CUDA": True})
    monkeypatch.setattr(model_module.xgb, "train", lambda *args, **kwargs: None)
    assert resolve_device(True) == "cuda"

    def no_gpu(*args, **kwargs):
        raise model_module.xgb.core.XGBoostError("no CUDA device")

    monkeypatch.setattr(model_module.xgb, "train", no_gpu)
    assert resolve_device(True) == "cpu"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"test_fraction": 0.6},
        {"cv_folds": 1},
        {"calibration": "platt"},
        {"calibration": "isotonic"},  # calibration needs refit=False
        {"threads": 0},
        {"max_rounds": 5},
    ],
)
def test_train_config_validation(kwargs):
    with pytest.raises(ConfigurationError):
        TrainConfig(**kwargs).validate()
