import pytest

from atp_core.artifacts import load_run
from atp_core.exceptions import ConfigurationError, InvalidSurfaceError, PlayerNotFoundError
from atp_core.predict import PlayerIndex, Predictor, confidence_label


@pytest.fixture(scope="module")
def predictor(trained_models_dir):
    return Predictor(load_run(trained_models_dir))


@pytest.fixture
def index():
    names = {
        "1": "Rafael Nadal",
        "2": "Roger Federer",
        "3": "Ana Rossi",
        "4": "Luca Rossi",
        "5": "José Álvarez",
        "6": "Old Rafael Nadal",
    }
    return PlayerIndex(names, {"1": 100, "2": 90, "3": 80, "4": 70, "5": 60, "6": 10})


def test_exact_and_accent_insensitive_matches(index):
    assert index.resolve("Rafael Nadal") == "1"
    assert index.resolve("  rafael   NADAL ") == "1"
    assert index.resolve("Jose Alvarez") == "5"


def test_unique_partial_name(index):
    assert index.resolve("Federer") == "2"


def test_ambiguous_partial_name_lists_candidates(index):
    with pytest.raises(PlayerNotFoundError) as excinfo:
        index.resolve("Rossi")
    assert excinfo.value.reason == "is ambiguous"
    assert excinfo.value.suggestions == ["Ana Rossi", "Luca Rossi"]


def test_partial_names_prefer_the_only_active_player():
    index = PlayerIndex(
        {"1": "Jannik Sinner", "2": "Martin Sinner", "3": "Alexander Zverev", "4": "Alexander Zverev Sr"},
        {"1": 20_000, "2": 10_000, "3": 20_000, "4": 6_000},
    )
    assert index.resolve("Sinner") == "1"
    assert index.resolve("Zverev, Alexander") == "3"
    both_active = PlayerIndex({"1": "Ana Rossi", "2": "Luca Rossi"}, {"1": 20_000, "2": 19_900})
    with pytest.raises(PlayerNotFoundError, match="ambiguous"):
        both_active.resolve("Rossi")


def test_small_typo_is_accepted_but_unknown_names_are_not(index):
    assert index.resolve("Roger Federrer") == "2"
    with pytest.raises(PlayerNotFoundError) as excinfo:
        index.resolve("Fedrer")
    assert "Roger Federer" in excinfo.value.suggestions
    with pytest.raises(PlayerNotFoundError):
        index.resolve("Somebody Else Entirely")
    with pytest.raises(PlayerNotFoundError, match="empty"):
        index.resolve("   ")


def test_prediction_is_symmetric(predictor):
    ab = predictor.predict("Jannik Sinner", "Carlos Alcaraz", "Hard")
    ba = predictor.predict("Carlos Alcaraz", "Jannik Sinner", "Hard")
    assert ab.probability_a + ab.probability_b == pytest.approx(1.0)
    assert ab.probability_a == pytest.approx(ba.probability_b, abs=1e-6)
    assert ab.winner == ba.winner
    assert ab.confidence in {"low", "medium", "high"}
    assert set(ab.to_dict()) >= {"probability_a", "probability_b", "predicted_winner", "elo_probability_a"}


def test_prediction_options_and_explanation(predictor):
    result = predictor.predict("sinner", "alcaraz", "clay", best_of=5, as_of="2030-01-01", explain=True)
    assert result.surface == "Clay" and result.best_of == 5
    assert result.explanation and {"feature", "log_odds"} <= set(result.explanation[0])
    assert any("has not played since" in w for w in result.warnings)


@pytest.mark.parametrize(
    ("args", "error"),
    [
        (("Jannik Sinner", "Carlos Alcaraz", "Sand"), InvalidSurfaceError),
        (("Jannik Sinner", "Jannik Sinner", "Hard"), PlayerNotFoundError),
        (("Nobody Here", "Carlos Alcaraz", "Hard"), PlayerNotFoundError),
    ],
)
def test_invalid_predictions(predictor, args, error):
    with pytest.raises(error):
        predictor.predict(*args)


def test_dates_before_the_end_of_the_data_are_refused(predictor):
    with pytest.raises(ConfigurationError, match="must not be earlier"):
        predictor.predict("Jannik Sinner", "Carlos Alcaraz", "Hard", as_of="2015-01-01")


def test_invalid_best_of(predictor):
    with pytest.raises(ConfigurationError):
        predictor.predict("Jannik Sinner", "Carlos Alcaraz", "Hard", best_of=4)


@pytest.mark.parametrize(("p", "label"), [(0.5, "low"), (0.62, "medium"), (0.25, "high"), (0.9, "high")])
def test_confidence_label(p, label):
    assert confidence_label(p) == label
