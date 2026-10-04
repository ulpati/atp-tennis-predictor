import pytest

from atp_core.elo import EloEngine, expected_score


def test_expected_score_is_symmetric():
    assert expected_score(1500, 1500) == 0.5
    assert expected_score(1700, 1500) + expected_score(1500, 1700) == pytest.approx(1.0)


def test_fixed_k_update_is_zero_sum():
    elo = EloEngine(k=32)
    elo.update("a", "b", "Hard")
    assert elo.rating("a") == pytest.approx(1516)
    assert elo.rating("b") == pytest.approx(1484)
    assert elo.rating("a", "Hard") == pytest.approx(1516)
    assert elo.rating("a", "Clay") == 1500  # never played on clay


def test_dynamic_k_shrinks_with_experience():
    elo = EloEngine()
    assert elo.k_factor(0) > elo.k_factor(10) > elo.k_factor(200)


def test_unknown_surface_only_updates_overall_rating():
    elo = EloEngine(k=20)
    elo.update("a", "b", "Unknown")
    assert "Unknown" not in elo.surface_ratings
    assert elo.matches("a") == 1
    assert elo.matches("a", "Hard") == 0


def test_round_trip():
    elo = EloEngine()
    for _ in range(3):
        elo.update("a", "b", "Clay")
    clone = EloEngine.from_dict(elo.to_dict())
    assert clone.rating("a", "Clay") == elo.rating("a", "Clay")
    assert clone.expected("a", "b") == elo.expected("a", "b")
    assert clone.k is None
