import pandas as pd
import pytest

from ml.simulation import evaluate_lottery, simulate_draft


def test_ndcg_keeps_partial_lottery_matches_and_missing_players_at_zero():
    actual = pd.DataFrame({
        "pick": list(range(1, 15)),
        "team": [f"Team {number}" for number in range(1, 15)],
        "player": [f"Player {number}" for number in range(1, 15)],
    })
    predicted = pd.DataFrame({
        "Team": ["Team 1"] + ["Other"] * 13,
        "Player": ["Player 1"] + [f"Other {number}" for number in range(13)],
    })
    metrics = evaluate_lottery(predicted, actual)
    assert metrics["team_match"] == pytest.approx(1 / 14)
    assert metrics["lottery_hit_rate"] == pytest.approx(1 / 14)
    assert 0 < metrics["ndcg_at_14"] < 1
    assert metrics["first_hit_reciprocal_rank"] == 1


def test_historical_simulation_returns_finite_metrics():
    picks, match, hit, first_hit_rr, ndcg = simulate_draft(2024)
    assert len(picks) == 14
    assert picks["Player"].is_unique
    assert picks["FitScore"].notna().all()
    assert all(0 <= metric <= 1 for metric in (match, hit, first_hit_rr, ndcg))
    assert ndcg > 0


def test_evaluation_matches_known_draft_name_variants():
    predicted = pd.DataFrame({"Team": ["A", "B"], "Player": ["Alexandre Sarr", "Nikola Topić"]})
    actual = pd.DataFrame({
        "pick": [1, 2, 15], "team": ["A", "B", "C"],
        "player": ["Alex Sarr", "Nikola Topic", "Other player"],
    })
    metrics = evaluate_lottery(predicted, actual)
    assert metrics["team_match"] == 1
    assert metrics["lottery_hit_rate"] == 1
