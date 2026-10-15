import math

import pandas as pd
import pytest

from ml.scoring import compute_team_needs, rank_candidates


def test_lower_is_better_for_team_turnovers_and_defensive_rating():
    teams = pd.DataFrame({
        "team": ["Weak", "Strong", "Strong"],
        "ORB%": [20, 20, 20],
        "DRB%": [70, 80, 80],
        "TOV%": [20, 10, 10],
        "TS%": [0.45, 0.65, 0.65],
        "ORtg": [90, 120, 120],
        "DRtg": [120, 90, 90],
    })
    weak_needs, warnings = compute_team_needs(teams, "Weak")
    strong_needs, _ = compute_team_needs(teams, "Strong")

    assert warnings == ("Team stat ORB% has no usable variation",)
    assert set(weak_needs) == {"DRB%", "TOV%", "TS%", "ORtg", "DRtg"}
    assert sum(weak_needs.values()) == pytest.approx(1.0)
    assert strong_needs == {}


def test_player_direction_contributions_and_missing_value_warning():
    teams = pd.DataFrame({
        "team": ["Needs help", "Steady"],
        "ORB%": [20, 20],
        "TOV%": [20, 10],
        "TS%": [0.55, 0.55],
        "ORtg": [110, 110],
        "DRtg": [110, 110],
    })
    players = pd.DataFrame({
        "Name": ["Low turnovers", "High turnovers", "Unknown turnovers"],
        "classification": ["College"] * 3,
        "TOV%": [8.0, 20.0, None],
        "TS%": [0.6] * 3,
        "PER": [20.0] * 3,
        "AST/TO": [2.0] * 3,
        "Age": [20.0] * 3,
    })
    ranking = rank_candidates(teams, players, "Needs help")

    assert [candidate.player for candidate in ranking.candidates] == [
        "Low turnovers", "Unknown turnovers", "High turnovers"
    ]
    assert [candidate.contributions["team_fit:TOV%"] for candidate in ranking.candidates] == [
        1.0, 0.5, 0.0
    ]
    assert "Missing TOV%; used neutral 0.5" in ranking.candidates[1].warnings
    for candidate in ranking.candidates:
        assert candidate.score == pytest.approx(sum(candidate.contributions.values()))
        assert math.isfinite(candidate.score)


def test_lower_player_defensive_rating_is_rewarded():
    teams = pd.DataFrame({
        "team": ["Weak defense", "Strong defense"],
        "DRtg": [120, 90],
    })
    players = pd.DataFrame({
        "Name": ["Lower rating", "Higher rating"],
        "classification": ["College", "College"],
        "DRTG": [95, 120],
        "TS%": [0.55, 0.55],
        "PER": [20, 20],
        "AST/TO": [2, 2],
        "Age": [20, 20],
    })
    ranking = rank_candidates(teams, players, "Weak defense", composite_weight=0)
    assert ranking.candidates[0].player == "Lower rating"
    assert ranking.candidates[0].contributions["team_fit:DRtg"] == 1


def test_higher_player_defensive_rebounding_fits_a_weak_rebounding_team():
    teams = pd.DataFrame({"team": ["Weak boards", "Strong boards"], "DRB%": [70, 80]})
    players = pd.DataFrame({
        "Name": ["Better rebounder", "Worse rebounder"],
        "classification": ["College", "College"],
        "DRB%": [25, 10],
        "TS%": [0.55, 0.55],
        "PER": [20, 20],
        "AST/TO": [2, 2],
        "Age": [20, 20],
    })
    ranking = rank_candidates(teams, players, "Weak boards", composite_weight=0)
    assert ranking.team_needs == {"DRB%": 1}
    assert ranking.candidates[0].player == "Better rebounder"
    assert ranking.candidates[0].contributions["team_fit:DRB%"] == 1


def test_invalid_or_ambiguous_inputs_are_rejected():
    teams = pd.DataFrame({"team": ["A"]})
    players = pd.DataFrame({"Name": ["Same", "Same"], "classification": ["College"] * 2})
    with pytest.raises(ValueError, match="unique"):
        rank_candidates(teams, players, "A")
    with pytest.raises(ValueError, match="composite_weight"):
        rank_candidates(teams, players.iloc[:1], "A", composite_weight=1.1)
    with pytest.raises(KeyError):
        rank_candidates(teams, players.iloc[:1], "Unknown")
