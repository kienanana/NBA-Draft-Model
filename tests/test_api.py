import pytest
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_successful_historical_ranking_is_explainable():
    response = client.get("/api/v1/rankings", params={
        "year": 2024, "team": "Atlanta Hawks", "limit": 5, "composite_weight": 0.3,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "baseline-v1"
    assert body["team"] == "Atlanta Hawks"
    assert body["total_candidates"] == 74
    assert len(body["candidates"]) == 5
    assert [candidate["rank"] for candidate in body["candidates"]] == list(range(1, 6))
    assert sum(body["team_needs"].values()) == pytest.approx(1)
    for candidate in body["candidates"]:
        assert candidate["score"] == pytest.approx(sum(candidate["contributions"].values()))
    assert [c["score"] for c in body["candidates"]] == sorted(
        [c["score"] for c in body["candidates"]], reverse=True
    )


def test_teams_endpoint_returns_unique_teams_in_lottery_order():
    response = client.get("/api/v1/teams", params={"year": 2024})
    assert response.status_code == 200
    teams = response.json()["teams"]
    assert teams[0] == "Atlanta Hawks"
    assert len(teams) == len(set(teams))


@pytest.mark.parametrize("params", [
    {"year": 2025, "team": "Atlanta Hawks"},
    {"year": "wrong", "team": "Atlanta Hawks"},
    {"year": 2024, "team": "Atlanta Hawks", "limit": 0},
    {"year": 2024, "team": "Atlanta Hawks", "composite_weight": 1.1},
    {"year": 2024, "team": "  "},
])
def test_invalid_requests_return_422(params):
    response = client.get("/api/v1/rankings", params=params)
    assert response.status_code == 422
    assert response.json()["detail"]


def test_unknown_team_returns_404():
    response = client.get("/api/v1/rankings", params={"year": 2024, "team": "Unknown City"})
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_missing_historical_data_returns_503(monkeypatch):
    def unavailable(_year):
        raise FileNotFoundError("missing")

    monkeypatch.setattr("api.main.load_historical_data", unavailable)
    response = client.get("/api/v1/rankings", params={"year": 2024, "team": "Atlanta Hawks"})
    assert response.status_code == 503
    assert response.json() == {"detail": "Historical data for 2024 is unavailable"}
