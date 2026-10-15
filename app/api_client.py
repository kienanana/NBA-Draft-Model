"""Small HTTP client used by the Streamlit ranking view."""

import os

import requests


DEFAULT_API_URL = "http://127.0.0.1:8000"


class RankingAPIError(RuntimeError):
    pass


def _request(path: str, params: dict) -> dict:
    base_url = os.getenv("DRAFTLAB_API_URL", DEFAULT_API_URL).rstrip("/")
    try:
        response = requests.get(f"{base_url}{path}", params=params, timeout=10)
    except requests.RequestException as exc:
        raise RankingAPIError(f"Cannot reach the ranking API at {base_url}. Start the FastAPI server.") from exc

    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RankingAPIError(f"Ranking API returned {response.status_code}: {detail}")
    try:
        return response.json()
    except ValueError as exc:
        raise RankingAPIError("Ranking API returned an invalid JSON response") from exc


def get_teams(year: int) -> list[str]:
    return _request("/api/v1/teams", {"year": year})["teams"]


def get_rankings(year: int, team: str, composite_weight: float = 0.2) -> dict:
    return _request("/api/v1/rankings", {
        "year": year,
        "team": team,
        "limit": 100,
        "composite_weight": composite_weight,
    })
