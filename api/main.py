"""FastAPI endpoints for the historical, explainable baseline model."""

import math
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from ml.scoring import rank_candidates
from ml.simulation import load_historical_data


DraftYear = Annotated[int, Query(ge=2020, le=2024)]
app = FastAPI(title="NBA Draft Lab API", version="1.0.0")


class CandidateResponse(BaseModel):
    rank: int
    player: str
    classification: str
    score: float
    contributions: dict[str, float]
    warnings: list[str]


class RankingResponse(BaseModel):
    year: int
    team: str
    model_version: str
    composite_weight: float
    team_needs: dict[str, float]
    warnings: list[str]
    total_candidates: int
    candidates: list[CandidateResponse]


class TeamsResponse(BaseModel):
    year: int
    teams: list[str]


def _load_data(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        return load_historical_data(year)
    except (FileNotFoundError, pd.errors.EmptyDataError) as exc:
        raise HTTPException(status_code=503, detail=f"Historical data for {year} is unavailable") from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/teams", response_model=TeamsResponse)
def teams(year: DraftYear):
    _, team_data = _load_data(year)
    if "team" not in team_data.columns:
        raise HTTPException(status_code=503, detail=f"Historical team data for {year} is invalid")
    return TeamsResponse(year=year, teams=team_data["team"].dropna().drop_duplicates().tolist())


@app.get("/api/v1/rankings", response_model=RankingResponse)
def rankings(
    year: DraftYear,
    team: Annotated[str, Query(min_length=1, max_length=100)],
    limit: Annotated[int, Query(ge=1, le=100)] = 100,
    composite_weight: Annotated[float, Query(ge=0, le=1)] = 0.2,
):
    if not team.strip():
        raise HTTPException(status_code=422, detail="team must not be blank")
    if not math.isfinite(composite_weight):
        raise HTTPException(status_code=422, detail="composite_weight must be finite")

    players, team_data = _load_data(year)
    try:
        result = rank_candidates(team_data, players, team.strip(), composite_weight)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Team '{team.strip()}' not found for {year}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=f"Historical data for {year} is invalid: {exc}") from exc

    candidates = [CandidateResponse(
        rank=candidate.rank,
        player=candidate.player,
        classification=candidate.classification,
        score=candidate.score,
        contributions=dict(candidate.contributions),
        warnings=list(candidate.warnings),
    ) for candidate in result.candidates[:limit]]
    return RankingResponse(
        year=year,
        team=result.team,
        model_version=result.model_version,
        composite_weight=result.composite_weight,
        team_needs=dict(result.team_needs),
        warnings=list(result.warnings),
        total_candidates=len(result.candidates),
        candidates=candidates,
    )
