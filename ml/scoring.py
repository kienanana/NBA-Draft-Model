"""Explainable, deterministic scoring for a historical draft class.

All scores are relative to the selected draft class. Missing prospect values
receive a neutral 0.5 after scaling and are reported in candidate warnings.
The old optimized weight files used a different scoring contract and are not
used by this baseline.
"""

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


MODEL_VERSION = "baseline-v1"


@dataclass(frozen=True)
class StatDefinition:
    name: str
    player_column: str
    player_higher_is_better: bool
    team_column: str | None = None
    team_higher_is_better: bool | None = None


# These team columns have a documented counterpart in the prospect pool.
# In the historical team CSV, DRB% is in the opponent section, so it is
# deliberately excluded rather than matched to a prospect's own DRB%.
TEAM_FIT_STATS = (
    StatDefinition("ORB%", "ORB%", True, "ORB%", True),
    StatDefinition("TOV%", "TOV%", False, "TOV%", False),
    StatDefinition("TS%", "TS%", True, "TS%", True),
    StatDefinition("ORtg", "ORTG", True, "ORtg", True),
    StatDefinition("DRtg", "DRTG", False, "DRtg", False),
)

# A small, transparent baseline for general prospect quality. These columns
# are populated for both college and non-college prospects in the current data.
QUALITY_STATS = (
    StatDefinition("TS%", "TS%", True),
    StatDefinition("PER", "PER", True),
    StatDefinition("AST/TO", "AST/TO", True),
    StatDefinition("Age", "Age", False),
)


@dataclass(frozen=True)
class CandidateScore:
    rank: int
    player: str
    classification: str
    score: float
    contributions: Mapping[str, float]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class RankingResult:
    team: str
    model_version: str
    composite_weight: float
    team_needs: Mapping[str, float]
    warnings: tuple[str, ...]
    candidates: tuple[CandidateScore, ...]


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def compute_team_needs(
    team_stats: pd.DataFrame, team: str,
    definitions: tuple[StatDefinition, ...] = TEAM_FIT_STATS,
) -> tuple[dict[str, float], tuple[str, ...]]:
    """Return weights for a team's below-average, direction-aware stats.

    The reference population is the unique teams in the historical lottery
    order. A team with multiple picks counts once in the reference mean.
    """
    if "team" not in team_stats.columns:
        raise ValueError("Team data must contain a 'team' column")
    reference = team_stats.drop_duplicates(subset="team").set_index("team")
    if team not in reference.index:
        raise KeyError(team)

    gaps: dict[str, float] = {}
    warnings: list[str] = []
    for stat in definitions:
        column = stat.team_column
        if column is None or column not in reference.columns:
            warnings.append(f"Team stat {column or stat.name} is unavailable")
            continue
        values = _numeric(reference[column])
        value = values.loc[team]
        std = values.std(ddof=0)
        if not np.isfinite(value):
            warnings.append(f"Team stat {column} is missing for {team}")
            continue
        if not np.isfinite(std) or std == 0:
            warnings.append(f"Team stat {column} has no usable variation")
            continue
        z = (float(value) - float(values.mean())) / float(std)
        oriented_z = z if stat.team_higher_is_better else -z
        if oriented_z < 0:
            gaps[stat.name] = -oriented_z

    total = sum(gaps.values())
    return ({name: gap / total for name, gap in gaps.items()} if total else {}), tuple(warnings)


def _scaled_player_stat(
    players: pd.DataFrame, stat: StatDefinition,
) -> tuple[pd.Series, pd.Series]:
    """Scale within college/non-college cohorts; return values and missing mask."""
    if stat.player_column in players.columns:
        raw = _numeric(players[stat.player_column])
    else:
        raw = pd.Series(np.nan, index=players.index, dtype=float)
    missing = raw.isna()
    scaled = pd.Series(0.5, index=players.index, dtype=float)
    college = players["classification"].eq("College")
    for mask in (college, ~college):
        values = raw.loc[mask]
        observed = values.dropna()
        if observed.empty or observed.min() == observed.max():
            continue
        oriented = values if stat.player_higher_is_better else -values
        low, high = float(oriented.min()), float(oriented.max())
        scaled.loc[mask] = ((oriented - low) / (high - low)).fillna(0.5).clip(0, 1)
    return scaled, missing


def rank_candidates(
    team_stats: pd.DataFrame,
    players: pd.DataFrame,
    team: str,
    composite_weight: float = 0.2,
    excluded_players: frozenset[str] = frozenset(),
) -> RankingResult:
    """Rank candidates using one additive score with an auditable breakdown."""
    if not np.isfinite(composite_weight) or not 0 <= composite_weight <= 1:
        raise ValueError("composite_weight must be between 0 and 1")
    required = {"Name", "classification"}
    missing_columns = required - set(players.columns)
    if missing_columns:
        raise ValueError(f"Prospect data is missing columns: {', '.join(sorted(missing_columns))}")
    if players["Name"].isna().any() or players["Name"].duplicated().any():
        raise ValueError("Prospect names must be present and unique within a draft class")

    needs, team_warnings = compute_team_needs(team_stats, team)
    unique_stats = {stat.player_column: stat for stat in (*TEAM_FIT_STATS, *QUALITY_STATS)}
    normalized = {key: _scaled_player_stat(players, stat) for key, stat in unique_stats.items()}
    by_name = {stat.name: stat for stat in TEAM_FIT_STATS}

    scored: list[CandidateScore] = []
    for index, player in players.iterrows():
        if player["Name"] in excluded_players:
            continue
        contributions: dict[str, float] = {}
        warnings = list(team_warnings)
        for name, weight in needs.items():
            stat = by_name[name]
            scaled, missing = normalized[stat.player_column]
            contributions[f"team_fit:{name}"] = weight * float(scaled.loc[index])
            if missing.loc[index]:
                warnings.append(f"Missing {stat.player_column}; used neutral 0.5")
        for stat in QUALITY_STATS:
            scaled, missing = normalized[stat.player_column]
            contributions[f"quality:{stat.name}"] = composite_weight / len(QUALITY_STATS) * float(scaled.loc[index])
            if missing.loc[index]:
                warnings.append(f"Missing {stat.player_column}; used neutral 0.5")
        scored.append(CandidateScore(
            rank=0,
            player=str(player["Name"]),
            classification=str(player["classification"]),
            score=sum(contributions.values()),
            contributions=contributions,
            warnings=tuple(dict.fromkeys(warnings)),
        ))

    scored.sort(key=lambda candidate: (-candidate.score, candidate.player.casefold()))
    ranked = tuple(CandidateScore(
        rank=index + 1,
        player=candidate.player,
        classification=candidate.classification,
        score=candidate.score,
        contributions=candidate.contributions,
        warnings=candidate.warnings,
    ) for index, candidate in enumerate(scored))
    return RankingResult(team, MODEL_VERSION, composite_weight, needs, team_warnings, ranked)
