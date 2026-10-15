"""Historical lottery simulation built on the shared ranking engine."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .scoring import rank_candidates


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED_ROOT = Path(os.getenv("DRAFTLAB_DATA_DIR", PROJECT_ROOT / "data" / "processed"))
DATA_RAW_ROOT = Path(os.getenv("DRAFTLAB_DATA_RAW_DIR", PROJECT_ROOT / "data" / "raw"))
MODELS_ROOT = Path(os.getenv("DRAFTLAB_MODELS_DIR", PROJECT_ROOT / "models"))
SUPPORTED_YEARS = (2020, 2021, 2022, 2023, 2024)


def load_historical_data(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the prospect pool and lottery-order team context for one year."""
    if year not in SUPPORTED_YEARS:
        raise ValueError(f"Unsupported year {year}; choose one of {SUPPORTED_YEARS}")
    players_path = DATA_PROCESSED_ROOT / str(year) / f"draftpool_stats_{year}.csv"
    teams_path = DATA_PROCESSED_ROOT / str(year) / f"draft_{year}_team_stats.csv"
    return pd.read_csv(players_path), pd.read_csv(teams_path)


def load_clusters():
    """Load models used by the separate playstyle visualization."""
    import joblib

    college_path = MODELS_ROOT / "kmeans_college.pkl"
    noncollege_path = MODELS_ROOT / "kmeans_noncollege.pkl"
    if not college_path.exists() or not noncollege_path.exists():
        raise FileNotFoundError("Cluster model files not found. Run fit_global_clusters.py first.")
    return joblib.load(college_path), joblib.load(noncollege_path)


def evaluate_lottery(sim_df: pd.DataFrame, actual_df: pd.DataFrame) -> dict[str, float]:
    """Evaluate a ranked set of picks; unselected actual players have relevance 0."""
    actual = actual_df.drop_duplicates(subset="player").set_index("player")
    actual_names = set(actual.index)
    selected_names = set(sim_df["Player"])

    team_matches = [
        name in actual_names and actual.loc[name, "team"] == team
        for name, team in zip(sim_df["Player"], sim_df["Team"])
    ]
    team_match = sum(team_matches) / len(sim_df) if len(sim_df) else 0.0
    lottery_hit_rate = len(actual_names & selected_names) / len(actual_names) if len(actual_names) else 0.0

    relevance = [
        max(0, 15 - int(actual.loc[name, "pick"])) if name in actual_names else 0
        for name in sim_df["Player"].head(14)
    ]
    ideal_relevance = sorted(
        (max(0, 15 - int(pick)) for pick in actual["pick"]), reverse=True
    )[:14]

    def dcg(values):
        return sum(value / np.log2(index + 2) for index, value in enumerate(values))

    ideal_dcg = dcg(ideal_relevance)
    ndcg = dcg(relevance) / ideal_dcg if ideal_dcg else 0.0
    first_hit_reciprocal_rank = next(
        (1.0 / rank for rank, name in enumerate(sim_df["Player"], start=1) if name in actual_names),
        0.0,
    )
    return {
        "team_match": float(team_match),
        "lottery_hit_rate": float(lottery_hit_rate),
        "first_hit_reciprocal_rank": float(first_hit_reciprocal_rank),
        "ndcg_at_14": float(ndcg),
    }


def simulate_draft(
    year: int,
    composite_weight: float = 0.2,
    plot_distribution: bool = False,
    college_weights=None,
    noncollege_weights=None,
):
    """Pick the top remaining candidate for each team in historical draft order.

    The tuple return shape is retained for existing callers. Its fourth value
    is now reciprocal rank of the first lottery hit, rather than the old
    rank-distance value that was incorrectly labelled MRR.
    """
    if college_weights is not None or noncollege_weights is not None:
        raise ValueError("Legacy optimized weights are incompatible with baseline-v1 scoring")

    players, teams = load_historical_data(year)
    selected: list[dict] = []
    taken: set[str] = set()
    for team in teams["team"]:
        ranking = rank_candidates(teams, players, team, composite_weight, frozenset(taken))
        if not ranking.candidates:
            raise ValueError("The prospect pool has fewer candidates than lottery picks")
        choice = ranking.candidates[0]
        selected.append({
            "Team": team,
            "Player": choice.player,
            "FitScore": choice.score,
            "Warnings": "; ".join(choice.warnings),
        })
        taken.add(choice.player)

    sim_df = pd.DataFrame(selected)
    actual = pd.read_csv(DATA_RAW_ROOT / str(year) / f"draft_{year}.csv")
    metrics = evaluate_lottery(sim_df, actual)

    if plot_distribution:
        import matplotlib.pyplot as plt

        plt.plot(range(1, len(sim_df) + 1), sim_df["FitScore"], marker="o")
        plt.xlabel("Pick")
        plt.ylabel("Fit score")
        plt.title(f"{year} simulated pick scores")
        plt.show()

    return (
        sim_df,
        metrics["team_match"],
        metrics["lottery_hit_rate"],
        metrics["first_hit_reciprocal_rank"],
        metrics["ndcg_at_14"],
    )


def average_rank_difference(df_pred, df_true):
    """Average absolute position difference for players present in both lists."""
    pred_ranks = df_pred.reset_index().reset_index().set_index("Player")["level_0"]
    true_ranks = df_true.reset_index().reset_index().set_index("Player")["level_0"]
    common_players = pred_ranks.index.intersection(true_ranks.index)
    return (pred_ranks.loc[common_players] - true_ranks.loc[common_players]).abs().mean()
