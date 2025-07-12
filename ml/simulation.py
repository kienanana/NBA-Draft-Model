import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from .team_needs import get_team_needs
from .features import add_composite_scores, college_features, noncollege_features
from .playstyles import compute_playstyle_clusters
from .weights import college_weights, noncollege_weights, unpack_weights

def load_default_weights():
    college_path = "best_college_weights.npy"
    noncollege_path = "best_noncollege_weights.npy"

    if os.path.exists(college_path):
        college_array = np.load(college_path)
        college_w = unpack_weights(college_array, is_college=True)
    else:
        college_w = unpack_weights(np.ones(len(college_features) * 3), is_college=True)

    if os.path.exists(noncollege_path):
        noncollege_array = np.load(noncollege_path)
        noncollege_w = unpack_weights(noncollege_array, is_college=False)
    else:
        noncollege_w = unpack_weights(np.ones(len(noncollege_features) * 3), is_college=False)

    return college_w, noncollege_w

def score_player_for_team(player_row, team_needs, use_composite=True, composite_weight=0.5):
    score = 0.0
    for stat, weight in team_needs.items():
        norm_stat = stat + "_norm"
        if norm_stat in player_row and not pd.isna(player_row[norm_stat]):
            score += weight * player_row[norm_stat]
    if use_composite:
        comps = [
            player_row.get("OffenseScore"),
            player_row.get("DefenseScore"),
            player_row.get("GeneralScore"),
        ]
        comps = [c for c in comps if pd.notna(c)]
        if comps:
            score += composite_weight * np.mean(comps)
    return score

def simulate_draft(year, composite_weight=0.2, plot_distribution=True, college_weights=None, noncollege_weights=None):
    if college_weights is None or noncollege_weights is None:
        college_weights, noncollege_weights = load_default_weights()
        
    # Load data
    players_df = pd.read_csv(f"../data/processed/{year}/draftpool_stats_{year}.csv")
    draft_order = pd.read_csv(f"../data/processed/{year}/draft_{year}_team_stats.csv")["team"].tolist()
    team_stats_df = pd.read_csv(f"../data/processed/{year}/draft_{year}_team_stats.csv")
    team_needs = get_team_needs(team_stats_df)

    # Clustering by classification
    college_df = players_df[players_df["classification"] == "College"].copy()
    noncollege_df = players_df[players_df["classification"] != "College"].copy()
    college_df = compute_playstyle_clusters(college_df, feature_cols=college_features, plot=False)
    noncollege_df = compute_playstyle_clusters(noncollege_df, feature_cols=noncollege_features, plot=False)
    players_df = pd.concat([college_df, noncollege_df], ignore_index=True)

    # Composite scores
    players_df = add_composite_scores(players_df, college_weights, noncollege_weights)

    # Normalize team-relevant stats by classification
    all_needed_stats = set(stat for needs in team_needs.values() for stat in needs)
    players_df_normalized = players_df.copy()

    for stat in all_needed_stats:
        if stat in players_df_normalized.columns:
            for cls in players_df_normalized["classification"].unique():
                mask = players_df_normalized["classification"] == cls
                min_val = players_df_normalized.loc[mask, stat].min()
                max_val = players_df_normalized.loc[mask, stat].max()
                if max_val - min_val != 0:
                    players_df_normalized.loc[mask, stat + "_norm"] = (
                        (players_df_normalized.loc[mask, stat] - min_val) / (max_val - min_val)
                    )
                else:
                    players_df_normalized.loc[mask, stat + "_norm"] = 0.5

    # Simulate draft
    selected_players = []
    for team in draft_order:
        needs = team_needs.get(team, {})
        available_players = players_df_normalized[
            ~players_df_normalized["Name"].isin([p["Player"] for p in selected_players])
        ].copy()

        available_players["FitScore"] = available_players.apply(
            lambda row: score_player_for_team(row, needs, use_composite=True, composite_weight=composite_weight),
            axis=1
        )
        best_pick = available_players.sort_values("FitScore", ascending=False).iloc[0]
        selected_players.append({
            "Team": team,
            "Player": best_pick["Name"],
            "FitScore": best_pick["FitScore"]
        })

    sim_df = pd.DataFrame(selected_players)
    actual_df = pd.read_csv(f"../data/raw/{year}/draft_{year}.csv")
    merged = sim_df.merge(actual_df, left_on="Player", right_on="player", how="left")
    merged["Correct"] = merged["Team"] == merged["team"]
    accuracy = merged["Correct"].mean()

    # Lottery hit rate
    actual_lottery_players = set(actual_df.sort_values("pick").head(14)["player"])
    sim_players = set(sim_df["Player"])
    lottery_hits = actual_lottery_players.intersection(sim_players)
    lottery_accuracy = len(lottery_hits) / 14

    # Composite distribution plot
    if plot_distribution:
        players_df_normalized["CompositeAvg"] = players_df_normalized[
            ["OffenseScore", "DefenseScore", "GeneralScore"]
        ].mean(axis=1)

        sorted_players = players_df_normalized.sort_values("CompositeAvg", ascending=False).reset_index(drop=True)
        plt.figure(figsize=(14, 6))
        sns.scatterplot(
            data=sorted_players,
            x=sorted_players.index,
            y="CompositeAvg",
            hue="classification",
            palette="Set2",
            s=100,
            edgecolor="black"
        )
        plt.title("Composite Score Distribution by Prospect Classification")
        plt.xlabel("Prospect Rank (by Composite Score)")
        plt.ylabel("CompositeAvg")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    print(f"✅ Draft Accuracy (Team Match): {accuracy:.2%}")
    print(f"🎯 Lottery Hit Rate: {lottery_accuracy:.2%}")
    return sim_df, accuracy, lottery_accuracy
