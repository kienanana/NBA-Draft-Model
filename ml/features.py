import numpy as np
import pandas as pd

### --- FEATURE LISTS --- ###

college_features = [
    "FGA", "FG%", "3PA", "3P%", "2PA", "2P%", "eFG%", "FTA", "FT%", "ORB", "TRB",
    "AST", "STL", "BLK", "TOV", "PF", "PTS", "PER", "TS%", "PProd", "ORB%", "DRB%", 
    "TRB%", "AST%", "STL%", "BLK%", "TOV%", "USG%", "OWS", "DWS", "WS", "OBPM", 
    "DBPM", "BPM", "EFG%", "3PAR", "FTAR", "NBA 3P%", "AST/USG", "AST/TO", "OWS/40", 
    "DWS/40", "ORTG", "DRTG"
]

noncollege_features = [
    "FGA", "FG%", "3PA", "3P%", "FTA", "FT%", "TRB", "AST", "STL", "BLK", "TOV", 
    "PF", "PTS", "PER", "TS%", "USG%", "EFG%", "3PAR", "FTAR", "NBA 3P%", 
    "AST/USG", "AST/TO", "ORTG", "DRTG"
]

### --- FEATURE MATRIX BUILDERS --- ###

def build_features_for_college(df):
    return df[college_features].copy()

def build_features_for_noncollege(df):
    return df[noncollege_features].copy()

### --- COMPOSITE SCORE FUNCTIONS --- ###

def compute_composite_score(row, feature_weights):
    """
    Generic function to compute weighted sum of features for a player row.
    """
    score = 0
    for feature, weight in feature_weights.items():
        if feature in row and not pd.isna(row[feature]):
            score += row[feature] * weight
    return score


def normalize_series(series):
    """
    Normalize a pandas Series to a 0–100 range.
    """
    min_val = series.min()
    max_val = series.max()
    if max_val - min_val == 0:
        return pd.Series(50.0, index=series.index)  # avoid division by zero
    return 100 * (series - min_val) / (max_val - min_val)


def add_composite_scores(df, college_weights, noncollege_weights):
    """
    Adds normalized composite scores to dataframe, per classification.
    """
    offense_scores, defense_scores, general_scores = [], [], []

    for _, row in df.iterrows():
        weights = college_weights if row["classification"] == "College" else noncollege_weights
        offense_scores.append(compute_composite_score(row, weights["offense"]))
        defense_scores.append(compute_composite_score(row, weights["defense"]))
        general_scores.append(compute_composite_score(row, weights["general"]))

    df["OffenseScore_raw"] = offense_scores
    df["DefenseScore_raw"] = defense_scores
    df["GeneralScore_raw"] = general_scores

    # 🧩 Normalize across all players (not per classification)
    df["OffenseScore"] = normalize_series(df["OffenseScore_raw"])
    df["DefenseScore"] = normalize_series(df["DefenseScore_raw"])
    df["GeneralScore"] = normalize_series(df["GeneralScore_raw"])


    # Optional cleanup
    df.drop(columns=["OffenseScore_raw", "DefenseScore_raw", "GeneralScore_raw"], inplace=True)
    return df
