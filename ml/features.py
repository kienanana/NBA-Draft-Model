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


def add_composite_scores(df, college_weights, noncollege_weights):
    """
    Adds composite scores to dataframe, applying different formulas based on classification.
    """
    offense_scores, defense_scores, general_scores = [], [], []
    
    for _, row in df.iterrows():
        if row["classification"] == "College":
            weights = college_weights
        else:
            weights = noncollege_weights
        
        offense_scores.append(compute_composite_score(row, weights["offense"]))
        defense_scores.append(compute_composite_score(row, weights["defense"]))
        general_scores.append(compute_composite_score(row, weights["general"]))
    
    df["OffenseScore"] = offense_scores
    df["DefenseScore"] = defense_scores
    df["GeneralScore"] = general_scores
    
    return df
