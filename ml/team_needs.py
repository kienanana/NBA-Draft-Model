import pandas as pd

def get_team_needs(team_stats_df, relevant_features=None):
    """
    Identify weighted team needs using negative z-scores across selected features.
    Returns a dict: {Team: {Feature: NeedWeight}}.
    """
    team_needs = {}
    df = team_stats_df.copy()

    if relevant_features is None:
        # Choose relevant features if not specified
        relevant_features = [
            "ORB%", "DRB%", "TRB%", "AST%", "TOV%", "STL%", "BLK%", 
            "3P%", "FT%", "TS%", "USG%", "ORTG", "DRTG"
        ]
        relevant_features = [f for f in relevant_features if f in df.columns]

    z_scores = df[relevant_features].apply(lambda x: (x - x.mean()) / x.std(), axis=0)

    for idx, row in z_scores.iterrows():
        team = df.loc[idx, "team"]
        needs = {}

        for stat in relevant_features:
            z = row[stat]
            if z < 0:
                needs[stat] = abs(z)

        # Normalize needs so they sum to 1
        total = sum(needs.values())
        if total > 0:
            for stat in needs:
                needs[stat] /= total

        team_needs[team] = needs

    return team_needs

