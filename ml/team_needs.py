"""Compatibility wrapper around the shared scoring contract."""

from .scoring import TEAM_FIT_STATS, compute_team_needs


def get_team_needs(team_stats_df, relevant_features=None):
    definitions = TEAM_FIT_STATS
    if relevant_features is not None:
        requested = set(relevant_features)
        definitions = tuple(stat for stat in TEAM_FIT_STATS if stat.team_column in requested)
        unknown = requested - {stat.team_column for stat in definitions}
        if unknown:
            raise ValueError(f"Unsupported team features: {', '.join(sorted(unknown))}")
    return {
        team: compute_team_needs(team_stats_df, team, definitions)[0]
        for team in team_stats_df["team"].drop_duplicates()
    }
