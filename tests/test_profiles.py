import pandas as pd

from ml.profiles import scale_for_radar


def test_profile_radar_reverses_turnovers():
    profiles = pd.DataFrame({"TOV": [1.0, 4.0], "PTS": [10.0, 20.0]})
    scaled = scale_for_radar(profiles, ["TOV", "PTS"])
    assert scaled.loc[0, "TOV"] == 1.0
    assert scaled.loc[1, "TOV"] == 0.0
    assert scaled.loc[0, "PTS"] == 0.0
    assert scaled.loc[1, "PTS"] == 1.0
