import pytest

from ml.lottery_inclusion import FEATURES, backtest, forecast, load_class, load_pool, train_through
from ml.names import player_key


def test_known_draft_aliases_match_pool_names():
    assert player_key("Jabari Smith Jr.") == player_key("Jabari Smith")
    assert player_key("Alex Sarr") == player_key("Alexandre Sarr")
    assert player_key("Nikola Topic") == player_key("Nikola Topić")


def test_every_historical_class_has_fourteen_matched_lottery_labels():
    for year in range(2020, 2025):
        prospects = load_class(year)
        assert prospects["lottery"].sum() == 14
        assert len(prospects) == len(load_pool(year))
    assert not {"Name", "year", "lottery", "pick", "team"} & set(FEATURES)


def test_forecast_uses_only_later_years_and_does_not_require_results(monkeypatch):
    artifact = train_through(2023)
    with pytest.raises(ValueError, match="trained on this year"):
        forecast(2023, artifact)

    def unexpected_class_load(year):
        raise AssertionError("Forecast must not read draft results")

    monkeypatch.setattr("ml.lottery_inclusion.load_class", unexpected_class_load)
    ranked = forecast(2024, artifact)
    assert len(ranked) == 74
    assert ranked["Name"].is_unique
    assert ranked["model_score"].is_monotonic_decreasing
    assert 0 < ranked["model_score"].min() < ranked["model_score"].max() < 1


def test_backtest_trains_only_on_prior_years():
    report = backtest()
    assert report["test_year"].tolist() == [2022, 2023, 2024]
    assert report["train_through"].tolist() == [2021, 2022, 2023]
    assert report["model_hits_at_14"].between(0, 14).all()
    assert report["quality_hits_at_14"].between(0, 14).all()
