"""Train and backtest a prospect-only model for top-14 draft inclusion.

Run ``python -m ml.lottery_inclusion`` from the repository root. The saved
model is trained through 2023, leaving 2024 as a historical holdout.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .names import player_key
from .scoring import rank_candidates
from .simulation import (
    DATA_PROCESSED_ROOT, DATA_RAW_ROOT, PROJECT_ROOT, evaluate_lottery,
    load_historical_data,
)


MODEL_VERSION = "lottery-inclusion-v1"
MODEL_PATH = PROJECT_ROOT / "models" / "lottery_inclusion_v1.joblib"
BACKTEST_PATH = PROJECT_ROOT / "reports" / "lottery_inclusion_backtest.csv"
YEARS = (2020, 2021, 2022, 2023, 2024)
NUMERIC_FEATURES = (
    "Age", "G", "MP", "PTS", "TRB", "AST", "STL", "BLK", "TOV",
    "TS%", "PER", "3PA", "3P%", "FTA",
)
CATEGORICAL_FEATURES = ("classification", "Pos")
FEATURES = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)


def load_pool(year: int) -> pd.DataFrame:
    """Load only prospect-side columns; scoring does not require draft results."""
    prospects = pd.read_csv(DATA_PROCESSED_ROOT / str(year) / f"draftpool_stats_{year}.csv")
    missing_columns = set(("Name", *FEATURES)) - set(prospects.columns)
    if missing_columns:
        raise ValueError(f"Missing prospect features for {year}: {sorted(missing_columns)}")
    keys = prospects["Name"].map(player_key)
    if keys.eq("").any() or keys.duplicated().any():
        raise ValueError(f"Prospect names are blank or ambiguous in {year}")
    result = prospects.loc[:, ["Name", *FEATURES]].copy()
    result.insert(1, "year", year)
    return result


def load_class(year: int) -> pd.DataFrame:
    """Attach one inclusion label per prospect and validate historical coverage."""
    if year not in YEARS:
        raise ValueError(f"No historical labels configured for {year}")
    prospects = load_pool(year)
    draft = pd.read_csv(DATA_RAW_ROOT / str(year) / f"draft_{year}.csv")
    lottery = draft.loc[draft["pick"].between(1, 14), "player"]
    if len(lottery) != 14:
        raise ValueError(f"Expected 14 lottery picks for {year}, found {len(lottery)}")

    keys = prospects["Name"].map(player_key)
    lottery_keys = lottery.map(player_key)
    if lottery_keys.duplicated().any():
        raise ValueError(f"Duplicate lottery names in {year}")
    missing = sorted(set(lottery_keys) - set(keys))
    if missing:
        raise ValueError(f"Lottery players missing from {year} prospect pool: {missing}")

    result = prospects.copy()
    result["lottery"] = keys.isin(set(lottery_keys)).astype(int)
    return result


def make_pipeline() -> Pipeline:
    """Use a fixed, modest model; preprocessing is fitted only on train years."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    transform = ColumnTransformer([
        ("numeric", numeric, list(NUMERIC_FEATURES)),
        ("categorical", categorical, list(CATEGORICAL_FEATURES)),
    ])
    return Pipeline([
        ("features", transform),
        ("classifier", LogisticRegression(C=0.1, max_iter=2000)),
    ])


def quality_baseline(year: int) -> pd.Series:
    """Return the existing four-stat quality score, with no team-fit term."""
    prospects = pd.read_csv(DATA_PROCESSED_ROOT / str(year) / f"draftpool_stats_{year}.csv")
    teams = pd.read_csv(DATA_PROCESSED_ROOT / str(year) / f"draft_{year}_team_stats.csv")
    ranked = rank_candidates(teams, prospects, teams.iloc[0]["team"])
    return pd.Series({candidate.player: sum(
        value for name, value in candidate.contributions.items()
        if name.startswith("quality:")
    ) for candidate in ranked.candidates})


def _hit_count(scores: np.ndarray, labels: np.ndarray, names: np.ndarray) -> int:
    selected = np.lexsort((names, -scores))[:14]
    return int(labels[selected].sum())


def backtest() -> pd.DataFrame:
    """Expanding-window evaluation; every test year follows all train years."""
    classes = {year: load_class(year) for year in YEARS}
    rows = []
    for test_year in YEARS[2:]:
        train_years = tuple(year for year in YEARS if year < test_year)
        train = pd.concat([classes[year] for year in train_years], ignore_index=True)
        test = classes[test_year]
        model = make_pipeline().fit(train[list(FEATURES)], train["lottery"])
        scores = model.predict_proba(test[list(FEATURES)])[:, 1]
        quality = test["Name"].map(quality_baseline(test_year)).to_numpy(dtype=float)
        labels = test["lottery"].to_numpy(dtype=int)
        rows.append({
            "test_year": test_year,
            "train_through": max(train_years),
            "pool_size": len(test),
            "model_hits_at_14": _hit_count(scores, labels, test["Name"].to_numpy()),
            "quality_hits_at_14": _hit_count(quality, labels, test["Name"].to_numpy()),
            "random_expected_hits_at_14": 196 / len(test),
            "model_average_precision": average_precision_score(labels, scores),
            "quality_average_precision": average_precision_score(labels, quality),
        })
    return pd.DataFrame(rows)


def train_through(year: int = 2023) -> dict:
    """Fit an artifact on historical classes before the forecast year."""
    years = tuple(candidate for candidate in YEARS if candidate <= year)
    if len(years) < 2 or year >= max(YEARS):
        raise ValueError("Training requires at least two classes and a later holdout year")
    train = pd.concat([load_class(candidate) for candidate in years], ignore_index=True)
    model = make_pipeline().fit(train[list(FEATURES)], train["lottery"])
    return {"version": MODEL_VERSION, "train_years": years, "model": model}


def forecast(year: int, artifact: dict) -> pd.DataFrame:
    """Rank a later historical class; refuse to score a training year."""
    if artifact.get("version") != MODEL_VERSION or year <= max(artifact["train_years"]):
        raise ValueError("Model artifact is incompatible with or trained on this year")
    prospects = load_pool(year)
    scores = artifact["model"].predict_proba(prospects[list(FEATURES)])[:, 1]
    result = prospects[["Name", "classification"]].copy()
    result["model_score"] = scores
    result = result.sort_values(["model_score", "Name"], ascending=[False, True]).reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result


def assign_teams(year: int, ranked: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Assign a fixed model top 14 using team fit, without changing inclusion."""
    selected = ranked.head(14)["Name"].tolist()
    if len(selected) != 14 or len(set(selected)) != 14:
        raise ValueError("Team assignment needs 14 distinct model selections")
    players, teams = load_historical_data(year)
    unavailable = set(players["Name"]) - set(selected)
    if len(unavailable) != len(players) - 14:
        raise ValueError("Model selections must be present in the prospect pool")
    taken: set[str] = set()
    picks = []
    for team in teams["team"]:
        ranking = rank_candidates(teams, players, team, excluded_players=frozenset(unavailable | taken))
        choice = ranking.candidates[0]
        picks.append({"Team": team, "Player": choice.player, "FitScore": choice.score})
        taken.add(choice.player)
    result = pd.DataFrame(picks)
    actual = pd.read_csv(DATA_RAW_ROOT / str(year) / f"draft_{year}.csv")
    return result, evaluate_lottery(result, actual)


def main() -> None:
    report = backtest()
    artifact = train_through(2023)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    BACKTEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    report.to_csv(BACKTEST_PATH, index=False)
    print(report.to_string(index=False))
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved backtest: {BACKTEST_PATH}")


if __name__ == "__main__":
    main()
