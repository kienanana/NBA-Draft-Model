# NBA Draft Lab

NBA Draft Lab explores prospect data and tests whether a model can identify the 14 players selected in each NBA draft lottery. Its main learned model ranks prospects using only player information. A separate, explainable team-fit simulator assigns prospects to teams; exact team matches are a secondary metric and do not affect model training.

The repository also includes prospect profiles, playstyle clusters, a Streamlit app, and a FastAPI service for historical team rankings. It is a retrospective research prototype, not a live draft forecasting service.

## Start locally

From the repository root:

```bash
conda env create -f env/environment.dev.yml
conda activate nba-draft
python -m pip install -r requirements.txt
python -m pytest -q tests
python -m streamlit run app/streamlit_app.py
```

The **Lottery Forecast** tab shows the model's top 14 for 2024 and its year-by-year backtest. The included model artifact was trained on 2020–2023; 2024 was excluded from its training. To rebuild the artifact and report:

```bash
python -m ml.lottery_inclusion
```

This writes `models/lottery_inclusion_v1.joblib` and `reports/lottery_inclusion_backtest.csv`. Training is deterministic with a fixed feature set and model settings. The [model notes](docs/lottery-model.md) give the target, features, evaluation method, and limitations.

The other tabs provide prospect profiles, clusters, the original team-fit draft simulation, and team rankings. Team rankings require the API in a second terminal:

```bash
python -m uvicorn api.main:app --reload
```

The API uses `http://127.0.0.1:8000` by default. Set `DRAFTLAB_API_URL` to use another address. `GET /api/v1/teams?year=2024` lists historical teams; `GET /api/v1/rankings?year=2024&team=Atlanta%20Hawks` returns explainable team-fit rankings. See the [team scoring contract](docs/scoring-contract.md) for its inputs and interpretation.

## Data and evaluation

The checked-in prospect pools contain roughly 70 players per year for 2020–2024, assembled from college, international, G League, and Overtime Elite data. `data/processed/<year>/draftpool_stats_<year>.csv` supplies model features; `data/raw/<year>/draft_<year>.csv` supplies top-14 labels. Collection and cleaning work remains in `notebooks/`. Name aliases are resolved before assigning labels or evaluating the simulation.

The main metric is **lottery hits@14**: among the model's 14 highest-ranked prospects, how many were actually picked in the lottery? Average precision measures ranking over the full pool. The backtest trains on earlier years only and evaluates 2022, 2023, and 2024 separately. It compares the learned model with the existing four-stat quality score and the expected count from random selection. Team-match accuracy belongs only to the separate simulation.

| Test year | Model hits@14 | Quality-score hits@14 | Random expected hits@14 |
| --- | ---: | ---: | ---: |
| 2022 | 4 | 5 | 2.7 |
| 2023 | 9 | 6 | 2.8 |
| 2024 | 8 | 8 | 2.6 |

These are small historical samples. More seriously, the first 14 names in every saved `data/raw/<year>/prospect_pool_<year>.csv` exactly match that year's actual draft order, and the files have no verified pre-draft snapshot dates. **Neither the candidate pools nor these scores should be represented as prospective pre-draft validation.** The model does not use the pool's row order, draft pick, team, or other outcomes as features. The saved pool order also cannot serve as a mock-draft benchmark. A comparison with public mock rankings needs dated, genuinely pre-draft snapshots.

## Project structure

- `ml/lottery_inclusion.py`: labeled data, training, historical backtest, and ranking.
- `ml/scoring.py` and `ml/simulation.py`: original team-fit scorer and pick-by-pick simulation.
- `app/streamlit_app.py` and `api/main.py`: user interface and team-ranking API.
- `data/`, `models/`, `reports/`: historical inputs and generated outputs.
- `tests/`: label, temporal split, scoring, simulation, API, and profile checks.

The immediate research priority is a dated pre-draft candidate universe and mock-rank snapshots. More historical classes are needed before selecting a more flexible model or claiming stable performance.
