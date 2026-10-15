# 🏀 NBA Draft Lab — Lottery Simulation & Prospect Profiles

Predict the NBA lottery (picks 1–14) by matching teams to prospects using historical team needs, prospect quality, and draft logic — plus interactive prospect profiles with radar charts and NBA-style comps.

> ⚠️ This is a learning-focused project. Accuracy is still modest, but the repo shows end-to-end skills: data collection & cleaning, feature engineering, clustering, simulation, evaluation, and a Streamlit frontend.

---

## ✨ What’s inside

- **Streamlit app** for profiles, simulation, candidate rankings and comparisons (`app/streamlit_app.py`)
- **FastAPI service** for historical team rankings (`api/main.py`)
- **Shared scoring engine** with contribution breakdowns and missing-data warnings (`ml/scoring.py`)
- **Data & legacy weights** (`data/processed/<year>/…`, `weights/…`)
- **Notebooks** for scraping / exploration (`notebooks/`)
- **Environment and tests** via `requirements.txt`, `env/environment.dev.yml`, and `tests/`

## Run locally

From the repository root, create the environment once and install the current
application dependencies:

```bash
conda env create -f env/environment.dev.yml
conda activate nba-draft
python -m pip install -r requirements.txt
python -m pytest -q tests
```

Start the API and Streamlit in separate terminals with the environment active:

```bash
python -m uvicorn api.main:app --reload
```

```bash
python -m streamlit run app/streamlit_app.py
```

The ranking view uses `http://127.0.0.1:8000` by default. Set
`DRAFTLAB_API_URL` if the API runs elsewhere. FastAPI's interactive docs are at
`http://127.0.0.1:8000/docs`.

## Ranking API

`GET /api/v1/teams?year=2024` lists teams for the selected historical draft.
`GET /api/v1/rankings?year=2024&team=Atlanta%20Hawks` returns ranked candidates,
their total scores, every additive contribution, missing-data warnings, team
need weights, and the scoring version. `limit` (1–100, default 100) and
`composite_weight` (0–1, default 0.2) are optional query parameters.

Invalid requests return **422**, an unknown team returns **404**, and unavailable
historical files return **503**. The [scoring contract](docs/scoring-contract.md)
documents direction, scaling, missing values, and evaluation. The API currently
uses historical year and team selection; it does not accept custom team stats.

---

## 📊 Project scope

**Objective.** Simulate the lottery part of the NBA Draft: for each pick, score all remaining team–prospect pairs, select the best fit, remove the player, and continue until pick 14.

Three key surfaces:

1. **Prospect Profiles**
   - Per-game & advanced stat **radar charts**
   - **Z-score → min–max** scaling for fair visual comparison
   - Optional **Upside** and **UpdatedOverall** scores
   - **NBA comps & archetype** when comps CSV exists

2. **Draft Simulation (2020-2024)**
   - Uses the direction-aware **baseline-v1** scorer
   - Reports **Team-match accuracy**, **Lottery hit rate**, **first-hit reciprocal rank**, and **nDCG@14**

3. **Team Rankings (2020-2024)**
   - Ranks all candidates for a selected historical team through the API
   - Explains a candidate's score and compares two candidates' contributions

---

## 🧱 Data sources & scraping

Data is built via notebooks and scripts under `notebooks/`, saved as yearly processed CSVs under `data/processed/<year>/…`.

- **Prospects (college, international, OTE, G League Ignite):**
  Scraped from public sites (e.g., RealGM, NBA G League stats, OvertimeElite.com), cleaned to a unified schema.
- **Combine measurements:**
  Pulled from public NBA stats pages.
- **Mock draft signal:**
  **Tankathon** profiles used to fill gaps.
- **Team context:**
  Season summaries (ratings, record, depth, timeline) compiled per year.

Processed outputs:
- `draftpool_stats_<year>.csv` (core dataset)
- Optional enrichments:
  - `prospect_profiles_with_upside_<year>.csv`
  - `prospect_comps_<year>.csv`

---

## 🧠 Modeling & techniques

### Prospect scoring & profiles
- **Standardisation & scaling:** z-scores → min–max [0,1] for radars
- **Composite score:** weighted blend of per-game index, advanced index, and **age bonus**
- **Archetypes & comps:**
  - KMeans/PCA+KMeans clustering for playstyle tags
  - KNN-style nearest-neighbor comps to recent NBA players

### Draft simulation
- **Stateful process:** pick-by-pick, removing selected players
- **Fit features:** team needs in ORB%, TOV%, TS%, ORtg and DRtg, scored against direction-aware prospect metrics
- **Score blending:** team fit plus a small prospect-quality score from TS%, PER, AST/TO and age
- **Missing data:** neutral 0.5 contribution with a candidate warning; normalization within college and non-college cohorts
- **Evaluation metrics:** Team-match accuracy, Lottery hit rate, first-hit reciprocal rank, nDCG@14

### Optimisation & experiments
- Earlier **Bayesian optimisation** and cluster-adjustment work remains in notebooks and weight files. Those weights were fitted to the old scoring contract and are not used by baseline-v1. They need retraining before use in live rankings.
- The current historical results are a demonstration, not held-out predictive performance. A time-based evaluation is a next step.
