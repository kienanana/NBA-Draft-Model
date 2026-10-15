# 🏀 NBA Draft Lab — Lottery Simulation & Prospect Profiles

Predict the NBA lottery (picks 1–14) by matching teams to prospects using historical team needs, prospect quality, and draft logic — plus interactive prospect profiles with radar charts and NBA-style comps.

> ⚠️ This is a learning-focused project. Accuracy is still modest, but the repo shows end-to-end skills: data collection & cleaning, feature engineering, clustering, simulation, evaluation, and a Streamlit frontend.

---

## ✨ What’s inside

- **Streamlit app** for profiles & simulation (`app/streamlit_app.py`)
- **ML modules** (`ml/`) for profiles, simulation, weights, playstyles, etc.
- **Data & weights** (`data/processed/<year>/…`, `weights/…`)
- **Notebooks** for scraping / exploration (`notebooks/`)
- **Reproducible env** via `requirements.txt` / `environment.dev.yml`

---

## 📊 Project scope

**Objective.** Simulate the lottery part of the NBA Draft: for each pick, score all remaining team–prospect pairs, select the best fit, remove the player, and continue until pick 14.

Two key surfaces:

1. **Prospect Profiles**
   - Per-game & advanced stat **radar charts**
   - **Z-score → min–max** scaling for fair visual comparison
   - Optional **Upside** and **UpdatedOverall** scores
   - **NBA comps & archetype** when comps CSV exists

2. **Draft Simulation (2020-2024)**
   - Uses the direction-aware **baseline-v1** scorer
   - Reports **Team-match accuracy**, **Lottery hit rate**, **first-hit reciprocal rank**, and **nDCG@14**

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
