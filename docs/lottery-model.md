# Lottery inclusion model

**Target:** For a player in a historical candidate pool, predict selection in picks 1–14. The model ranks every player in a class and selects the top 14. It does not predict the team or exact pick.

**Inputs:** Age; games and minutes; points, rebounds, assists, steals, blocks, turnovers, three-point attempts and percentage, free-throw attempts, true shooting percentage, PER, classification, and position. All are taken from `draftpool_stats_<year>.csv`. Player name, draft result, team, and pool row order are excluded. The numeric features receive training-set median imputation and standardization; classification and position receive one-hot encoding. A fixed L2-regularized logistic regression (`C=0.1`) learns from lottery labels. Scores order candidates; their calibration has not been established.

**Training and test:** `python -m ml.lottery_inclusion` runs an expanding-window backtest: 2020–2021 → 2022, 2020–2022 → 2023, and 2020–2023 → 2024. Preprocessing is fitted inside each training fold. The command then saves the 2020–2023 model for the app's 2024 historical replay. Settings were fixed for this first experiment; there is no hyperparameter search against the 2024 result.

**Metrics:** Hits@14 is the primary measure; dividing it by 14 gives recall@14 and precision@14. Average precision measures the full-class ranking. The report also contains a four-stat quality-score baseline and random expected hits, `14 × 14 / pool size`. Exact team matches are supplementary and never enter training or model selection.

The app can also assign the model's fixed top 14 to historical teams using the original team-fit scorer. Its exact team-match rate is supplementary and cannot change which 14 players the model selected.

**Identity checks:** Names are normalized for accents and punctuation, with explicit aliases for Jabari Smith Jr./Jabari Smith and Alex Sarr/Alexandre Sarr. Every historical class must have exactly 14 unique matched lottery labels. The same normalization fixes name matching in the old simulation evaluator.

**Limits:** Five draft years give only 70 positive examples. Candidate lists have no verified pre-draft snapshot dates; their first 14 rows exactly reproduce actual draft order. Although the model ignores row order, the candidate universe may reflect post-draft curation. Reported backtests measure ranking within those saved pools, not live forecasting ability. A public mock baseline is deferred until dated pre-draft snapshots are available. Differences between college and other leagues, sparse player samples, and missing values remain sources of error.
