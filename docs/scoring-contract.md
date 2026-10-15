# Scoring contract: baseline-v1

The active draft simulation and ranking API use the same function in `ml/scoring.py`.
The score is a **relative ranking signal**, not a draft probability or an NBA
outcome prediction. Earlier optimized weight files belong to a different score
definition and are not loaded by this version.

## Team needs

For each selected stat, the historical team's value is compared with the
**unique teams in that year's lottery order**. A team with two picks enters the
reference mean once. Values are converted to z-scores, then flipped when a lower
value is better. Only below-average oriented values become needs. Need weights
are normalized to sum to 1. If no need is identified, the score uses only the
prospect-quality term.

| Team column | Prospect column | Better team value | Better prospect value |
| --- | --- | --- | --- |
| `ORB%` | `ORB%` | Higher | Higher |
| `DRB%` | `DRB%` | Higher | Higher |
| `TOV%` | `TOV%` | Lower | Lower |
| `TS%` | `TS%` | Higher | Higher |
| `ORtg` | `ORTG` | Higher | Higher |
| `DRtg` | `DRTG` | Lower | Lower |

The source table groups `DRB%` under defensive Four Factors, alongside
opponent shooting and turnover columns. It is the team's **own** defensive
rebounding percentage, so it can be paired with a prospect's `DRB%`. The
flattened CSV loses that grouped heading; nearby columns must be interpreted
individually. Other columns in the old needs list are absent from the team
CSVs. Their intended mapping needs source-level validation before use.

## Prospect values and score

Each prospect stat is scaled to 0–1 within its **college** or **non-college**
draft-class cohort. Lower-is-better stats are flipped, so 1 always means a more
favorable observed value. A constant stat gets 0.5. A missing stat also gets
0.5 and adds a warning to that prospect's result. This neutral imputation is a
product choice, not an estimate of the missing measurement.

The quality term averages `TS%` (higher), `PER` (higher), `AST/TO` (higher), and
`Age` (lower). For the default quality weight of 0.2:

```text
score = sum(team_need_weight × scaled_prospect_stat)
      + 0.2 × mean(scaled_TS%, scaled_PER, scaled_AST/TO, scaled_Age)
```

The API returns each addend in `contributions`; their sum equals `score`.
Changing the quality weight scales its four contributions without changing
team-need weights. Scores from different years are not directly comparable.
Different leagues and small cohorts remain a source of uncertainty even after
within-cohort scaling.

## Evaluation

The simulation reports team-match rate, lottery hit rate, reciprocal rank of
the first real lottery player in its picks, and nDCG@14. Unselected real lottery
players contribute zero relevance to nDCG. These historical years were used in
earlier optimization work, so current results must not be presented as held-out
predictive performance. A time-based split and simple baselines are next steps.
