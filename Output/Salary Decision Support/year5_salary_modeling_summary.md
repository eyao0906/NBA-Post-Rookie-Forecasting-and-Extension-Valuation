# Year-5 Salary Modeling Table Summary

This phase prepares a leakage-aware baseline modeling table for Year-5 salary-cap regression.
It uses the canonical target plus reusable Deliverable 2 predictors already available in the project bundle.

## Summary metrics

| Metric group | Metric | Value | Notes |
|---|---|---:|---|
| workflow_status | players_total | 1058 | Total player rows carried into the modeling-prep table. |
| workflow_status | columns_total | 117 | Total columns in the modeling-prep table. |
| target_status | target_observed_rows | 665 | Rows with observed Year-5 salary-cap target available. |
| target_status | target_missing_rows | 393 | Rows without an observed Year-5 salary-cap target. |
| training_status | holdout_rows | 3 | Reserved demo rows excluded from model fitting. |
| training_status | training_eligible_rows | 662 | Rows eligible for baseline salary-model training after excluding demo holdouts. |
| feature_inventory | recommended_numeric_predictors | 78 | Numeric predictors available before Year 5. |
| feature_inventory | recommended_categorical_predictors | 7 | Categorical predictors available before Year 5 that will need encoding later. |
| feature_inventory | max_missing_pct_recommended_predictors | 0.457467 | Maximum missingness across recommended predictor columns. |

## Notes

- This phase uses the richest reusable Deliverable 2 predictor backbone currently available in the uploaded project bundle: player_identity_drift_table.csv.
- No comp-salary anchors or provisional action buckets are used as model predictors here, because those would contaminate the baseline salary-estimation module with target-adjacent context.
- Trae Young and Nikola Vučević remain reserved as case-study demos and are explicitly excluded from training via training_eligible_flag = 0.
- The original flattened Seasons 1–4 player-feature table is not part of the current uploaded Deliverable 3 file bundle, so this phase reuses the saved macro-role / shot-style / drift backbone instead.
