# Deliverable 3 Staged Final Framework

This export completes Deliverable 3 as a staged salary-decision-support system rather than as a single fully validated final recommendation layer.

## What is complete now
- Block 2 market-anchor logic remains the pricing backbone.
- Salary-model and market-band reconciliation are preserved.
- Official Block 1 forecast support is now merged through the finalized handoff file.
- Forecast-adjusted posture is only activated on the supported subset with test-set performance forecast support.

## What stays conditional
- Full-cohort final extension guidance remains unvalidated.
- Durability / availability risk is still not integrated as a separate resolved module.
- The current uncertainty input is still the Block 1 confidence proxy, not a richer interval package.

## Core counts

| Metric | Value | Notes |
|---|---:|---|
| framework_rows | 1058 | Total player rows in the staged Deliverable 3 framework. |
| market_anchor_supported_rows | 941 | Rows with supported protected/fair/walk-away anchor band. |
| forecast_probabilities_available_rows | 657 | Rows with official Block 1 class probabilities merged. |
| supported_forecast_adjustment_rows | 139 | Rows eligible for supported forecast-adjusted logic. |
| salary_model_training_sample_rows | 662 | Rows eligible for salary-model training after case-study holdout exclusion. |

### Supported forecast-adjustment evaluation subset

| Metric | Value |
|---|---:|
| supported_subset_rows | 139.0 |
| supported_subset_with_salary_target | 138.0 |
| supported_subset_label_accuracy | 0.6330935251798561 |
| supported_subset_perf_rmse | 10.646770521071424 |
| supported_subset_perf_mae | 8.290989611654675 |