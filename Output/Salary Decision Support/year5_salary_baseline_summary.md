# Year-5 Salary Baseline Models

Selected model: **ridge_alpha_10**

Training rows used: **662**

## Grouped out-of-fold comparison

| model_name           |   n_oof |      rmse |       mae |     medae |          r2 |   rank_rmse |
|:---------------------|--------:|----------:|----------:|----------:|------------:|------------:|
| ridge_alpha_10       |     662 | 0.0397933 | 0.0304471 | 0.0240689 |  0.727059   |           1 |
| ridge_alpha_1        |     662 | 0.0401556 | 0.0305976 | 0.0233505 |  0.722066   |           2 |
| linear_regression    |     662 | 0.0408124 | 0.0310519 | 0.0250414 |  0.712899   |           3 |
| macro_archetype_mean |     662 | 0.0559527 | 0.0441306 | 0.0357543 |  0.460375   |           4 |
| global_mean          |     662 | 0.0762072 | 0.0642774 | 0.0614074 | -0.00101914 |           5 |

## Demo holdout predictions

| PLAYER_NAME    |   draft_year | baseline_selected_model   |   predicted_year5_salary_cap_pct | holdout_reason                                                                                 |
|:---------------|-------------:|:--------------------------|---------------------------------:|:-----------------------------------------------------------------------------------------------|
| Andrew Bynum   |         2005 | ridge_alpha_10            |                         0.164762 | Reserved salary suggestion demo player; exclude from future salary-model training/calibration. |
| Nikola Vučević |         2011 | ridge_alpha_10            |                         0.18448  | Reserved salary suggestion demo player; exclude from future salary-model training/calibration. |
| Trae Young     |         2018 | ridge_alpha_10            |                         0.255866 | Reserved salary suggestion demo player; exclude from future salary-model training/calibration. |
