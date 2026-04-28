# Deliverable 3 Salary Reconciliation Layer

This phase reconciles the baseline salary model with the comp-market anchor band and optionally overlays the current Deliverable 1 macro forecast table.

Important boundary: the output remains **interim**. It is not the final extension recommendation because full Deliverable 1 uncertainty, durability, and merge-safe identifiers are still not fully integrated.

## Summary metrics

| metric_group              | metric                           |   value | notes                                                                          |
|:--------------------------|:---------------------------------|--------:|:-------------------------------------------------------------------------------|
| workflow_status           | players_total                    |    1058 | Total player-level rows in the Phase 7 reconciliation layer.                   |
| workflow_status           | market_anchor_supported          |     941 | Players with a supported comp-market anchor band carried forward from Phase 2. |
| workflow_status           | forecast_overlay_matched_players |       0 | Players successfully matched to the Deliverable 1 macro forecast table.        |
| alignment_bucket          | below_protected                  |     328 | Phase 7 count for alignment bucket.                                            |
| alignment_bucket          | between_protected_and_fair       |     248 | Phase 7 count for alignment bucket.                                            |
| alignment_bucket          | above_walkaway                   |     189 | Phase 7 count for alignment bucket.                                            |
| alignment_bucket          | between_fair_and_walkaway        |     176 | Phase 7 count for alignment bucket.                                            |
| alignment_bucket          | model_only_no_market_anchor      |     117 | Phase 7 count for alignment bucket.                                            |
| pricing_anchor_preference | market_band_confirmed_by_model   |     380 | Phase 7 count for pricing anchor preference.                                   |
| pricing_anchor_preference | model_discount_vs_market_review  |     328 | Phase 7 count for pricing anchor preference.                                   |
| pricing_anchor_preference | model_premium_vs_market_review   |     189 | Phase 7 count for pricing anchor preference.                                   |
| pricing_anchor_preference | baseline_model_only              |     117 | Phase 7 count for pricing anchor preference.                                   |
| pricing_anchor_preference | mixed_review                     |      44 | Phase 7 count for pricing anchor preference.                                   |
| forecast_outlook_bucket   | forecast_unavailable             |    1058 | Phase 7 count for forecast outlook bucket.                                     |
| forecast_overlay_signal   | no_forecast_overlay              |    1058 | Phase 7 count for forecast overlay signal.                                     |

## Case-study extract

| PLAYER_NAME    |   draft_year | decision_card_label                                                                             | decision_card_band_text                                                       |   predicted_year5_salary_cap_pct |   predicted_class | pricing_anchor_preference      | forecast_overlay_signal   |
|:---------------|-------------:|:------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------|---------------------------------:|------------------:|:-------------------------------|:--------------------------|
| Andrew Bynum   |         2005 | Andrew Bynum — Scoring Bigs / Two-Way Forwards | Protected 0.072, fair 0.112, walk-away 0.173   | Protected around 0.072 of cap, fair around 0.112, walk-away max around 0.173. |                         0.164762 |               nan | market_band_confirmed_by_model | no_forecast_overlay       |
| Nikola Vučević |         2011 | Nikola Vučević — Scoring Bigs / Two-Way Forwards | Protected 0.137, fair 0.188, walk-away 0.229 | Protected around 0.137 of cap, fair around 0.188, walk-away max around 0.229. |                         0.18448  |               nan | market_band_confirmed_by_model | no_forecast_overlay       |
| Trae Young     |         2018 | Trae Young — High-Usage Primary Creators | Protected 0.250, fair 0.253, walk-away 0.253         | Protected around 0.250 of cap, fair around 0.253, walk-away max around 0.253. |                         0.255866 |               nan | model_premium_vs_market_review | no_forecast_overlay       |

## Forecast overlay note

Forecast overlay matched players: **0** out of the salary backbone.
The currently available Deliverable 1 macro table is useful as an overlay because it contains class probabilities and a predicted performance score, but it still should not be treated as full final Deliverable 1 integration by itself.
