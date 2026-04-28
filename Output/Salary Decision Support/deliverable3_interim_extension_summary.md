# Deliverable 3 Interim Extension Guidance

This export assembles the forecast-adjusted **interim** extension-guidance layer from the Phase 7 reconciliation table.
The status remains **interim_with_current_D1_macro_forecast** and must not be treated as the final extension recommendation until durability and richer uncertainty outputs are integrated.

## Inputs

- `deliverable3_salary_reconciliation_table.csv`: `/home/runner/work/STAT-946-Case-Study-3/STAT-946-Case-Study-3/Output/Salary Decision Support/deliverable3_salary_reconciliation_table.csv`

## Case-study snippets

### Andrew Bynum — Wait and save flexibility

- **Player:** Andrew Bynum
- **Draft year:** 2005
- **Hybrid archetype:** Scoring Bigs / Two-Way Forwards | Arc-Heavy Spacers
- **Interim stance:** Wait and save flexibility
- **Market band:** Protected around 0.072 of cap, fair around 0.112, walk-away max around 0.173.
- **Interim posture:** Open around 7.2%, target around 7.2%, hard max around 11.2%.
- **Forecast support:** no_forecast_overlay
- **Why:** Wait and save flexibility: comp support is moderate; no forecast overlay is available, so the salary stance should stay conservative; the archetype can justify some premium, but only selectively; the baseline salary model supports the fair-to-upper part of the market band; drift remains role_shifting_materially; forecast confidence is forecast_unavailable.
- **Status note:** Status: interim_with_current_D1_macro_forecast. This is not the final extension recommendation; durability / availability risk, richer uncertainty bands, and the fully locked Deliverable 1 merge table are still pending.

### Nikola Vučević — Offer now at disciplined price

- **Player:** Nikola Vučević
- **Draft year:** 2011
- **Hybrid archetype:** Scoring Bigs / Two-Way Forwards | Midrange Shot Creators
- **Interim stance:** Offer now at disciplined price
- **Market band:** Protected around 0.137 of cap, fair around 0.188, walk-away max around 0.229.
- **Interim posture:** Open around 13.7%, target around 18.8%, hard max around 18.8%.
- **Forecast support:** no_forecast_overlay
- **Why:** Offer now, but stay disciplined: comp support is strong; no forecast overlay is available, so the salary stance should stay conservative; the archetype can justify some premium, but only selectively; the baseline salary model supports the lower-to-middle part of the market band; drift remains role_shifting_materially; forecast confidence is forecast_unavailable.
- **Status note:** Status: interim_with_current_D1_macro_forecast. This is not the final extension recommendation; durability / availability risk, richer uncertainty bands, and the fully locked Deliverable 1 merge table are still pending.

### Trae Young — Offer now at disciplined price

- **Player:** Trae Young
- **Draft year:** 2018
- **Hybrid archetype:** High-Usage Primary Creators | Balanced Three-Level Scorers
- **Interim stance:** Offer now at disciplined price
- **Market band:** Protected around 0.250 of cap, fair around 0.253, walk-away max around 0.253.
- **Interim posture:** Open around 25.0%, target around 25.3%, hard max around 25.3%.
- **Forecast support:** no_forecast_overlay
- **Why:** Offer now, but stay disciplined: comp support is strong; no forecast overlay is available, so the salary stance should stay conservative; role scarcity supports acting earlier than usual; the baseline salary model sits above the current walk-away max, so the market band should not be expanded automatically; drift remains role_shifting_materially; forecast confidence is forecast_unavailable.
- **Status note:** Status: interim_with_current_D1_macro_forecast. This is not the final extension recommendation; durability / availability risk, richer uncertainty bands, and the fully locked Deliverable 1 merge table are still pending.

## Summary metrics

| Metric group | Metric | Value | Notes |
|---|---|---:|---|
| workflow_status | interim_guidance_rows | 1058 | Total player-level rows in the interim forecast-adjusted extension guidance layer. |
| workflow_status | forecast_overlay_available_rows | 0 | Rows carrying the current Deliverable 1 macro forecast overlay from Phase 7. |
| workflow_status | demo_case_study_rows | 3 | Rows reserved for Trae Young / Andrew Bynum case-study guidance when present. |
| interim_extension_bucket | offer_now_disciplined_band | 474 | Counts by interim forecast-adjusted extension bucket. |
| interim_extension_bucket | wait_and_save_flexibility | 343 | Counts by interim forecast-adjusted extension bucket. |
| interim_extension_bucket | avoid_overcommitting | 241 | Counts by interim forecast-adjusted extension bucket. |
| forecast_support_tier | no_forecast_overlay | 1058 | Counts by current Deliverable 1 macro forecast support tier. |
| pricing_anchor_preference | market_band_confirmed_by_model | 380 | Counts by how the salary model and market band align after reconciliation. |
| pricing_anchor_preference | model_discount_vs_market_review | 328 | Counts by how the salary model and market band align after reconciliation. |
| pricing_anchor_preference | model_premium_vs_market_review | 189 | Counts by how the salary model and market band align after reconciliation. |
| pricing_anchor_preference | baseline_model_only | 117 | Counts by how the salary model and market band align after reconciliation. |
| pricing_anchor_preference | mixed_review | 44 | Counts by how the salary model and market band align after reconciliation. |
