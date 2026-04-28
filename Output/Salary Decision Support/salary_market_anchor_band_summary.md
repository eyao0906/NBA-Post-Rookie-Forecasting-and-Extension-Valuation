# Deliverable 3 Market Anchor Band

## Inputs

- `comp_salary_detail`: `/home/runner/work/STAT-946-Case-Study-3/STAT-946-Case-Study-3/Output/Player Archetype Analysis/SalaryBlock/comp_salary_detail_table.csv`
- `deliverable3_block2_context`: `/home/runner/work/STAT-946-Case-Study-3/STAT-946-Case-Study-3/Output/Player Archetype Analysis/SalaryBlock/deliverable3_block2_archetype_comp_market_context.csv`
- `salary_demo_player_holdout_manifest`: `/home/runner/work/STAT-946-Case-Study-3/STAT-946-Case-Study-3/Output/Salary Decision Support/salary_demo_player_holdout_manifest.csv`

## Summary metrics

| Metric group | Metric | Value | Notes |
|---|---|---:|---|
| anchor_table | players_in_block2_context | 1058.000 | Row count from deliverable3_block2_archetype_comp_market_context.csv. |
| anchor_table | players_with_supported_market_anchor_band | 941.000 | Players with non-missing protected, fair, and walk-away prices. |
| anchor_table | supported_market_anchor_share | 0.889414 | Share of Block 2 players with a usable market anchor band. |
| anchor_table | holdout_demo_players_flagged | 3.000000 | Reserved demo players carried forward for later salary-suggestion case studies. |
| supported_distribution | protected_price_p25 | 0.015758 | Protected price from the comp-based market anchor. |
| supported_distribution | protected_price_median | 0.034192 | Protected price from the comp-based market anchor. |
| supported_distribution | protected_price_p75 | 0.064678 | Protected price from the comp-based market anchor. |
| supported_distribution | fair_price_p25 | 0.046177 | Fair price from the comp-based market anchor. |
| supported_distribution | fair_price_median | 0.073333 | Fair price from the comp-based market anchor. |
| supported_distribution | fair_price_p75 | 0.110643 | Fair price from the comp-based market anchor. |
| supported_distribution | walk_away_price_p25 | 0.062178 | Walk-away max from the comp-based market anchor. |
| supported_distribution | walk_away_price_median | 0.101483 | Walk-away max from the comp-based market anchor. |
| supported_distribution | walk_away_price_p75 | 0.160514 | Walk-away max from the comp-based market anchor. |
| supported_distribution | anchor_band_width_p25 | 0.028407 | Absolute width of the market anchor band. |
| supported_distribution | anchor_band_width_median | 0.054844 | Absolute width of the market anchor band. |
| supported_distribution | anchor_band_width_p75 | 0.091854 | Absolute width of the market anchor band. |
| supported_distribution | anchor_band_relative_width_p25 | 0.420122 | Band width divided by fair price. |
| supported_distribution | anchor_band_relative_width_median | 0.781483 | Band width divided by fair price. |
| supported_distribution | anchor_band_relative_width_p75 | 1.255969 | Band width divided by fair price. |
