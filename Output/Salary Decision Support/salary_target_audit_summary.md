# Deliverable 3 Salary Target Audit

## Inputs

- `ceiling_comp_engine`: `C:\Users\Ethan Yao\Downloads\STAT-946-Case-Study-3\Output\Player Archetype Analysis\ceiling_comps_hof.csv`
- `comp_salary_detail`: `C:\Users\Ethan Yao\Downloads\STAT-946-Case-Study-3\Output\Player Archetype Analysis\SalaryBlock\comp_salary_detail_table.csv`
- `deliverable3_block2_context`: `C:\Users\Ethan Yao\Downloads\STAT-946-Case-Study-3\Output\Player Archetype Analysis\SalaryBlock\deliverable3_block2_archetype_comp_market_context.csv`
- `final_player_profile`: `C:\Users\Ethan Yao\Downloads\STAT-946-Case-Study-3\Output\Player Archetype Analysis\final_player_archetype_profile_table.csv`
- `hybrid_archetype_table`: `C:\Users\Ethan Yao\Downloads\STAT-946-Case-Study-3\Output\Player Archetype Analysis\player_hybrid_archetype_table.csv`
- `identity_drift_table`: `C:\Users\Ethan Yao\Downloads\STAT-946-Case-Study-3\Output\Player Archetype Analysis\player_identity_drift_table.csv`
- `macro_archetype_backbone`: `C:\Users\Ethan Yao\Downloads\STAT-946-Case-Study-3\Output\Player Archetype Analysis\archetype_macro_player_table.csv`
- `realistic_comp_engine`: `C:\Users\Ethan Yao\Downloads\STAT-946-Case-Study-3\Output\Player Archetype Analysis\realistic_comps.csv`
- `report_ready_dossier_manifest`: `C:\Users\Ethan Yao\Downloads\STAT-946-Case-Study-3\Output\Player Archetype Analysis\player_dossier_demo\report_ready_player_dossier_manifest.csv`
- `report_ready_dossier_selection`: `C:\Users\Ethan Yao\Downloads\STAT-946-Case-Study-3\Output\Player Archetype Analysis\player_dossier_demo\report_ready_player_dossier_selection.csv`
- `report_ready_dossier_table`: `C:\Users\Ethan Yao\Downloads\STAT-946-Case-Study-3\Output\Player Archetype Analysis\player_dossier_demo\report_ready_player_dossier_table.csv`
- `report_ready_dossiers_markdown`: `C:\Users\Ethan Yao\Downloads\STAT-946-Case-Study-3\Output\Player Archetype Analysis\player_dossier_demo\report_ready_player_dossiers.md`
- `year5_salary_merge_summary`: `C:\Users\Ethan Yao\Downloads\STAT-946-Case-Study-3\Output\Player Archetype Analysis\SalaryBlock\year5_salary_merge_summary.csv`
- `year5_salary_target`: `C:\Users\Ethan Yao\Downloads\STAT-946-Case-Study-3\Output\Player Archetype Analysis\SalaryBlock\year5_salary_target_table.csv`
- `year5_salary_unmatched_diagnostic`: `C:\Users\Ethan Yao\Downloads\STAT-946-Case-Study-3\Output\Player Archetype Analysis\SalaryBlock\year5_salary_unmatched_diagnostic.csv`

## Summary metrics

| Metric group | Metric | Value | Notes |
|---|---|---:|---|
| merge_summary_input | total_players | 1058.000 | Reported directly from year5_salary_merge_summary.csv. |
| merge_summary_input | players_with_year5_salary_cap_match | 665.000 | Reported directly from year5_salary_merge_summary.csv. |
| merge_summary_input | players_without_year5_salary_cap_match | 393.000 | Reported directly from year5_salary_merge_summary.csv. |
| merge_summary_input | year5_salary_match_rate | 0.628544 | Reported directly from year5_salary_merge_summary.csv. |
| merge_summary_input | players_matched_via_manual_alias | 0.000000 | Reported directly from year5_salary_merge_summary.csv. |
| merge_summary_input | players_with_multirow_salary_season | 21.000 | Reported directly from year5_salary_merge_summary.csv. |
| merge_summary_input | players_with_multiteam_salary_season | 21.000 | Reported directly from year5_salary_merge_summary.csv. |
| recomputed_target_summary | total_players_from_target_table | 1058.000 | Row count from year5_salary_target_table.csv. |
| recomputed_target_summary | players_with_year5_salary_cap_match | 665.000 | Computed from year5_salary_match_flag == 1. |
| recomputed_target_summary | players_without_year5_salary_cap_match | 393.000 | Computed from year5_salary_match_flag != 1. |
| recomputed_target_summary | year5_salary_match_rate | 0.628544 | Computed from year5_salary_match_flag == 1. |
| salary_cap_pct_distribution | matched_cap_pct_min | 0.000093 | Minimum matched Year-5 salary as cap percentage. |
| salary_cap_pct_distribution | matched_cap_pct_p25 | 0.027267 | 25th percentile of matched Year-5 salary as cap percentage. |
| salary_cap_pct_distribution | matched_cap_pct_median | 0.075222 | Median matched Year-5 salary as cap percentage. |
| salary_cap_pct_distribution | matched_cap_pct_p75 | 0.147693 | 75th percentile of matched Year-5 salary as cap percentage. |
| salary_cap_pct_distribution | matched_cap_pct_max | 0.300000 | Maximum matched Year-5 salary as cap percentage. |
| salary_cap_pct_distribution | matched_cap_pct_mean | 0.094976 | Mean matched Year-5 salary as cap percentage. |
