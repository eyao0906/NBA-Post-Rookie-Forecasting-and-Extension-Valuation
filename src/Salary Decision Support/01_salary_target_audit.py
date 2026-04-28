from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from salary_workflow_utils import (
    DEMO_PLAYERS,
    append_workflow_log,
    ensure_dir,
    get_expected_files,
    resolve_all_expected_files,
    write_json,
)

SCRIPT_NAME = "01_salary_target_audit.py"
REQUIRED_ROLES = {
    "year5_salary_target",
    "year5_salary_merge_summary",
    "year5_salary_unmatched_diagnostic",
    "final_player_profile",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the canonical Year-5 salary-cap target table, verify merge coverage, "
            "and write reproducible summary tables for downstream Deliverable 3 phases."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root containing src/, data/, Output/, and visual/ directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override for Output/Salary Decision Support.",
    )
    parser.add_argument(
        "--inventory-csv",
        type=Path,
        default=None,
        help=(
            "Optional project inventory CSV from Phase 0. If provided, the script will use it as a "
            "reference for resolved file paths when those paths exist locally."
        ),
    )
    parser.add_argument(
        "--log-name",
        type=str,
        default="salary_decision_support_workflow_log.txt",
        help="Workflow log file name written under the output directory.",
    )
    return parser.parse_args()


def load_inventory_table(inventory_path: Path | None) -> pd.DataFrame | None:
    if inventory_path is None or not inventory_path.exists():
        return None
    return pd.read_csv(inventory_path)



def resolve_role_paths(project_root: Path, inventory_df: pd.DataFrame | None) -> dict[str, Path]:
    role_to_path: dict[str, Path] = {}

    if inventory_df is not None and {"role", "resolved_path", "exists"}.issubset(inventory_df.columns):
        for _, row in inventory_df.iterrows():
            role = str(row["role"])
            raw_path = row.get("resolved_path")
            exists_flag = bool(row.get("exists", False))
            if pd.notna(raw_path) and exists_flag:
                candidate = Path(str(raw_path))
                if candidate.exists():
                    role_to_path[role] = candidate.resolve()

    if REQUIRED_ROLES.difference(role_to_path):
        for rec in resolve_all_expected_files(project_root):
            if rec.exists and rec.resolved_path is not None and rec.role not in role_to_path:
                role_to_path[rec.role] = Path(rec.resolved_path).resolve()

    missing = sorted(REQUIRED_ROLES.difference(role_to_path))
    if missing:
        specs = {spec.role: spec.relative_path for spec in get_expected_files()}
        hints = "\n".join(f"- {role}: {specs.get(role, 'unknown')}" for role in missing)
        raise FileNotFoundError(
            "Unable to resolve all required Deliverable 3 salary-audit inputs. Missing roles:\n"
            f"{hints}"
        )

    return role_to_path



def require_columns(df: pd.DataFrame, required: list[str], table_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")



def safe_mode(series: pd.Series) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return pd.NA
    mode_vals = non_null.mode()
    if mode_vals.empty:
        return pd.NA
    return mode_vals.iloc[0]



def normalize_demo_name(name: str) -> str:
    return (
        name.lower()
        .replace("č", "c")
        .replace("ć", "c")
        .replace("š", "s")
        .replace("ž", "z")
        .replace("đ", "d")
    )



def build_metric_rows(target_df: pd.DataFrame, merge_summary_df: pd.DataFrame) -> pd.DataFrame:
    matched_mask = target_df["year5_salary_match_flag"].fillna(0).astype(int) == 1
    unmatched_mask = ~matched_mask
    matched_cap = pd.to_numeric(target_df.loc[matched_mask, "year5_salary_cap_pct"], errors="coerce")

    metrics: list[dict[str, Any]] = []

    for _, row in merge_summary_df.iterrows():
        metrics.append(
            {
                "metric_group": "merge_summary_input",
                "metric": str(row["metric"]),
                "value": row["value"],
                "notes": "Reported directly from year5_salary_merge_summary.csv.",
            }
        )

    metrics.extend(
        [
            {
                "metric_group": "recomputed_target_summary",
                "metric": "total_players_from_target_table",
                "value": int(len(target_df)),
                "notes": "Row count from year5_salary_target_table.csv.",
            },
            {
                "metric_group": "recomputed_target_summary",
                "metric": "players_with_year5_salary_cap_match",
                "value": int(matched_mask.sum()),
                "notes": "Computed from year5_salary_match_flag == 1.",
            },
            {
                "metric_group": "recomputed_target_summary",
                "metric": "players_without_year5_salary_cap_match",
                "value": int(unmatched_mask.sum()),
                "notes": "Computed from year5_salary_match_flag != 1.",
            },
            {
                "metric_group": "recomputed_target_summary",
                "metric": "year5_salary_match_rate",
                "value": float(matched_mask.mean()),
                "notes": "Computed from year5_salary_match_flag == 1.",
            },
            {
                "metric_group": "salary_cap_pct_distribution",
                "metric": "matched_cap_pct_min",
                "value": float(matched_cap.min()) if not matched_cap.empty else pd.NA,
                "notes": "Minimum matched Year-5 salary as cap percentage.",
            },
            {
                "metric_group": "salary_cap_pct_distribution",
                "metric": "matched_cap_pct_p25",
                "value": float(matched_cap.quantile(0.25)) if not matched_cap.empty else pd.NA,
                "notes": "25th percentile of matched Year-5 salary as cap percentage.",
            },
            {
                "metric_group": "salary_cap_pct_distribution",
                "metric": "matched_cap_pct_median",
                "value": float(matched_cap.median()) if not matched_cap.empty else pd.NA,
                "notes": "Median matched Year-5 salary as cap percentage.",
            },
            {
                "metric_group": "salary_cap_pct_distribution",
                "metric": "matched_cap_pct_p75",
                "value": float(matched_cap.quantile(0.75)) if not matched_cap.empty else pd.NA,
                "notes": "75th percentile of matched Year-5 salary as cap percentage.",
            },
            {
                "metric_group": "salary_cap_pct_distribution",
                "metric": "matched_cap_pct_max",
                "value": float(matched_cap.max()) if not matched_cap.empty else pd.NA,
                "notes": "Maximum matched Year-5 salary as cap percentage.",
            },
            {
                "metric_group": "salary_cap_pct_distribution",
                "metric": "matched_cap_pct_mean",
                "value": float(matched_cap.mean()) if not matched_cap.empty else pd.NA,
                "notes": "Mean matched Year-5 salary as cap percentage.",
            },
        ]
    )

    return pd.DataFrame(metrics)



def markdown_from_summary(summary_df: pd.DataFrame, path_map: dict[str, Path]) -> str:
    lines = [
        "# Deliverable 3 Salary Target Audit",
        "",
        "## Inputs",
        "",
        *[f"- `{role}`: `{path}`" for role, path in sorted(path_map.items())],
        "",
        "## Summary metrics",
        "",
        "| Metric group | Metric | Value | Notes |",
        "|---|---|---:|---|",
    ]
    for _, row in summary_df.iterrows():
        value = row["value"]
        if pd.isna(value):
            value_str = ""
        elif isinstance(value, float):
            value_str = f"{value:.6f}" if abs(value) < 10 else f"{value:.3f}"
        else:
            value_str = str(value)
        lines.append(
            f"| {row['metric_group']} | {row['metric']} | {value_str} | {row['notes']} |"
        )
    return "\n".join(lines) + "\n"



def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else project_root / "Output" / "Salary Decision Support"
    ensure_dir(output_dir)
    workflow_log_path = output_dir / args.log_name

    inventory_csv = args.inventory_csv.resolve() if args.inventory_csv else (output_dir / "project_inventory.csv")
    inventory_df = load_inventory_table(inventory_csv)
    path_map = resolve_role_paths(project_root, inventory_df)

    target_df = pd.read_csv(path_map["year5_salary_target"])
    merge_summary_df = pd.read_csv(path_map["year5_salary_merge_summary"])
    unmatched_df = pd.read_csv(path_map["year5_salary_unmatched_diagnostic"])
    profile_df = pd.read_csv(path_map["final_player_profile"])

    require_columns(
        target_df,
        [
            "PLAYER_ID",
            "PLAYER_NAME",
            "draft_year",
            "year5_salary_cap_pct",
            "year5_salary_match_flag",
            "salary_match_type",
            "salary_name_match_flag",
            "cap_match_flag",
            "raw_salary_rows",
            "raw_team_count",
            "any_team_option",
            "any_player_option",
            "any_qualifying_offer",
            "any_two_way_contract",
            "any_terminated",
        ],
        "year5_salary_target_table.csv",
    )
    require_columns(merge_summary_df, ["metric", "value"], "year5_salary_merge_summary.csv")
    require_columns(
        unmatched_df,
        ["PLAYER_ID", "PLAYER_NAME", "draft_year", "year5_season_start", "year5_season_string"],
        "year5_salary_unmatched_diagnostic.csv",
    )
    require_columns(
        profile_df,
        ["PLAYER_ID", "PLAYER_NAME", "macro_archetype", "identity_drift_class"],
        "final_player_archetype_profile_table.csv",
    )

    matched_mask = target_df["year5_salary_match_flag"].fillna(0).astype(int) == 1

    summary_df = build_metric_rows(target_df, merge_summary_df)

    by_draft_year_df = (
        target_df.assign(year5_matched=matched_mask.astype(int))
        .groupby("draft_year", as_index=False)
        .agg(
            total_players=("PLAYER_ID", "count"),
            matched_players=("year5_matched", "sum"),
        )
        .assign(
            unmatched_players=lambda d: d["total_players"] - d["matched_players"],
            match_rate=lambda d: d["matched_players"] / d["total_players"],
        )
        .sort_values("draft_year")
    )

    match_type_df = (
        target_df.groupby("salary_match_type", dropna=False, as_index=False)
        .agg(players=("PLAYER_ID", "count"))
        .assign(share=lambda d: d["players"] / d["players"].sum())
        .sort_values(["players", "salary_match_type"], ascending=[False, True])
    )

    contract_flag_cols = [
        "any_team_option",
        "any_player_option",
        "any_qualifying_offer",
        "any_two_way_contract",
        "any_terminated",
    ]
    contract_flag_df = pd.DataFrame(
        [
            {
                "flag": col,
                "matched_players_flagged": int(target_df.loc[matched_mask, col].fillna(0).astype(int).sum()),
                "matched_share_flagged": float(target_df.loc[matched_mask, col].fillna(0).astype(int).mean()),
                "all_players_flagged": int(target_df[col].fillna(0).astype(int).sum()),
                "all_players_share_flagged": float(target_df[col].fillna(0).astype(int).mean()),
            }
            for col in contract_flag_cols
        ]
    )

    cap_pct_summary_df = pd.DataFrame(
        {
            "distribution": ["matched_year5_salary_cap_pct"],
            "count": [int(target_df.loc[matched_mask, "year5_salary_cap_pct"].notna().sum())],
            "min": [pd.to_numeric(target_df.loc[matched_mask, "year5_salary_cap_pct"], errors="coerce").min()],
            "p10": [pd.to_numeric(target_df.loc[matched_mask, "year5_salary_cap_pct"], errors="coerce").quantile(0.10)],
            "p25": [pd.to_numeric(target_df.loc[matched_mask, "year5_salary_cap_pct"], errors="coerce").quantile(0.25)],
            "median": [pd.to_numeric(target_df.loc[matched_mask, "year5_salary_cap_pct"], errors="coerce").median()],
            "mean": [pd.to_numeric(target_df.loc[matched_mask, "year5_salary_cap_pct"], errors="coerce").mean()],
            "p75": [pd.to_numeric(target_df.loc[matched_mask, "year5_salary_cap_pct"], errors="coerce").quantile(0.75)],
            "p90": [pd.to_numeric(target_df.loc[matched_mask, "year5_salary_cap_pct"], errors="coerce").quantile(0.90)],
            "max": [pd.to_numeric(target_df.loc[matched_mask, "year5_salary_cap_pct"], errors="coerce").max()],
        }
    )

    profile_key_df = profile_df[["PLAYER_ID", "PLAYER_NAME", "macro_archetype", "identity_drift_class"]].drop_duplicates()
    target_plus_profile_df = target_df.merge(profile_key_df, on=["PLAYER_ID", "PLAYER_NAME"], how="left")

    by_macro_df = (
        target_plus_profile_df.assign(year5_matched=matched_mask.astype(int))
        .groupby("macro_archetype", dropna=False, as_index=False)
        .agg(total_players=("PLAYER_ID", "count"), matched_players=("year5_matched", "sum"))
        .assign(
            unmatched_players=lambda d: d["total_players"] - d["matched_players"],
            match_rate=lambda d: d["matched_players"] / d["total_players"],
        )
        .sort_values(["match_rate", "total_players", "macro_archetype"], ascending=[False, False, True])
    )

    by_drift_df = (
        target_plus_profile_df.assign(year5_matched=matched_mask.astype(int))
        .groupby("identity_drift_class", dropna=False, as_index=False)
        .agg(total_players=("PLAYER_ID", "count"), matched_players=("year5_matched", "sum"))
        .assign(
            unmatched_players=lambda d: d["total_players"] - d["matched_players"],
            match_rate=lambda d: d["matched_players"] / d["total_players"],
        )
        .sort_values(["match_rate", "total_players", "identity_drift_class"], ascending=[False, False, True])
    )

    unmatched_with_profile_df = unmatched_df.merge(profile_key_df, on=["PLAYER_ID", "PLAYER_NAME"], how="left")
    unmatched_with_profile_df["audit_note"] = "No matched Year-5 salary row after current normalization and season merge."
    unmatched_with_profile_df = unmatched_with_profile_df.sort_values(["draft_year", "PLAYER_NAME"])

    demo_normalized = {normalize_demo_name(name) for name in DEMO_PLAYERS}
    demo_mask = target_df["PLAYER_NAME"].astype(str).map(normalize_demo_name).isin(demo_normalized)
    demo_holdout_df = (
        target_plus_profile_df.loc[demo_mask, [
            "PLAYER_ID",
            "PLAYER_NAME",
            "draft_year",
            "macro_archetype",
            "identity_drift_class",
            "year5_salary_match_flag",
            "salary_match_type",
            "year5_salary_cap_pct",
        ]]
        .drop_duplicates()
        .assign(
            holdout_for_case_study=1,
            holdout_reason="Reserved salary suggestion demo player; exclude from future salary-model training/calibration.",
        )
        .sort_values(["draft_year", "PLAYER_NAME"])
    )

    output_files = {
        "salary_target_audit_summary_csv": output_dir / "salary_target_audit_summary.csv",
        "salary_target_audit_summary_md": output_dir / "salary_target_audit_summary.md",
        "salary_target_match_by_draft_year_csv": output_dir / "salary_target_match_by_draft_year.csv",
        "salary_target_match_type_breakdown_csv": output_dir / "salary_target_match_type_breakdown.csv",
        "salary_target_contract_flag_summary_csv": output_dir / "salary_target_contract_flag_summary.csv",
        "salary_target_cap_pct_distribution_summary_csv": output_dir / "salary_target_cap_pct_distribution_summary.csv",
        "salary_target_match_by_macro_archetype_csv": output_dir / "salary_target_match_by_macro_archetype.csv",
        "salary_target_match_by_identity_drift_csv": output_dir / "salary_target_match_by_identity_drift.csv",
        "salary_target_unmatched_with_profile_csv": output_dir / "salary_target_unmatched_with_profile.csv",
        "salary_demo_player_holdout_manifest_csv": output_dir / "salary_demo_player_holdout_manifest.csv",
        "salary_target_audit_metadata_json": output_dir / "salary_target_audit_metadata.json",
    }

    summary_df.to_csv(output_files["salary_target_audit_summary_csv"], index=False)
    output_files["salary_target_audit_summary_md"].write_text(markdown_from_summary(summary_df, path_map), encoding="utf-8")
    by_draft_year_df.to_csv(output_files["salary_target_match_by_draft_year_csv"], index=False)
    match_type_df.to_csv(output_files["salary_target_match_type_breakdown_csv"], index=False)
    contract_flag_df.to_csv(output_files["salary_target_contract_flag_summary_csv"], index=False)
    cap_pct_summary_df.to_csv(output_files["salary_target_cap_pct_distribution_summary_csv"], index=False)
    by_macro_df.to_csv(output_files["salary_target_match_by_macro_archetype_csv"], index=False)
    by_drift_df.to_csv(output_files["salary_target_match_by_identity_drift_csv"], index=False)
    unmatched_with_profile_df.to_csv(output_files["salary_target_unmatched_with_profile_csv"], index=False)
    demo_holdout_df.to_csv(output_files["salary_demo_player_holdout_manifest_csv"], index=False)

    write_json(
        output_files["salary_target_audit_metadata_json"],
        {
            "script": SCRIPT_NAME,
            "project_root": str(project_root),
            "inventory_csv_used": str(inventory_csv) if inventory_df is not None else None,
            "resolved_inputs": {role: str(path) for role, path in path_map.items()},
            "demo_players_reserved_for_case_study": DEMO_PLAYERS,
            "demo_holdout_count": int(len(demo_holdout_df)),
            "matched_player_count": int(matched_mask.sum()),
            "unmatched_player_count": int((~matched_mask).sum()),
            "year5_salary_match_rate": float(matched_mask.mean()),
            "outputs": {k: str(v) for k, v in output_files.items()},
        },
    )

    log_text = f"""
Phase 1 — salary target audit
Script: {SCRIPT_NAME}
Project root: {project_root}
Inventory reference: {inventory_csv if inventory_df is not None else 'not used'}

Purpose:
- treat year5_salary_target_table.csv as the canonical Deliverable 3 target table,
- verify merge coverage and recompute the core match-rate summary from the target rows,
- profile match coverage by draft year, macro archetype, and identity drift class,
- summarize contract-option flags and cap-percentage distribution for matched rows,
- carry Trae Young and Nikola Vučević forward as reserved demo players for later salary-suggestion case studies.

Resolved inputs:
- year5_salary_target: {path_map['year5_salary_target']}
- year5_salary_merge_summary: {path_map['year5_salary_merge_summary']}
- year5_salary_unmatched_diagnostic: {path_map['year5_salary_unmatched_diagnostic']}
- final_player_profile: {path_map['final_player_profile']}

Key recomputed counts:
- matched players: {int(matched_mask.sum())}
- unmatched players: {int((~matched_mask).sum())}
- match rate: {float(matched_mask.mean()):.6f}
- demo holdout players found: {int(len(demo_holdout_df))}

Outputs written:
- {output_files['salary_target_audit_summary_csv'].name}
- {output_files['salary_target_audit_summary_md'].name}
- {output_files['salary_target_match_by_draft_year_csv'].name}
- {output_files['salary_target_match_type_breakdown_csv'].name}
- {output_files['salary_target_contract_flag_summary_csv'].name}
- {output_files['salary_target_cap_pct_distribution_summary_csv'].name}
- {output_files['salary_target_match_by_macro_archetype_csv'].name}
- {output_files['salary_target_match_by_identity_drift_csv'].name}
- {output_files['salary_target_unmatched_with_profile_csv'].name}
- {output_files['salary_demo_player_holdout_manifest_csv'].name}
- {output_files['salary_target_audit_metadata_json'].name}
""".strip()
    append_workflow_log(workflow_log_path, log_text)

    print("Salary target audit complete.")
    print(f"Matched players: {int(matched_mask.sum())}")
    print(f"Unmatched players: {int((~matched_mask).sum())}")
    print(f"Match rate: {float(matched_mask.mean()):.6f}")
    print(f"Outputs written under: {output_dir}")


if __name__ == "__main__":
    main()
