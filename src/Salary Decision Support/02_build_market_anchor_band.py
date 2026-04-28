from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
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

SCRIPT_NAME = "02_build_market_anchor_band.py"
REQUIRED_ROLES = {
    "deliverable3_block2_context",
    "comp_salary_detail",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Deliverable 3 market-anchor band from Block 2 comp-market context "
            "and row-level comp salary detail. The resulting band is intentionally descriptive "
            "and audit-friendly: protected price, fair price, and walk-away max."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root containing Output/, src/, and visual/ folders.",
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
            "Optional project inventory CSV from Phase 0. If provided, the script will use it "
            "as a reference for resolved paths when those paths exist locally."
        ),
    )
    parser.add_argument(
        "--holdout-csv",
        type=Path,
        default=None,
        help=(
            "Optional explicit path to salary_demo_player_holdout_manifest.csv from Phase 1. "
            "If omitted, the script will look under Output/Salary Decision Support."
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
            "Unable to resolve all required Deliverable 3 market-anchor inputs. Missing roles:\n"
            f"{hints}"
        )

    return role_to_path


def require_columns(df: pd.DataFrame, required: list[str], table_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def normalize_demo_name(name: str) -> str:
    return (
        name.lower()
        .replace("č", "c")
        .replace("ć", "c")
        .replace("š", "s")
        .replace("ž", "z")
        .replace("đ", "d")
    )


def weighted_quantile(values: pd.Series, weights: pd.Series, q: float) -> float | pd.NA:
    df = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "weight": pd.to_numeric(weights, errors="coerce")})
    df = df.dropna()
    df = df[df["weight"] > 0].sort_values("value")
    if df.empty:
        return pd.NA
    cumulative = df["weight"].cumsum()
    cutoff = q * df["weight"].sum()
    idx = cumulative.searchsorted(cutoff, side="left")
    idx = min(int(idx), len(df) - 1)
    return float(df.iloc[idx]["value"])


def weighted_std(values: pd.Series, weights: pd.Series) -> float | pd.NA:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = v.notna() & w.notna() & (w > 0)
    v = v[mask]
    w = w[mask]
    if len(v) == 0:
        return pd.NA
    if len(v) == 1:
        return 0.0
    mean = np.average(v, weights=w)
    var = np.average((v - mean) ** 2, weights=w)
    return float(np.sqrt(var))


def fallback_protected_price(row: pd.Series) -> float | pd.NA:
    for col in ["comp_salary_anchor_p25", "detail_anchor_p25", "comp_salary_anchor_p50", "fair_price_cap_pct"]:
        value = row.get(col, pd.NA)
        if pd.notna(value):
            return float(value)
    return pd.NA


def fallback_walkaway_price(row: pd.Series) -> float | pd.NA:
    for col in ["comp_salary_anchor_p75", "detail_anchor_p75", "comp_salary_anchor_p50", "fair_price_cap_pct"]:
        value = row.get(col, pd.NA)
        if pd.notna(value):
            return float(value)
    return pd.NA


def classify_band_source(row: pd.Series) -> str:
    if pd.notna(row.get("comp_salary_anchor_p25")) and pd.notna(row.get("comp_salary_anchor_weighted_mean")) and pd.notna(row.get("comp_salary_anchor_p75")):
        return "block2_direct"
    if pd.notna(row.get("detail_anchor_p25")) and pd.notna(row.get("detail_anchor_weighted_mean")) and pd.notna(row.get("detail_anchor_p75")):
        return "recomputed_from_comp_detail"
    return "mixed_fallback"


def fair_price_source(row: pd.Series) -> str:
    if pd.notna(row.get("comp_salary_anchor_weighted_mean")):
        return "block2_weighted_mean"
    if pd.notna(row.get("detail_anchor_weighted_mean")):
        return "detail_weighted_mean"
    if pd.notna(row.get("comp_salary_anchor_p50")):
        return "block2_p50_fallback"
    if pd.notna(row.get("detail_anchor_p50")):
        return "detail_p50_fallback"
    return "missing"


def band_text(row: pd.Series) -> str:
    if pd.isna(row["protected_price_cap_pct"]) or pd.isna(row["walk_away_max_cap_pct"]):
        return "Comp-based market anchor not supported."
    return (
        f"Protected around {row['protected_price_cap_pct']:.3f} of cap, "
        f"fair around {row['fair_price_cap_pct']:.3f}, "
        f"walk-away max around {row['walk_away_max_cap_pct']:.3f}."
    )


def make_markdown_summary(summary_df: pd.DataFrame, input_map: dict[str, Path]) -> str:
    lines = [
        "# Deliverable 3 Market Anchor Band",
        "",
        "## Inputs",
        "",
        *[f"- `{role}`: `{path}`" for role, path in sorted(input_map.items())],
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

    holdout_csv = None
    if args.holdout_csv is not None and args.holdout_csv.exists():
        holdout_csv = args.holdout_csv.resolve()
    else:
        candidate = output_dir / "salary_demo_player_holdout_manifest.csv"
        if candidate.exists():
            holdout_csv = candidate.resolve()

    block2_df = pd.read_csv(path_map["deliverable3_block2_context"])
    comp_detail_df = pd.read_csv(path_map["comp_salary_detail"])

    require_columns(
        block2_df,
        [
            "PLAYER_ID",
            "PLAYER_NAME",
            "draft_year",
            "macro_archetype",
            "shot_style_subtype",
            "hybrid_archetype_label",
            "identity_drift_class",
            "prototype_fit_ambiguity",
            "comp_salary_match_count",
            "comp_salary_match_rate",
            "comp_salary_anchor_weighted_mean",
            "comp_salary_anchor_p25",
            "comp_salary_anchor_p50",
            "comp_salary_anchor_p75",
            "comp_salary_anchor_support",
            "historical_year5_salary_cap_pct_observed",
            "historical_year5_salary_observed_flag",
        ],
        "deliverable3_block2_archetype_comp_market_context.csv",
    )
    require_columns(
        comp_detail_df,
        [
            "PLAYER_ID",
            "PLAYER_NAME",
            "comp_rank",
            "same_macro_archetype",
            "similarity_score",
            "similarity_weight",
            "comp_year5_salary_cap_pct",
            "comp_year5_salary_match_flag",
            "comp_name_alignment_flag",
        ],
        "comp_salary_detail_table.csv",
    )

    comp_detail_df["comp_year5_salary_match_flag"] = comp_detail_df["comp_year5_salary_match_flag"].fillna(0).astype(int)
    matched_detail_df = comp_detail_df.loc[comp_detail_df["comp_year5_salary_match_flag"] == 1].copy()
    matched_detail_df["similarity_weight"] = pd.to_numeric(matched_detail_df["similarity_weight"], errors="coerce")
    matched_detail_df["comp_year5_salary_cap_pct"] = pd.to_numeric(matched_detail_df["comp_year5_salary_cap_pct"], errors="coerce")
    matched_detail_df["same_macro_archetype"] = pd.to_numeric(matched_detail_df["same_macro_archetype"], errors="coerce")
    matched_detail_df["comp_name_alignment_flag"] = pd.to_numeric(matched_detail_df["comp_name_alignment_flag"], errors="coerce")

    detail_summary_df = (
        matched_detail_df.groupby(["PLAYER_ID", "PLAYER_NAME"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "detail_matched_comp_count": int(len(g)),
                    "detail_anchor_weighted_mean": np.average(
                        g["comp_year5_salary_cap_pct"], weights=g["similarity_weight"]
                    ) if g["similarity_weight"].fillna(0).sum() > 0 else pd.NA,
                    "detail_anchor_p25": weighted_quantile(g["comp_year5_salary_cap_pct"], g["similarity_weight"], 0.25),
                    "detail_anchor_p50": weighted_quantile(g["comp_year5_salary_cap_pct"], g["similarity_weight"], 0.50),
                    "detail_anchor_p75": weighted_quantile(g["comp_year5_salary_cap_pct"], g["similarity_weight"], 0.75),
                    "detail_anchor_std_weighted": weighted_std(g["comp_year5_salary_cap_pct"], g["similarity_weight"]),
                    "detail_top_comp_weight": float(pd.to_numeric(g["similarity_weight"], errors="coerce").max()),
                    "detail_effective_comp_n": float(
                        1.0 / np.square(pd.to_numeric(g["similarity_weight"], errors="coerce")).sum()
                    ) if np.square(pd.to_numeric(g["similarity_weight"], errors="coerce")).sum() > 0 else pd.NA,
                    "detail_same_macro_share": float(pd.to_numeric(g["same_macro_archetype"], errors="coerce").mean()),
                    "detail_name_alignment_share": float(pd.to_numeric(g["comp_name_alignment_flag"], errors="coerce").mean()),
                }
            )
        , include_groups=False)
        .reset_index(drop=True)
    )

    anchor_df = block2_df.merge(detail_summary_df, on=["PLAYER_ID", "PLAYER_NAME"], how="left")

    anchor_df["fair_price_cap_pct"] = pd.to_numeric(anchor_df["comp_salary_anchor_weighted_mean"], errors="coerce")
    missing_fair_mask = anchor_df["fair_price_cap_pct"].isna()
    anchor_df.loc[missing_fair_mask, "fair_price_cap_pct"] = pd.to_numeric(
        anchor_df.loc[missing_fair_mask, "detail_anchor_weighted_mean"], errors="coerce"
    )
    missing_fair_mask = anchor_df["fair_price_cap_pct"].isna()
    anchor_df.loc[missing_fair_mask, "fair_price_cap_pct"] = pd.to_numeric(
        anchor_df.loc[missing_fair_mask, "comp_salary_anchor_p50"], errors="coerce"
    )
    missing_fair_mask = anchor_df["fair_price_cap_pct"].isna()
    anchor_df.loc[missing_fair_mask, "fair_price_cap_pct"] = pd.to_numeric(
        anchor_df.loc[missing_fair_mask, "detail_anchor_p50"], errors="coerce"
    )

    anchor_df["protected_price_cap_pct"] = anchor_df.apply(fallback_protected_price, axis=1)
    anchor_df["walk_away_max_cap_pct"] = anchor_df.apply(fallback_walkaway_price, axis=1)

    anchor_df["protected_price_cap_pct"] = pd.to_numeric(anchor_df["protected_price_cap_pct"], errors="coerce")
    anchor_df["fair_price_cap_pct"] = pd.to_numeric(anchor_df["fair_price_cap_pct"], errors="coerce")
    anchor_df["walk_away_max_cap_pct"] = pd.to_numeric(anchor_df["walk_away_max_cap_pct"], errors="coerce")

    anchor_df["protected_price_cap_pct"] = np.where(
        pd.notna(anchor_df["protected_price_cap_pct"]) & pd.notna(anchor_df["fair_price_cap_pct"]),
        np.minimum(anchor_df["protected_price_cap_pct"], anchor_df["fair_price_cap_pct"]),
        anchor_df["protected_price_cap_pct"],
    )
    anchor_df["walk_away_max_cap_pct"] = np.where(
        pd.notna(anchor_df["walk_away_max_cap_pct"]) & pd.notna(anchor_df["fair_price_cap_pct"]),
        np.maximum(anchor_df["walk_away_max_cap_pct"], anchor_df["fair_price_cap_pct"]),
        anchor_df["walk_away_max_cap_pct"],
    )

    anchor_df["anchor_band_width"] = anchor_df["walk_away_max_cap_pct"] - anchor_df["protected_price_cap_pct"]
    anchor_df["anchor_band_relative_width"] = anchor_df["anchor_band_width"] / anchor_df["fair_price_cap_pct"]
    anchor_df["fair_price_source"] = anchor_df.apply(fair_price_source, axis=1)
    anchor_df["anchor_band_source"] = anchor_df.apply(classify_band_source, axis=1)
    anchor_df["market_anchor_band_text"] = anchor_df.apply(band_text, axis=1)

    anchor_df["historical_year5_salary_cap_pct_observed"] = pd.to_numeric(
        anchor_df["historical_year5_salary_cap_pct_observed"], errors="coerce"
    )
    anchor_df["historical_vs_fair_gap"] = (
        anchor_df["historical_year5_salary_cap_pct_observed"] - anchor_df["fair_price_cap_pct"]
    )
    anchor_df["historical_vs_protected_gap"] = (
        anchor_df["historical_year5_salary_cap_pct_observed"] - anchor_df["protected_price_cap_pct"]
    )
    anchor_df["historical_vs_walkaway_gap"] = (
        anchor_df["historical_year5_salary_cap_pct_observed"] - anchor_df["walk_away_max_cap_pct"]
    )

    anchor_df["holdout_for_case_study"] = 0
    anchor_df["holdout_reason"] = pd.NA
    if holdout_csv is not None:
        holdout_df = pd.read_csv(holdout_csv)
        require_columns(
            holdout_df,
            ["PLAYER_ID", "PLAYER_NAME", "holdout_for_case_study", "holdout_reason"],
            "salary_demo_player_holdout_manifest.csv",
        )
        holdout_key_df = holdout_df[["PLAYER_ID", "PLAYER_NAME", "holdout_for_case_study", "holdout_reason"]].drop_duplicates()
        anchor_df = anchor_df.merge(
            holdout_key_df,
            on=["PLAYER_ID", "PLAYER_NAME"],
            how="left",
            suffixes=("", "_phase1"),
        )
        anchor_df["holdout_for_case_study"] = (
            pd.to_numeric(anchor_df["holdout_for_case_study_phase1"], errors="coerce")
            .fillna(anchor_df["holdout_for_case_study"])
            .fillna(0)
            .astype(int)
        )
        anchor_df["holdout_reason"] = anchor_df["holdout_reason_phase1"].combine_first(anchor_df["holdout_reason"])
        anchor_df = anchor_df.drop(columns=["holdout_for_case_study_phase1", "holdout_reason_phase1"])
    else:
        demo_normalized = {normalize_demo_name(name) for name in DEMO_PLAYERS}
        demo_mask = anchor_df["PLAYER_NAME"].astype(str).map(normalize_demo_name).isin(demo_normalized)
        anchor_df.loc[demo_mask, "holdout_for_case_study"] = 1
        anchor_df.loc[demo_mask, "holdout_reason"] = (
            "Reserved salary suggestion demo player; exclude from future salary-model training/calibration."
        )

    anchor_df["market_anchor_supported_flag"] = (
        anchor_df["protected_price_cap_pct"].notna()
        & anchor_df["fair_price_cap_pct"].notna()
        & anchor_df["walk_away_max_cap_pct"].notna()
    ).astype(int)

    core_cols = [
        "PLAYER_ID",
        "PLAYER_NAME",
        "draft_year",
        "macro_archetype",
        "shot_style_subtype",
        "hybrid_archetype_label",
        "identity_drift_class",
        "prototype_fit_ambiguity",
        "comp_salary_match_count",
        "comp_salary_match_rate",
        "comp_salary_anchor_support",
        "detail_matched_comp_count",
        "detail_effective_comp_n",
        "detail_top_comp_weight",
        "detail_same_macro_share",
        "detail_name_alignment_share",
        "detail_anchor_std_weighted",
        "protected_price_cap_pct",
        "fair_price_cap_pct",
        "walk_away_max_cap_pct",
        "anchor_band_width",
        "anchor_band_relative_width",
        "fair_price_source",
        "anchor_band_source",
        "market_anchor_supported_flag",
        "market_anchor_band_text",
        "historical_year5_salary_cap_pct_observed",
        "historical_year5_salary_observed_flag",
        "historical_vs_fair_gap",
        "historical_vs_protected_gap",
        "historical_vs_walkaway_gap",
        "holdout_for_case_study",
        "holdout_reason",
    ]
    anchor_table_df = anchor_df[core_cols].copy().sort_values(["draft_year", "PLAYER_NAME"])

    summary_rows: list[dict[str, Any]] = [
        {
            "metric_group": "anchor_table",
            "metric": "players_in_block2_context",
            "value": int(len(anchor_df)),
            "notes": "Row count from deliverable3_block2_archetype_comp_market_context.csv.",
        },
        {
            "metric_group": "anchor_table",
            "metric": "players_with_supported_market_anchor_band",
            "value": int(anchor_df["market_anchor_supported_flag"].sum()),
            "notes": "Players with non-missing protected, fair, and walk-away prices.",
        },
        {
            "metric_group": "anchor_table",
            "metric": "supported_market_anchor_share",
            "value": float(anchor_df["market_anchor_supported_flag"].mean()),
            "notes": "Share of Block 2 players with a usable market anchor band.",
        },
        {
            "metric_group": "anchor_table",
            "metric": "holdout_demo_players_flagged",
            "value": int(anchor_df["holdout_for_case_study"].sum()),
            "notes": "Reserved demo players carried forward for later salary-suggestion case studies.",
        },
    ]

    supported_df = anchor_df.loc[anchor_df["market_anchor_supported_flag"] == 1].copy()
    if not supported_df.empty:
        for col, metric_stub, note_stub in [
            ("protected_price_cap_pct", "protected_price", "Protected price from the comp-based market anchor."),
            ("fair_price_cap_pct", "fair_price", "Fair price from the comp-based market anchor."),
            ("walk_away_max_cap_pct", "walk_away_price", "Walk-away max from the comp-based market anchor."),
            ("anchor_band_width", "anchor_band_width", "Absolute width of the market anchor band."),
            ("anchor_band_relative_width", "anchor_band_relative_width", "Band width divided by fair price."),
        ]:
            series = pd.to_numeric(supported_df[col], errors="coerce")
            summary_rows.extend(
                [
                    {
                        "metric_group": "supported_distribution",
                        "metric": f"{metric_stub}_p25",
                        "value": float(series.quantile(0.25)),
                        "notes": note_stub,
                    },
                    {
                        "metric_group": "supported_distribution",
                        "metric": f"{metric_stub}_median",
                        "value": float(series.median()),
                        "notes": note_stub,
                    },
                    {
                        "metric_group": "supported_distribution",
                        "metric": f"{metric_stub}_p75",
                        "value": float(series.quantile(0.75)),
                        "notes": note_stub,
                    },
                ]
            )

    summary_df = pd.DataFrame(summary_rows)

    support_breakdown_df = (
        anchor_df.groupby("comp_salary_anchor_support", dropna=False, as_index=False)
        .agg(
            players=("PLAYER_ID", "count"),
            supported_market_anchor_players=("market_anchor_supported_flag", "sum"),
            fair_price_mean=("fair_price_cap_pct", "mean"),
            fair_price_median=("fair_price_cap_pct", "median"),
            anchor_band_width_median=("anchor_band_width", "median"),
            effective_comp_n_median=("detail_effective_comp_n", "median"),
        )
        .assign(
            supported_market_anchor_share=lambda d: d["supported_market_anchor_players"] / d["players"],
        )
        .sort_values(["players", "comp_salary_anchor_support"], ascending=[False, True])
    )

    fair_distribution_df = pd.DataFrame(
        {
            "distribution": ["fair_price_cap_pct_supported_only"],
            "count": [int(supported_df["fair_price_cap_pct"].notna().sum())],
            "min": [pd.to_numeric(supported_df["fair_price_cap_pct"], errors="coerce").min()],
            "p10": [pd.to_numeric(supported_df["fair_price_cap_pct"], errors="coerce").quantile(0.10)],
            "p25": [pd.to_numeric(supported_df["fair_price_cap_pct"], errors="coerce").quantile(0.25)],
            "median": [pd.to_numeric(supported_df["fair_price_cap_pct"], errors="coerce").median()],
            "mean": [pd.to_numeric(supported_df["fair_price_cap_pct"], errors="coerce").mean()],
            "p75": [pd.to_numeric(supported_df["fair_price_cap_pct"], errors="coerce").quantile(0.75)],
            "p90": [pd.to_numeric(supported_df["fair_price_cap_pct"], errors="coerce").quantile(0.90)],
            "max": [pd.to_numeric(supported_df["fair_price_cap_pct"], errors="coerce").max()],
        }
    )

    output_files = {
        "salary_market_anchor_band_table_csv": output_dir / "salary_market_anchor_band_table.csv",
        "salary_market_anchor_band_summary_csv": output_dir / "salary_market_anchor_band_summary.csv",
        "salary_market_anchor_band_summary_md": output_dir / "salary_market_anchor_band_summary.md",
        "salary_market_anchor_support_breakdown_csv": output_dir / "salary_market_anchor_support_breakdown.csv",
        "salary_market_anchor_fair_price_distribution_csv": output_dir / "salary_market_anchor_fair_price_distribution.csv",
        "salary_market_anchor_band_metadata_json": output_dir / "salary_market_anchor_band_metadata.json",
    }

    anchor_table_df.to_csv(output_files["salary_market_anchor_band_table_csv"], index=False)
    summary_df.to_csv(output_files["salary_market_anchor_band_summary_csv"], index=False)
    output_files["salary_market_anchor_band_summary_md"].write_text(
        make_markdown_summary(summary_df, {
            "deliverable3_block2_context": path_map["deliverable3_block2_context"],
            "comp_salary_detail": path_map["comp_salary_detail"],
            **({"salary_demo_player_holdout_manifest": holdout_csv} if holdout_csv is not None else {}),
        }),
        encoding="utf-8",
    )
    support_breakdown_df.to_csv(output_files["salary_market_anchor_support_breakdown_csv"], index=False)
    fair_distribution_df.to_csv(output_files["salary_market_anchor_fair_price_distribution_csv"], index=False)

    write_json(
        output_files["salary_market_anchor_band_metadata_json"],
        {
            "script": SCRIPT_NAME,
            "project_root": str(project_root),
            "inventory_csv_used": str(inventory_csv) if inventory_df is not None else None,
            "holdout_csv_used": str(holdout_csv) if holdout_csv is not None else None,
            "resolved_inputs": {role: str(path) for role, path in path_map.items()},
            "demo_players_reserved_for_case_study": DEMO_PLAYERS,
            "players_in_block2_context": int(len(anchor_df)),
            "supported_market_anchor_players": int(anchor_df["market_anchor_supported_flag"].sum()),
            "supported_market_anchor_share": float(anchor_df["market_anchor_supported_flag"].mean()),
            "outputs": {k: str(v) for k, v in output_files.items()},
        },
    )

    log_text = f"""
Phase 2 — build market anchor band
Script: {SCRIPT_NAME}
Project root: {project_root}
Inventory reference: {inventory_csv if inventory_df is not None else 'not used'}
Holdout manifest: {holdout_csv if holdout_csv is not None else 'not used'}

Purpose:
- convert the Block 2 comp-market context into a reusable price board with protected price, fair price, and walk-away max,
- audit the anchor band against the row-level comp salary detail table rather than trusting only one preassembled table,
- preserve Trae Young and Nikola Vučević as reserved case-study players for later salary suggestions,
- produce a clean one-row-per-player anchor table for downstream rule logic and decision-card assembly.

Resolved inputs:
- deliverable3_block2_context: {path_map['deliverable3_block2_context']}
- comp_salary_detail: {path_map['comp_salary_detail']}
- salary_demo_player_holdout_manifest: {holdout_csv if holdout_csv is not None else 'not provided'}

Key outputs:
- players in Block 2 context: {int(len(anchor_df))}
- players with supported market anchor band: {int(anchor_df['market_anchor_supported_flag'].sum())}
- supported market anchor share: {float(anchor_df['market_anchor_supported_flag'].mean()):.6f}

Design choices:
- protected price = comp p25 with recomputed-detail fallback,
- fair price = weighted mean with p50 fallback,
- walk-away max = comp p75 with recomputed-detail fallback,
- ceiling comps remain excluded from the pricing anchor by design,
- no forecast premium/discount is applied in this phase.

Outputs written:
- {output_files['salary_market_anchor_band_table_csv'].name}
- {output_files['salary_market_anchor_band_summary_csv'].name}
- {output_files['salary_market_anchor_band_summary_md'].name}
- {output_files['salary_market_anchor_support_breakdown_csv'].name}
- {output_files['salary_market_anchor_fair_price_distribution_csv'].name}
- {output_files['salary_market_anchor_band_metadata_json'].name}
""".strip()
    append_workflow_log(workflow_log_path, log_text)

    print("Market anchor band build complete.")
    print(f"Players in Block 2 context: {int(len(anchor_df))}")
    print(f"Players with supported market anchor band: {int(anchor_df['market_anchor_supported_flag'].sum())}")
    print(f"Supported market anchor share: {float(anchor_df['market_anchor_supported_flag'].mean()):.6f}")
    print(f"Outputs written under: {output_dir}")


if __name__ == "__main__":
    main()
