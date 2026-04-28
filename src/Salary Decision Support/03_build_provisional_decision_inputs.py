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
    append_workflow_log,
    ensure_dir,
    get_expected_files,
    resolve_all_expected_files,
    write_json,
)

SCRIPT_NAME = "03_build_provisional_decision_inputs.py"
REQUIRED_ROLES = {
    "deliverable3_block2_context",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build provisional Deliverable 3 decision inputs from the Phase 2 market-anchor band "
            "and current Block 2 comp/risk proxies. This phase remains intentionally conservative: "
            "it prepares the decision-card structure, comp-support tiers, scarcity wording, and "
            "provisional action buckets without claiming final extension guidance before Deliverable 1."
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
            "Optional Phase 0 inventory CSV. If provided, resolved paths from that file will be "
            "used when they exist locally."
        ),
    )
    parser.add_argument(
        "--market-anchor-csv",
        type=Path,
        default=None,
        help=(
            "Optional explicit path to salary_market_anchor_band_table.csv from Phase 2. "
            "If omitted, the script will look under Output/Salary Decision Support first and "
            "then search recursively."
        ),
    )
    parser.add_argument(
        "--holdout-csv",
        type=Path,
        default=None,
        help=(
            "Optional explicit path to salary_demo_player_holdout_manifest.csv from Phase 1. "
            "If omitted, the script will look under Output/Salary Decision Support and then the "
            "project root."
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
            "Unable to resolve all required Deliverable 3 provisional-input files. Missing roles:\n"
            f"{hints}"
        )
    return role_to_path


def resolve_optional_input(project_root: Path, explicit_path: Path | None, default_candidates: list[Path], basename: str) -> Path | None:
    if explicit_path is not None and explicit_path.exists():
        return explicit_path.resolve()
    for candidate in default_candidates:
        if candidate.exists():
            return candidate.resolve()
    matches = list(project_root.rglob(basename))
    if matches:
        matches.sort(key=lambda p: (len(p.parts), str(p).lower()))
        return matches[0].resolve()
    return None


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


def make_scarcity_reference() -> pd.DataFrame:
    rows = [
        {
            "macro_archetype": "High-Usage Primary Creators",
            "scarcity_tier": "scarce",
            "scarcity_wording": "Scarce offensive engine archetype; earlier disciplined offers can be justified when comp support is credible.",
        },
        {
            "macro_archetype": "Scoring Bigs / Two-Way Forwards",
            "scarcity_tier": "selective_premium",
            "scarcity_wording": "Valuable scoring frontcourt archetype, but price discipline still matters because replacement is not impossible.",
        },
        {
            "macro_archetype": "Perimeter Wings & Connectors",
            "scarcity_tier": "replaceable_middle",
            "scarcity_wording": "Useful middle-class wing archetype; offers should stay disciplined unless other evidence is unusually strong.",
        },
        {
            "macro_archetype": "Low-Usage Interior Bigs",
            "scarcity_tier": "replaceable",
            "scarcity_wording": "Functional but more replaceable interior archetype; market evidence should be treated conservatively.",
        },
        {
            "macro_archetype": "Fringe / Low-Opportunity Players",
            "scarcity_tier": "highly_replaceable",
            "scarcity_wording": "Highly replaceable low-opportunity archetype; avoid paying ahead of weak evidence.",
        },
    ]
    return pd.DataFrame(rows)


def ambiguity_band(series: pd.Series, q25: float, q75: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return np.select(
        [
            values.isna(),
            values <= q25,
            values <= q75,
        ],
        [
            "unknown",
            "low_ambiguity",
            "medium_ambiguity",
        ],
        default="high_ambiguity",
    )


def width_band(series: pd.Series, q25: float, q50: float, q75: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return np.select(
        [
            values.isna(),
            values <= q25,
            values <= q50,
            values <= q75,
        ],
        [
            "unknown",
            "narrow",
            "balanced",
            "wide",
        ],
        default="very_wide",
    )


def comp_support_bucket(row: pd.Series, thresholds: dict[str, float]) -> str:
    if not bool(row.get("market_anchor_supported_flag", False)):
        return "insufficient"

    support_raw = str(row.get("comp_salary_anchor_support", "")).strip().lower()
    match_count = pd.to_numeric(row.get("comp_salary_match_count"), errors="coerce")
    match_rate = pd.to_numeric(row.get("comp_salary_match_rate"), errors="coerce")
    eff_n = pd.to_numeric(row.get("detail_effective_comp_n"), errors="coerce")
    same_macro = pd.to_numeric(row.get("detail_same_macro_share"), errors="coerce")
    rel_width = pd.to_numeric(row.get("anchor_band_relative_width"), errors="coerce")

    score = 0
    if support_raw == "high":
        score += 2
    elif support_raw == "medium":
        score += 1
    elif support_raw == "low":
        score += 0
    elif support_raw == "very_low":
        score -= 1
    elif support_raw == "insufficient":
        score -= 2

    if pd.notna(match_count):
        if match_count >= 4:
            score += 1
        elif match_count < 2:
            score -= 2
        elif match_count < 3:
            score -= 1

    if pd.notna(match_rate):
        if match_rate >= 0.80:
            score += 1
        elif match_rate < 0.40:
            score -= 2
        elif match_rate < 0.60:
            score -= 1

    if pd.notna(eff_n):
        if eff_n >= thresholds["eff_n_q50"]:
            score += 1
        elif eff_n < thresholds["eff_n_q25"]:
            score -= 1

    if pd.notna(same_macro):
        if same_macro >= thresholds["same_macro_q50"]:
            score += 1
        elif same_macro < thresholds["same_macro_q25"]:
            score -= 1

    if pd.notna(rel_width):
        if rel_width <= thresholds["width_q50"]:
            score += 1
        elif rel_width > thresholds["width_q75"]:
            score -= 1

    if support_raw == "insufficient":
        return "insufficient"
    if score >= 4 and pd.notna(match_count) and match_count >= 4 and pd.notna(match_rate) and match_rate >= 0.60:
        return "strong"
    if score >= 1:
        return "moderate"
    return "weak"


def drift_signal(identity_drift_class: Any) -> str:
    value = "" if pd.isna(identity_drift_class) else str(identity_drift_class)
    if value == "role_shifting_materially":
        return "elevated_drift_risk"
    if value == "evolving_gradually":
        return "manageable_drift"
    if value == "stable":
        return "stable_profile"
    return "unresolved_drift"


def provisional_action_bucket(row: pd.Series) -> str:
    support = row["comp_support_bucket"]
    scarcity = row["scarcity_tier"]
    ambiguity = row["ambiguity_band"]
    width = row["anchor_width_band"]
    drift = row["drift_signal"]

    if support == "insufficient":
        if scarcity in {"replaceable", "highly_replaceable", "replaceable_middle"}:
            return "avoid_overcommitting"
        return "wait_and_save_flexibility"

    if support == "weak":
        if scarcity in {"replaceable", "highly_replaceable"} and width in {"wide", "very_wide"}:
            return "avoid_overcommitting"
        return "wait_and_save_flexibility"

    if support == "strong":
        if (
            scarcity == "scarce"
            and ambiguity == "low_ambiguity"
            and width in {"narrow", "balanced"}
            and drift in {"stable_profile", "manageable_drift"}
        ):
            return "offer_now"
        return "offer_now_disciplined_band"

    if scarcity in {"scarce", "selective_premium"} and ambiguity != "high_ambiguity" and width != "very_wide":
        return "offer_now_disciplined_band"
    if scarcity == "highly_replaceable":
        return "avoid_overcommitting"
    return "wait_and_save_flexibility"


def build_reason(row: pd.Series) -> str:
    parts: list[str] = []

    support = row["comp_support_bucket"]
    if support == "strong":
        parts.append("comp support is strong")
    elif support == "moderate":
        parts.append("comp support is usable but not airtight")
    elif support == "weak":
        parts.append("comp support is weak")
    else:
        parts.append("comp support is insufficient")

    scarcity = row["scarcity_tier"]
    if scarcity == "scarce":
        parts.append("role family is relatively scarce")
    elif scarcity == "selective_premium":
        parts.append("frontcourt scoring archetype can justify a selective premium")
    elif scarcity == "replaceable_middle":
        parts.append("wing archetype sits in the replaceable middle")
    elif scarcity == "replaceable":
        parts.append("archetype is fairly replaceable")
    else:
        parts.append("archetype is highly replaceable")

    ambiguity = row["ambiguity_band"]
    if ambiguity == "high_ambiguity":
        parts.append("prototype fit remains ambiguous")
    elif ambiguity == "medium_ambiguity":
        parts.append("prototype fit is only moderately clear")
    elif ambiguity == "low_ambiguity":
        parts.append("prototype fit is relatively clear")

    width = row["anchor_width_band"]
    if width == "very_wide":
        parts.append("market anchor band is very wide")
    elif width == "wide":
        parts.append("market anchor band is wide")
    elif width == "balanced":
        parts.append("market anchor band is balanced")
    elif width == "narrow":
        parts.append("market anchor band is narrow")

    drift = row["drift_signal"]
    if drift == "elevated_drift_risk":
        parts.append("role identity is still shifting materially")
    elif drift == "unresolved_drift":
        parts.append("drift evidence is unresolved")
    elif drift == "manageable_drift":
        parts.append("drift looks manageable")
    elif drift == "stable_profile":
        parts.append("identity looks stable")

    action = row["provisional_action_bucket"]
    if action == "offer_now":
        lead = "Offer now"
    elif action == "offer_now_disciplined_band":
        lead = "Offer now, but stay disciplined"
    elif action == "wait_and_save_flexibility":
        lead = "Wait and save flexibility"
    else:
        lead = "Avoid overcommitting"

    return f"{lead}: " + "; ".join(parts) + "."


def decision_card_label(row: pd.Series) -> str:
    if bool(row.get("market_anchor_supported_flag", False)):
        return (
            f"{row['PLAYER_NAME']} — {row['macro_archetype']} | "
            f"Protected {row['protected_price_cap_pct']:.3f}, "
            f"fair {row['fair_price_cap_pct']:.3f}, "
            f"walk-away {row['walk_away_max_cap_pct']:.3f}"
        )
    return f"{row['PLAYER_NAME']} — {row['macro_archetype']} | market anchor not supported"


def make_rule_reference(thresholds: dict[str, float]) -> pd.DataFrame:
    rows = [
        {
            "rule_group": "comp_support_logic",
            "rule_name": "strong",
            "rule_description": (
                "Supported market anchor, good match count/rate, credible effective comp depth, "
                "reasonable same-macro share, and non-excessive anchor width."
            ),
        },
        {
            "rule_group": "comp_support_logic",
            "rule_name": "moderate",
            "rule_description": (
                "Usable comp evidence exists, but one or more support signals are only middle-strength."
            ),
        },
        {
            "rule_group": "comp_support_logic",
            "rule_name": "weak",
            "rule_description": (
                "Market anchor exists, but support is thin, noisy, or too wide to justify an aggressive stance."
            ),
        },
        {
            "rule_group": "comp_support_logic",
            "rule_name": "insufficient",
            "rule_description": (
                "Comp salary evidence is missing or too thin to support a meaningful current market band."
            ),
        },
        {
            "rule_group": "action_template",
            "rule_name": "offer_now",
            "rule_description": (
                "Reserved for rare Block-2-only cases: scarce archetype, strong comp support, clear fit, "
                "and a controlled anchor band."
            ),
        },
        {
            "rule_group": "action_template",
            "rule_name": "offer_now_disciplined_band",
            "rule_description": (
                "Comp evidence is credible enough to act now, but the offer should stay inside the anchored band."
            ),
        },
        {
            "rule_group": "action_template",
            "rule_name": "wait_and_save_flexibility",
            "rule_description": (
                "The player may still merit a deal later, but Block 2 evidence alone is not strong enough to force action now."
            ),
        },
        {
            "rule_group": "action_template",
            "rule_name": "avoid_overcommitting",
            "rule_description": (
                "Use when support is weak or absent and the archetype is too replaceable to justify aggressive early commitment."
            ),
        },
        {"rule_group": "threshold_reference", "rule_name": "ambiguity_q25", "rule_description": f"{thresholds['ambiguity_q25']:.6f}"},
        {"rule_group": "threshold_reference", "rule_name": "ambiguity_q75", "rule_description": f"{thresholds['ambiguity_q75']:.6f}"},
        {"rule_group": "threshold_reference", "rule_name": "width_q25", "rule_description": f"{thresholds['width_q25']:.6f}"},
        {"rule_group": "threshold_reference", "rule_name": "width_q50", "rule_description": f"{thresholds['width_q50']:.6f}"},
        {"rule_group": "threshold_reference", "rule_name": "width_q75", "rule_description": f"{thresholds['width_q75']:.6f}"},
        {"rule_group": "threshold_reference", "rule_name": "effective_comp_n_q25", "rule_description": f"{thresholds['eff_n_q25']:.6f}"},
        {"rule_group": "threshold_reference", "rule_name": "effective_comp_n_q50", "rule_description": f"{thresholds['eff_n_q50']:.6f}"},
        {"rule_group": "threshold_reference", "rule_name": "same_macro_share_q25", "rule_description": f"{thresholds['same_macro_q25']:.6f}"},
        {"rule_group": "threshold_reference", "rule_name": "same_macro_share_q50", "rule_description": f"{thresholds['same_macro_q50']:.6f}"},
    ]
    return pd.DataFrame(rows)


def make_markdown_summary(summary_df: pd.DataFrame, input_map: dict[str, Path]) -> str:
    lines = [
        "# Deliverable 3 Provisional Decision Inputs",
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
    role_paths = resolve_role_paths(project_root, inventory_df)

    market_anchor_csv = resolve_optional_input(
        project_root=project_root,
        explicit_path=args.market_anchor_csv.resolve() if args.market_anchor_csv is not None else None,
        default_candidates=[
            output_dir / "salary_market_anchor_band_table.csv",
            project_root / "salary_market_anchor_band_table.csv",
        ],
        basename="salary_market_anchor_band_table.csv",
    )
    if market_anchor_csv is None:
        raise FileNotFoundError(
            "Could not resolve salary_market_anchor_band_table.csv. Run Phase 2 first or pass --market-anchor-csv."
        )

    holdout_csv = resolve_optional_input(
        project_root=project_root,
        explicit_path=args.holdout_csv.resolve() if args.holdout_csv is not None else None,
        default_candidates=[
            output_dir / "salary_demo_player_holdout_manifest.csv",
            project_root / "salary_demo_player_holdout_manifest.csv",
        ],
        basename="salary_demo_player_holdout_manifest.csv",
    )

    anchor_df = pd.read_csv(market_anchor_csv)
    block2_df = pd.read_csv(role_paths["deliverable3_block2_context"])

    require_columns(
        anchor_df,
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
            "comp_salary_anchor_support",
            "detail_effective_comp_n",
            "detail_same_macro_share",
            "detail_anchor_std_weighted",
            "protected_price_cap_pct",
            "fair_price_cap_pct",
            "walk_away_max_cap_pct",
            "anchor_band_width",
            "anchor_band_relative_width",
            "market_anchor_supported_flag",
        ],
        "salary_market_anchor_band_table.csv",
    )
    require_columns(
        block2_df,
        [
            "PLAYER_ID",
            "realistic_comp_list",
            "realistic_comp_similarity_mean",
            "supporting_shot_style_explanation",
            "block2_market_context_interpretation",
            "historical_year5_salary_cap_pct_observed",
            "historical_year5_salary_observed_flag",
        ],
        "deliverable3_block2_archetype_comp_market_context.csv",
    )

    enrich_cols = [
        "PLAYER_ID",
        "realistic_comp_list",
        "realistic_comp_similarity_mean",
        "supporting_shot_style_explanation",
        "block2_market_context_interpretation",
        "historical_year5_salary_cap_pct_observed",
        "historical_year5_salary_observed_flag",
    ]
    df = anchor_df.merge(block2_df[enrich_cols], on="PLAYER_ID", how="left", suffixes=("", "_block2"))

    if holdout_csv is not None and holdout_csv.exists():
        holdout_df = pd.read_csv(holdout_csv)
        holdout_map = holdout_df[["PLAYER_ID", "holdout_for_case_study", "holdout_reason"]].drop_duplicates("PLAYER_ID")
        df = df.drop(columns=[c for c in ["holdout_for_case_study", "holdout_reason"] if c in df.columns], errors="ignore")
        df = df.merge(holdout_map, on="PLAYER_ID", how="left")
        df["holdout_for_case_study"] = df["holdout_for_case_study"].fillna(False).astype(bool)
        df["holdout_reason"] = df["holdout_reason"].fillna("")
    else:
        demo_names = {normalize_demo_name(name) for name in ["Trae Young", "Nikola Vučević", "Nikola Vucevic"]}
        df["holdout_for_case_study"] = df["PLAYER_NAME"].astype(str).map(lambda x: normalize_demo_name(x) in demo_names)
        df["holdout_reason"] = np.where(
            df["holdout_for_case_study"],
            "Reserved for Deliverable 3 stakeholder-facing salary suggestion demo.",
            "",
        )

    supported_mask = pd.to_numeric(df["market_anchor_supported_flag"], errors="coerce").fillna(0).astype(int).astype(bool)
    supported_df = df.loc[supported_mask].copy()
    ambiguity_q25 = float(supported_df["prototype_fit_ambiguity"].quantile(0.25)) if not supported_df.empty else 0.67
    ambiguity_q75 = float(supported_df["prototype_fit_ambiguity"].quantile(0.75)) if not supported_df.empty else 0.90
    width_q25 = float(supported_df["anchor_band_relative_width"].quantile(0.25)) if not supported_df.empty else 0.42
    width_q50 = float(supported_df["anchor_band_relative_width"].quantile(0.50)) if not supported_df.empty else 0.78
    width_q75 = float(supported_df["anchor_band_relative_width"].quantile(0.75)) if not supported_df.empty else 1.26
    eff_n_q25 = float(supported_df["detail_effective_comp_n"].quantile(0.25)) if not supported_df.empty else 5.0
    eff_n_q50 = float(supported_df["detail_effective_comp_n"].quantile(0.50)) if not supported_df.empty else 6.15
    same_macro_q25 = float(supported_df["detail_same_macro_share"].quantile(0.25)) if not supported_df.empty else 0.25
    same_macro_q50 = float(supported_df["detail_same_macro_share"].quantile(0.50)) if not supported_df.empty else 0.60

    thresholds = {
        "ambiguity_q25": ambiguity_q25,
        "ambiguity_q75": ambiguity_q75,
        "width_q25": width_q25,
        "width_q50": width_q50,
        "width_q75": width_q75,
        "eff_n_q25": eff_n_q25,
        "eff_n_q50": eff_n_q50,
        "same_macro_q25": same_macro_q25,
        "same_macro_q50": same_macro_q50,
    }

    scarcity_ref = make_scarcity_reference()
    df = df.merge(scarcity_ref, on="macro_archetype", how="left")

    df["ambiguity_band"] = ambiguity_band(df["prototype_fit_ambiguity"], ambiguity_q25, ambiguity_q75)
    df["anchor_width_band"] = width_band(df["anchor_band_relative_width"], width_q25, width_q50, width_q75)
    df["drift_signal"] = df["identity_drift_class"].map(drift_signal)
    df["comp_support_bucket"] = df.apply(lambda row: comp_support_bucket(row, thresholds), axis=1)
    df["provisional_action_bucket"] = df.apply(provisional_action_bucket, axis=1)
    df["provisional_action_reason"] = df.apply(build_reason, axis=1)
    df["decision_card_label"] = df.apply(decision_card_label, axis=1)
    df["decision_status"] = "provisional_block2_only"

    df["decision_card_band_text"] = np.where(
        pd.to_numeric(df["market_anchor_supported_flag"], errors="coerce").fillna(0).astype(int).astype(bool),
        df["market_anchor_band_text"],
        "Comp-based market anchor not supported.",
    )
    df["decision_card_support_text"] = (
        "Comp support "
        + df["comp_support_bucket"].astype(str)
        + "; drift "
        + df["drift_signal"].astype(str)
        + "; archetype "
        + df["scarcity_tier"].astype(str)
        + "."
    )

    output_cols = [
        "PLAYER_ID",
        "PLAYER_NAME",
        "draft_year",
        "macro_archetype",
        "shot_style_subtype",
        "hybrid_archetype_label",
        "scarcity_tier",
        "scarcity_wording",
        "prototype_fit_ambiguity",
        "ambiguity_band",
        "identity_drift_class",
        "drift_signal",
        "realistic_comp_list",
        "realistic_comp_similarity_mean",
        "comp_salary_match_count",
        "comp_salary_match_rate",
        "comp_salary_anchor_support",
        "detail_effective_comp_n",
        "detail_same_macro_share",
        "detail_anchor_std_weighted",
        "comp_support_bucket",
        "protected_price_cap_pct",
        "fair_price_cap_pct",
        "walk_away_max_cap_pct",
        "anchor_band_width",
        "anchor_band_relative_width",
        "anchor_width_band",
        "market_anchor_supported_flag",
        "market_anchor_band_text",
        "decision_card_band_text",
        "decision_card_support_text",
        "provisional_action_bucket",
        "provisional_action_reason",
        "decision_card_label",
        "decision_status",
        "supporting_shot_style_explanation",
        "block2_market_context_interpretation",
        "historical_year5_salary_cap_pct_observed",
        "historical_year5_salary_observed_flag",
        "holdout_for_case_study",
        "holdout_reason",
    ]
    final_df = df[output_cols].sort_values(["draft_year", "PLAYER_NAME"]).reset_index(drop=True)

    summary_rows = [
        {
            "metric_group": "workflow_status",
            "metric": "players_processed",
            "value": int(len(final_df)),
            "notes": "Total players carried into provisional Deliverable 3 decision-input assembly.",
        },
        {
            "metric_group": "workflow_status",
            "metric": "players_with_supported_market_anchor",
            "value": int(pd.to_numeric(final_df["market_anchor_supported_flag"], errors="coerce").fillna(0).astype(int).sum()),
            "notes": "Players with a usable protected/fair/walk-away comp-based band from Phase 2.",
        },
    ]
    for bucket, count in final_df["comp_support_bucket"].value_counts(dropna=False).sort_index().items():
        summary_rows.append(
            {
                "metric_group": "comp_support_bucket",
                "metric": str(bucket),
                "value": int(count),
                "notes": "Counts after applying the conservative Block 2 comp-support logic.",
            }
        )
    for bucket, count in final_df["provisional_action_bucket"].value_counts(dropna=False).sort_index().items():
        summary_rows.append(
            {
                "metric_group": "provisional_action_bucket",
                "metric": str(bucket),
                "value": int(count),
                "notes": "Provisional stance counts before any Deliverable 1 forecast adjustment.",
            }
        )
    for tier, count in final_df["scarcity_tier"].value_counts(dropna=False).sort_index().items():
        summary_rows.append(
            {
                "metric_group": "scarcity_tier",
                "metric": str(tier),
                "value": int(count),
                "notes": "Role-scarcity mapping inherited from the macro-archetype family.",
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    support_breakdown_df = (
        final_df.groupby(["comp_support_bucket", "provisional_action_bucket"], dropna=False)
        .size()
        .reset_index(name="player_count")
        .sort_values(["comp_support_bucket", "provisional_action_bucket"])
    )
    action_by_scarcity_df = (
        final_df.groupby(["scarcity_tier", "provisional_action_bucket"], dropna=False)
        .size()
        .reset_index(name="player_count")
        .sort_values(["scarcity_tier", "provisional_action_bucket"])
    )
    action_rule_reference_df = make_rule_reference(thresholds)

    outputs = {
        "salary_provisional_decision_inputs_table.csv": final_df,
        "salary_provisional_decision_inputs_summary.csv": summary_df,
        "salary_provisional_comp_support_breakdown.csv": support_breakdown_df,
        "salary_provisional_action_by_scarcity.csv": action_by_scarcity_df,
        "salary_comp_support_logic_reference.csv": action_rule_reference_df[action_rule_reference_df["rule_group"].isin(["comp_support_logic", "threshold_reference"])],
        "salary_scarcity_wording_reference.csv": scarcity_ref,
        "salary_provisional_action_rule_reference.csv": action_rule_reference_df[action_rule_reference_df["rule_group"].eq("action_template")],
    }
    for file_name, frame in outputs.items():
        frame.to_csv(output_dir / file_name, index=False)

    summary_md = make_markdown_summary(
        summary_df=summary_df,
        input_map={
            "deliverable3_block2_context": role_paths["deliverable3_block2_context"],
            "market_anchor_table": market_anchor_csv,
            **({"holdout_manifest": holdout_csv} if holdout_csv is not None else {}),
        },
    )
    (output_dir / "salary_provisional_decision_inputs_summary.md").write_text(summary_md, encoding="utf-8")

    metadata = {
        "script": SCRIPT_NAME,
        "project_root": str(project_root),
        "inputs": {
            "deliverable3_block2_context": str(role_paths["deliverable3_block2_context"]),
            "market_anchor_table": str(market_anchor_csv),
            "holdout_manifest": str(holdout_csv) if holdout_csv is not None else None,
        },
        "thresholds": thresholds,
        "notes": [
            "This phase is intentionally provisional and Block-2-only.",
            "No Deliverable 1 forecast premiums, sleeper/bust probabilities, durability penalties, or uncertainty penalties are applied here.",
            "The output is designed to prepare a later final-guidance layer rather than claim that the current provisional stance is final.",
        ],
    }
    write_json(output_dir / "salary_provisional_decision_inputs_metadata.json", metadata)

    log_text = f"""
Phase 3 complete — provisional Deliverable 3 decision inputs assembled.

Inputs used:
- deliverable3_block2_context: {role_paths['deliverable3_block2_context']}
- market_anchor_table: {market_anchor_csv}
- holdout_manifest: {holdout_csv}

Main outputs:
- salary_provisional_decision_inputs_table.csv
- salary_provisional_decision_inputs_summary.csv
- salary_provisional_comp_support_breakdown.csv
- salary_provisional_action_by_scarcity.csv
- salary_comp_support_logic_reference.csv
- salary_scarcity_wording_reference.csv
- salary_provisional_action_rule_reference.csv

Workflow logic:
- mapped macro archetypes to scarcity / replaceability wording,
- classified prototype ambiguity and anchor-width bands using empirical quartiles from supported anchors,
- built conservative comp-support tiers from match count, match rate, effective comp depth, same-macro share, and anchor width,
- assigned provisional action buckets using Block 2 comp evidence and risk proxies only,
- preserved Trae Young and Nikola Vučević as case-study holdouts when the holdout manifest was available.

Important boundary:
- the resulting stance remains provisional_block2_only and must not be treated as the final extension recommendation.
"""
    append_workflow_log(workflow_log_path, log_text)
    print(f"[{SCRIPT_NAME}] Wrote provisional Deliverable 3 decision-input outputs to {output_dir}")


if __name__ == "__main__":
    main()
