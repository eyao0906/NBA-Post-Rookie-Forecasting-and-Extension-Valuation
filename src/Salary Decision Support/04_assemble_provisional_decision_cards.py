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
    write_json,
)

SCRIPT_NAME = "04_assemble_provisional_decision_cards.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble stakeholder-facing provisional Deliverable 3 decision cards from the "
            "Phase 3 decision-input table. This phase remains intentionally provisional and "
            "must not be treated as the final extension-guidance layer."
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
        "--decision-inputs-csv",
        type=Path,
        default=None,
        help=(
            "Optional explicit path to salary_provisional_decision_inputs_table.csv from Phase 3. "
            "If omitted, the script will search standard output locations first."
        ),
    )
    parser.add_argument(
        "--log-name",
        type=str,
        default="salary_decision_support_workflow_log.txt",
        help="Workflow log file name written under the output directory.",
    )
    return parser.parse_args()


def resolve_input(project_root: Path, explicit_path: Path | None, output_dir: Path) -> Path:
    if explicit_path is not None and explicit_path.exists():
        return explicit_path.resolve()
    candidates = [
        output_dir / "salary_provisional_decision_inputs_table.csv",
        project_root / "Output" / "Salary Decision Support" / "salary_provisional_decision_inputs_table.csv",
        project_root / "salary_provisional_decision_inputs_table.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    matches = list(project_root.rglob("salary_provisional_decision_inputs_table.csv"))
    if matches:
        matches.sort(key=lambda p: (len(p.parts), str(p).lower()))
        return matches[0].resolve()
    raise FileNotFoundError(
        "Could not resolve salary_provisional_decision_inputs_table.csv. Run Phase 3 first or pass --decision-inputs-csv."
    )


def require_columns(df: pd.DataFrame, required: list[str], table_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def pct_text(value: Any, digits: int = 1) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.{digits}f}%"


def action_label(action: str) -> str:
    mapping = {
        "offer_now": "Offer now",
        "offer_now_disciplined_band": "Offer now at disciplined price",
        "wait_and_save_flexibility": "Wait and save flexibility",
        "avoid_overcommitting": "Avoid overcommitting",
    }
    return mapping.get(str(action), str(action))


def support_sentence(bucket: str) -> str:
    mapping = {
        "strong": "Comp support is strong enough to anchor a usable current market band.",
        "moderate": "Comp support is usable, but not strong enough to remove pricing caution.",
        "weak": "Comp support exists, but remains thin or noisy.",
        "insufficient": "Comp support is insufficient for a credible current market anchor.",
    }
    return mapping.get(str(bucket), "Comp support signal unavailable.")


def drift_sentence(signal: str) -> str:
    mapping = {
        "stable_profile": "Role identity looks stable through the rookie-contract window.",
        "manageable_drift": "Role identity evolved, but the drift still looks manageable.",
        "elevated_drift_risk": "Role identity is still shifting materially, so early commitment deserves caution.",
        "unresolved_drift": "Drift evidence is unresolved, so confidence should stay conservative.",
    }
    return mapping.get(str(signal), "Drift signal unavailable.")


def ambiguity_sentence(band: str) -> str:
    mapping = {
        "low_ambiguity": "Prototype fit is relatively clear.",
        "medium_ambiguity": "Prototype fit is only moderately clear.",
        "high_ambiguity": "Prototype fit remains ambiguous.",
        "unknown": "Prototype-fit clarity is unavailable.",
    }
    return mapping.get(str(band), "Prototype-fit clarity unavailable.")


def band_width_sentence(band: str) -> str:
    mapping = {
        "narrow": "The anchored price band is tight.",
        "balanced": "The anchored price band is reasonably controlled.",
        "wide": "The anchored price band is wide.",
        "very_wide": "The anchored price band is very wide.",
        "unknown": "The anchored price band is not available.",
    }
    return mapping.get(str(band), "The anchored price band is unavailable.")


def action_summary(action: str) -> str:
    mapping = {
        "offer_now": "Current Block 2 evidence is strong enough to justify acting now.",
        "offer_now_disciplined_band": "Current Block 2 evidence supports acting now, but only inside the anchored band.",
        "wait_and_save_flexibility": "Current Block 2 evidence does not force immediate commitment; preserving flexibility still has value.",
        "avoid_overcommitting": "Current Block 2 evidence does not support aggressive early commitment.",
    }
    return mapping.get(str(action), "Provisional action summary unavailable.")


def build_market_band_text(row: pd.Series) -> str:
    if not bool(row.get("market_anchor_supported_flag", False)):
        return "Comp-based market anchor not supported."
    return (
        f"Protected price {pct_text(row.get('protected_price_cap_pct'))}, "
        f"fair price {pct_text(row.get('fair_price_cap_pct'))}, "
        f"walk-away max {pct_text(row.get('walk_away_max_cap_pct'))}."
    )


def build_card_title(row: pd.Series) -> str:
    return f"{row['PLAYER_NAME']} — {action_label(row['provisional_action_bucket'])}"


def build_identity_text(row: pd.Series) -> str:
    subtype = row.get("shot_style_subtype")
    subtype_text = "No shot subtype" if pd.isna(subtype) or str(subtype).strip() == "" else str(subtype)
    return (
        f"{row['macro_archetype']} | {subtype_text}. "
        f"{row['scarcity_wording']}"
    )


def build_risk_text(row: pd.Series) -> str:
    parts = [
        support_sentence(str(row.get("comp_support_bucket", ""))),
        ambiguity_sentence(str(row.get("ambiguity_band", ""))),
        drift_sentence(str(row.get("drift_signal", ""))),
        band_width_sentence(str(row.get("anchor_width_band", ""))),
    ]
    return " ".join(parts)


def build_context_text(row: pd.Series) -> str:
    comp_list = row.get("realistic_comp_list")
    comp_text = "Realistic comp list unavailable."
    if pd.notna(comp_list) and str(comp_list).strip() != "":
        comp_text = f"Realistic comps: {str(comp_list)}."
    market_text = row.get("block2_market_context_interpretation")
    market_sentence = ""
    if pd.notna(market_text) and str(market_text).strip() != "":
        market_sentence = str(market_text).strip()
        if not market_sentence.endswith("."):
            market_sentence += "."
    return " ".join([comp_text, market_sentence]).strip()


def build_status_text(row: pd.Series) -> str:
    return (
        f"Status: {row['decision_status']}. "
        f"This is not the final extension recommendation; Deliverable 1 forecast, durability, "
        f"and uncertainty adjustments are still pending."
    )


def build_summary_text(row: pd.Series) -> str:
    pieces = [
        action_summary(str(row.get("provisional_action_bucket", ""))),
        build_market_band_text(row),
        support_sentence(str(row.get("comp_support_bucket", ""))),
    ]
    return " ".join(pieces)


def markdown_snippet(row: pd.Series) -> str:
    lines = [
        f"### {build_card_title(row)}",
        "",
        f"- **Player:** {row['PLAYER_NAME']}",
        f"- **Draft year:** {int(row['draft_year']) if pd.notna(row['draft_year']) else 'N/A'}",
        f"- **Hybrid archetype:** {row.get('hybrid_archetype_label', 'N/A')}",
        f"- **Provisional stance:** {action_label(row['provisional_action_bucket'])}",
        f"- **Market band:** {build_market_band_text(row)}",
        f"- **Comp support:** {row.get('comp_support_bucket', 'N/A')}",
        f"- **Scarcity tier:** {row.get('scarcity_tier', 'N/A')}",
        f"- **Why:** {row.get('provisional_action_reason', '')}",
        f"- **Status note:** {build_status_text(row)}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else project_root / "Output" / "Salary Decision Support"
    ensure_dir(output_dir)
    workflow_log_path = output_dir / args.log_name

    inputs_csv = resolve_input(
        project_root=project_root,
        explicit_path=args.decision_inputs_csv.resolve() if args.decision_inputs_csv is not None else None,
        output_dir=output_dir,
    )

    df = pd.read_csv(inputs_csv)
    require_columns(
        df,
        [
            "PLAYER_ID",
            "PLAYER_NAME",
            "draft_year",
            "macro_archetype",
            "shot_style_subtype",
            "hybrid_archetype_label",
            "scarcity_tier",
            "scarcity_wording",
            "ambiguity_band",
            "identity_drift_class",
            "drift_signal",
            "comp_support_bucket",
            "protected_price_cap_pct",
            "fair_price_cap_pct",
            "walk_away_max_cap_pct",
            "market_anchor_supported_flag",
            "provisional_action_bucket",
            "provisional_action_reason",
            "decision_status",
        ],
        "salary_provisional_decision_inputs_table.csv",
    )

    cards = df.copy()
    cards["decision_card_title"] = cards.apply(build_card_title, axis=1)
    cards["decision_card_market_band_text"] = cards.apply(build_market_band_text, axis=1)
    cards["decision_card_identity_text"] = cards.apply(build_identity_text, axis=1)
    cards["decision_card_risk_text"] = cards.apply(build_risk_text, axis=1)
    cards["decision_card_context_text"] = cards.apply(build_context_text, axis=1)
    cards["decision_card_summary_text"] = cards.apply(build_summary_text, axis=1)
    cards["decision_card_status_text"] = cards.apply(build_status_text, axis=1)
    cards["provisional_action_label"] = cards["provisional_action_bucket"].map(action_label)
    cards["protected_price_pct_text"] = cards["protected_price_cap_pct"].map(pct_text)
    cards["fair_price_pct_text"] = cards["fair_price_cap_pct"].map(pct_text)
    cards["walk_away_max_pct_text"] = cards["walk_away_max_cap_pct"].map(pct_text)
    cards["decision_card_ready_flag"] = True
    cards["case_study_priority"] = np.where(cards.get("holdout_for_case_study", False).fillna(False), "reserved_demo", "full_cohort")

    output_cols = [
        "PLAYER_ID",
        "PLAYER_NAME",
        "draft_year",
        "decision_card_title",
        "provisional_action_bucket",
        "provisional_action_label",
        "macro_archetype",
        "shot_style_subtype",
        "hybrid_archetype_label",
        "scarcity_tier",
        "scarcity_wording",
        "comp_support_bucket",
        "ambiguity_band",
        "identity_drift_class",
        "drift_signal",
        "anchor_width_band",
        "market_anchor_supported_flag",
        "protected_price_cap_pct",
        "fair_price_cap_pct",
        "walk_away_max_cap_pct",
        "protected_price_pct_text",
        "fair_price_pct_text",
        "walk_away_max_pct_text",
        "decision_card_market_band_text",
        "decision_card_identity_text",
        "decision_card_risk_text",
        "decision_card_context_text",
        "decision_card_summary_text",
        "provisional_action_reason",
        "decision_card_status_text",
        "decision_status",
        "supporting_shot_style_explanation",
        "block2_market_context_interpretation",
        "realistic_comp_list",
        "historical_year5_salary_cap_pct_observed",
        "historical_year5_salary_observed_flag",
        "holdout_for_case_study",
        "holdout_reason",
        "case_study_priority",
        "decision_card_ready_flag",
    ]
    cards_out = cards[output_cols].sort_values(["draft_year", "PLAYER_NAME"]).reset_index(drop=True)

    case_mask = cards_out["holdout_for_case_study"].fillna(False).astype(bool)
    case_df = cards_out.loc[case_mask].copy()
    if case_df.empty:
        case_df = cards_out.head(2).copy()
        case_df["holdout_reason"] = case_df.get("holdout_reason", "").fillna("")
        case_df["holdout_reason"] = np.where(
            case_df["holdout_reason"].eq(""),
            "Fallback case-study extract because no explicit holdout manifest was available.",
            case_df["holdout_reason"],
        )

    manifest_df = cards_out[[
        "PLAYER_ID",
        "PLAYER_NAME",
        "provisional_action_bucket",
        "provisional_action_label",
        "holdout_for_case_study",
        "holdout_reason",
        "case_study_priority",
        "decision_card_ready_flag",
    ]].copy()

    summary_rows = [
        {
            "metric_group": "workflow_status",
            "metric": "decision_cards_built",
            "value": int(len(cards_out)),
            "notes": "Total player-level provisional decision cards assembled from Phase 3 inputs.",
        },
        {
            "metric_group": "workflow_status",
            "metric": "case_study_rows",
            "value": int(len(case_df)),
            "notes": "Rows retained in the provisional case-study extract.",
        },
    ]
    for bucket, count in cards_out["provisional_action_bucket"].value_counts(dropna=False).sort_index().items():
        summary_rows.append(
            {
                "metric_group": "provisional_action_bucket",
                "metric": str(bucket),
                "value": int(count),
                "notes": "Decision-card counts by provisional Block-2-only action bucket.",
            }
        )
    for bucket, count in cards_out["comp_support_bucket"].value_counts(dropna=False).sort_index().items():
        summary_rows.append(
            {
                "metric_group": "comp_support_bucket",
                "metric": str(bucket),
                "value": int(count),
                "notes": "Decision-card counts by comp-support tier inherited from Phase 3.",
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    md_lines = [
        "# Deliverable 3 Provisional Decision Cards",
        "",
        "This export assembles stakeholder-facing provisional decision cards from the Phase 3 decision-input table.",
        "The cards remain **provisional_block2_only** and must not be treated as final extension guidance until Deliverable 1 is integrated.",
        "",
        "## Inputs",
        "",
        f"- `salary_provisional_decision_inputs_table.csv`: `{inputs_csv}`",
        "",
        "## Case-study snippets",
        "",
    ]
    for _, row in case_df.iterrows():
        md_lines.append(markdown_snippet(row))
    md_lines.append("")
    md_lines.append("## Summary metrics")
    md_lines.append("")
    md_lines.append("| Metric group | Metric | Value | Notes |")
    md_lines.append("|---|---|---:|---|")
    for _, row in summary_df.iterrows():
        md_lines.append(
            f"| {row['metric_group']} | {row['metric']} | {int(row['value']) if pd.notna(row['value']) else ''} | {row['notes']} |"
        )
    summary_md = "\n".join(md_lines) + "\n"

    cards_out.to_csv(output_dir / "deliverable3_provisional_decision_cards.csv", index=False)
    case_df.to_csv(output_dir / "deliverable3_provisional_case_study_table.csv", index=False)
    manifest_df.to_csv(output_dir / "deliverable3_provisional_decision_card_manifest.csv", index=False)
    summary_df.to_csv(output_dir / "deliverable3_provisional_decision_cards_summary.csv", index=False)
    (output_dir / "deliverable3_provisional_decision_cards_summary.md").write_text(summary_md, encoding="utf-8")

    metadata = {
        "script": SCRIPT_NAME,
        "project_root": str(project_root),
        "inputs": {
            "decision_inputs_table": str(inputs_csv),
        },
        "outputs": [
            "deliverable3_provisional_decision_cards.csv",
            "deliverable3_provisional_case_study_table.csv",
            "deliverable3_provisional_decision_card_manifest.csv",
            "deliverable3_provisional_decision_cards_summary.csv",
            "deliverable3_provisional_decision_cards_summary.md",
        ],
        "notes": [
            "This phase assembles stakeholder-facing decision-card outputs from the Phase 3 table.",
            "The stance remains provisional_block2_only.",
            "Trae Young and Nikola Vučević stay reserved as case-study demos when present in the holdout manifest.",
        ],
    }
    write_json(output_dir / "deliverable3_provisional_decision_cards_metadata.json", metadata)

    log_text = f"""
Phase 4 complete — provisional Deliverable 3 decision cards assembled.

Inputs used:
- decision_inputs_table: {inputs_csv}

Main outputs:
- deliverable3_provisional_decision_cards.csv
- deliverable3_provisional_case_study_table.csv
- deliverable3_provisional_decision_card_manifest.csv
- deliverable3_provisional_decision_cards_summary.csv
- deliverable3_provisional_decision_cards_summary.md

Workflow logic:
- converted Phase 3 decision inputs into stakeholder-facing provisional decision-card text fields,
- preserved protected / fair / walk-away band language in readable card form,
- assembled identity, risk, context, and status text blocks for later reporting and visuals,
- extracted case-study rows with holdout priority for Trae Young and Nikola Vučević when available.

Important boundary:
- the cards remain provisional_block2_only and must not be presented as final extension guidance.
"""
    append_workflow_log(workflow_log_path, log_text)
    print(f"[{SCRIPT_NAME}] Wrote provisional decision-card outputs to {output_dir}")


if __name__ == "__main__":
    main()
