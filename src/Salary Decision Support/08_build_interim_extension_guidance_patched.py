from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DEMO_PLAYERS = ["Trae Young", "Nikola Vučević", "Nikola Vucevic", "Andrew Bynum"]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_output_dir(project_root: Path) -> Path:
    candidates = [
        project_root / "Output/Salary%20Decision%20Support",
        project_root / "Output/Salary Decision Support",
    ]
    for c in candidates:
        if c.exists():
            return ensure_dir(c)
    return ensure_dir(candidates[0])


def append_workflow_log(log_path: Path, text_block: str) -> None:
    ensure_dir(log_path.parent)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n{'=' * 80}\n")
        fh.write(f"[{timestamp}]\n")
        fh.write(text_block.rstrip() + "\n")


def find_existing_path(project_root: Path, candidates: list[str]) -> Path:
    for candidate in candidates:
        path = project_root / candidate
        if path.exists():
            return path
    for candidate in candidates:
        basename = Path(candidate).name
        matches = sorted(project_root.rglob(basename), key=lambda p: (len(p.parts), str(p)))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not locate any of: {candidates}")


def pct_text(x: float | int | None) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{float(x) * 100:.1f}%"


def derive_forecast_support_tier(row: pd.Series) -> str:
    available = int(row.get("forecast_overlay_available_flag", 0))
    signal = str(row.get("forecast_overlay_signal", ""))
    outlook = str(row.get("forecast_outlook_bucket", ""))
    conf = str(row.get("forecast_confidence_proxy", ""))
    perf = str(row.get("forecast_performance_tier", ""))

    if not available:
        return "no_forecast_overlay"
    if signal in {"forecast_downside_caution", "premium_not_supported_by_forecast"}:
        return "downside_caution"
    if outlook == "downside_leaning" and conf in {"high_class_confidence", "medium_class_confidence"}:
        return "downside_caution"
    if (
        outlook == "upside_leaning"
        and conf in {"high_class_confidence", "medium_class_confidence"}
        and perf == "upper_forecast_tier"
        and signal in {"within_band_with_upside_support", "mixed_forecast_signal", "discount_supported_by_forecast"}
    ):
        return "strong_upside_support"
    if outlook == "upside_leaning":
        return "mild_upside_support"
    if outlook == "neutral_leaning" and conf in {"high_class_confidence", "medium_class_confidence"}:
        return "neutral_support"
    return "mixed_or_low_confidence"


def apply_interim_extension_bucket(row: pd.Series) -> str:
    base = str(row.get("provisional_action_bucket", "wait_and_save_flexibility"))
    market_ok = bool(row.get("market_anchor_supported_flag", 0))
    support = str(row.get("forecast_support_tier", "no_forecast_overlay"))
    comp = str(row.get("comp_support_bucket", "unknown"))
    scarcity = str(row.get("scarcity_tier", "unknown"))
    ambiguity = str(row.get("ambiguity_band", "unknown"))
    width = str(row.get("anchor_width_band", "unknown"))
    align = str(row.get("model_market_alignment_bucket", "unknown"))

    if not market_ok:
        if support in {"strong_upside_support", "mild_upside_support"} and comp in {"strong", "moderate"} and scarcity in {"scarce", "selective_premium"}:
            return "wait_and_save_flexibility"
        return "avoid_overcommitting"

    if base == "offer_now_disciplined_band":
        if (
            support == "strong_upside_support"
            and comp == "strong"
            and scarcity in {"scarce", "selective_premium"}
            and ambiguity != "high_ambiguity"
            and width in {"narrow", "balanced"}
            and align in {"between_protected_and_fair", "between_fair_and_walkaway"}
        ):
            return "offer_now"
        if support == "downside_caution":
            return "wait_and_save_flexibility"
        return "offer_now_disciplined_band"

    if base == "wait_and_save_flexibility":
        if (
            support == "strong_upside_support"
            and comp in {"strong", "moderate"}
            and scarcity in {"scarce", "selective_premium"}
            and align != "above_walkaway"
        ):
            return "offer_now_disciplined_band"
        if support == "downside_caution" and (comp in {"weak", "insufficient"} or scarcity in {"highly_replaceable", "replaceable"}):
            return "avoid_overcommitting"
        return "wait_and_save_flexibility"

    if base == "avoid_overcommitting":
        if support == "strong_upside_support" and comp in {"strong", "moderate"} and scarcity in {"scarce", "selective_premium"} and market_ok:
            return "wait_and_save_flexibility"
        return "avoid_overcommitting"

    if base == "offer_now":
        if support == "downside_caution":
            return "offer_now_disciplined_band"
        return "offer_now"

    return base


def action_label(bucket: str) -> str:
    return {
        "offer_now": "Offer now",
        "offer_now_disciplined_band": "Offer now at disciplined price",
        "wait_and_save_flexibility": "Wait and save flexibility",
        "avoid_overcommitting": "Avoid overcommitting",
    }.get(bucket, bucket)


def derive_negotiation_fields(row: pd.Series) -> tuple[float | None, float | None, float | None, str]:
    protected = row.get("protected_price_cap_pct")
    fair = row.get("fair_price_cap_pct")
    walk = row.get("walk_away_max_cap_pct")
    bucket = row.get("interim_extension_bucket")
    market_ok = bool(row.get("market_anchor_supported_flag", 0))

    if not market_ok or pd.isna(protected) or pd.isna(fair):
        return np.nan, np.nan, np.nan, "No defensible negotiation band yet; keep any discussion exploratory only."

    if bucket == "offer_now":
        open_px = max(protected, (protected + fair) / 2.0)
        target_px = fair
        hard_max = walk
        posture = "Act early. Open inside the band, defend the fair price, and keep the walk-away ceiling available because the current evidence supports moving now."
    elif bucket == "offer_now_disciplined_band":
        open_px = protected
        target_px = fair
        hard_max = fair
        posture = "Open from the protected side of the band and aim for fair value. Do not chase the full walk-away ceiling without stronger later evidence."
    elif bucket == "wait_and_save_flexibility":
        open_px = protected
        target_px = protected
        hard_max = fair
        posture = "Preserve optionality. The protected price is the only clean early anchor today, and the fair price should be reserved for later confirmation."
    else:
        open_px = np.nan
        target_px = np.nan
        hard_max = protected
        posture = "Do not commit meaningfully off the current evidence. If talks occur, treat the protected price as a strict outer boundary, not a target."
    return open_px, target_px, hard_max, posture


def build_reason(row: pd.Series) -> str:
    bucket = row.get("interim_extension_bucket")
    support = str(row.get("forecast_support_tier", ""))
    scarcity = str(row.get("scarcity_tier", ""))
    comp = str(row.get("comp_support_bucket", ""))
    align = str(row.get("model_market_alignment_bucket", ""))
    drift = str(row.get("identity_drift_class", ""))
    conf = str(row.get("forecast_confidence_proxy", ""))

    scarcity_text = {
        "scarce": "role scarcity supports acting earlier than usual",
        "selective_premium": "the archetype can justify some premium, but only selectively",
        "replaceable_middle": "the role is useful but not scarce enough to overpay",
        "replaceable": "the role remains replaceable in the market",
        "highly_replaceable": "the role is highly replaceable",
    }.get(scarcity, "role context is mixed")

    support_text = {
        "strong_upside_support": "the current Deliverable 1 overlay points to credible upside support",
        "mild_upside_support": "the forecast overlay leans positive, but not strongly enough to erase price discipline",
        "neutral_support": "the forecast overlay is broadly neutral and supports holding the current band rather than stretching it",
        "downside_caution": "the forecast overlay raises downside caution against an aggressive early commitment",
        "mixed_or_low_confidence": "the forecast overlay is too mixed or low-confidence to justify a stronger move",
        "no_forecast_overlay": "no forecast overlay is available, so the salary stance should stay conservative",
    }.get(support, "forecast context is limited")

    align_text = {
        "below_protected": "the baseline salary model sits below the protected price",
        "between_protected_and_fair": "the baseline salary model supports the lower-to-middle part of the market band",
        "between_fair_and_walkaway": "the baseline salary model supports the fair-to-upper part of the market band",
        "above_walkaway": "the baseline salary model sits above the current walk-away max, so the market band should not be expanded automatically",
        "model_only_no_market_anchor": "the baseline salary model exists, but a comp-market band is not reliable enough yet",
    }.get(align, "pricing alignment is mixed")

    bucket_prefix = {
        "offer_now": "Offer now",
        "offer_now_disciplined_band": "Offer now, but stay disciplined",
        "wait_and_save_flexibility": "Wait and save flexibility",
        "avoid_overcommitting": "Avoid overcommitting",
    }.get(bucket, "Interim guidance")

    return f"{bucket_prefix}: comp support is {comp}; {support_text}; {scarcity_text}; {align_text}; drift remains {drift}; forecast confidence is {conf}."


def build_status_text(row: pd.Series) -> str:
    return (
        "Status: interim_with_current_D1_macro_forecast. This is not the final extension recommendation; "
        "durability / availability risk, richer uncertainty bands, and the fully locked Deliverable 1 merge table are still pending."
    )


def build_summary_text(row: pd.Series) -> str:
    band = row.get("market_anchor_band_text", "Comp-based market anchor not supported.")
    bucket_label = action_label(str(row.get("interim_extension_bucket", "")))
    reason = row.get("interim_extension_reason", "")
    return f"{bucket_label}. {band} {reason}"


def build_card_title(row: pd.Series) -> str:
    return f"{row.get('PLAYER_NAME', 'Unknown')} — {action_label(str(row.get('interim_extension_bucket', '')))}"


def build_inputs_table(recon: pd.DataFrame) -> pd.DataFrame:
    out = recon.copy()
    out["forecast_support_tier"] = out.apply(derive_forecast_support_tier, axis=1)
    out["interim_extension_bucket"] = out.apply(apply_interim_extension_bucket, axis=1)
    out["interim_extension_label"] = out["interim_extension_bucket"].map(action_label)

    negotiation = out.apply(derive_negotiation_fields, axis=1, result_type="expand")
    negotiation.columns = [
        "interim_open_cap_pct",
        "interim_target_cap_pct",
        "interim_hard_max_cap_pct",
        "interim_price_posture_text",
    ]
    out = pd.concat([out, negotiation], axis=1)

    out["interim_open_pct_text"] = out["interim_open_cap_pct"].map(pct_text)
    out["interim_target_pct_text"] = out["interim_target_cap_pct"].map(pct_text)
    out["interim_hard_max_pct_text"] = out["interim_hard_max_cap_pct"].map(pct_text)
    out["interim_extension_reason"] = out.apply(build_reason, axis=1)
    out["interim_status_text"] = out.apply(build_status_text, axis=1)
    out["interim_guidance_summary_text"] = out.apply(build_summary_text, axis=1)
    out["interim_guidance_title"] = out.apply(build_card_title, axis=1)
    out["interim_guidance_ready_flag"] = True
    out["interim_guidance_status"] = "interim_with_current_D1_macro_forecast"
    out["case_study_priority"] = np.where(out["PLAYER_NAME"].isin(DEMO_PLAYERS) | out.get("holdout_for_case_study", False), "demo_holdout_priority", "full_cohort")
    return out


def build_cards(table: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "PLAYER_ID", "PLAYER_NAME", "draft_year", "interim_guidance_title", "interim_extension_bucket",
        "interim_extension_label", "macro_archetype", "shot_style_subtype", "hybrid_archetype_label",
        "forecast_support_tier", "forecast_outlook_bucket", "forecast_confidence_proxy", "forecast_performance_tier",
        "comp_support_bucket", "scarcity_tier", "ambiguity_band", "identity_drift_class", "anchor_width_band",
        "model_market_alignment_bucket", "pricing_anchor_preference", "market_anchor_supported_flag",
        "protected_price_cap_pct", "fair_price_cap_pct", "walk_away_max_cap_pct",
        "interim_open_cap_pct", "interim_target_cap_pct", "interim_hard_max_cap_pct",
        "interim_open_pct_text", "interim_target_pct_text", "interim_hard_max_pct_text",
        "market_anchor_band_text", "interim_price_posture_text", "interim_extension_reason",
        "interim_guidance_summary_text", "interim_status_text", "realistic_comp_list",
        "supporting_shot_style_explanation", "block2_market_context_interpretation",
        "holdout_for_case_study", "holdout_reason", "case_study_priority", "interim_guidance_ready_flag",
    ]
    keep = [c for c in cols if c in table.columns]
    return table[keep].copy()


def build_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    rows.append({
        "metric_group": "workflow_status",
        "metric": "interim_guidance_rows",
        "value": int(len(table)),
        "notes": "Total player-level rows in the interim forecast-adjusted extension guidance layer.",
    })
    rows.append({
        "metric_group": "workflow_status",
        "metric": "forecast_overlay_available_rows",
        "value": int(table["forecast_overlay_available_flag"].fillna(0).astype(int).sum()),
        "notes": "Rows carrying the current Deliverable 1 macro forecast overlay from Phase 7.",
    })
    rows.append({
        "metric_group": "workflow_status",
        "metric": "demo_case_study_rows",
        "value": int(table[table["case_study_priority"] == "demo_holdout_priority"].shape[0]),
        "notes": "Rows reserved for Trae Young / Andrew Bynum case-study guidance when present.",
    })

    for bucket, count in table["interim_extension_bucket"].value_counts().items():
        rows.append({
            "metric_group": "interim_extension_bucket",
            "metric": str(bucket),
            "value": int(count),
            "notes": "Counts by interim forecast-adjusted extension bucket.",
        })

    for support, count in table["forecast_support_tier"].value_counts().items():
        rows.append({
            "metric_group": "forecast_support_tier",
            "metric": str(support),
            "value": int(count),
            "notes": "Counts by current Deliverable 1 macro forecast support tier.",
        })

    for pref, count in table["pricing_anchor_preference"].value_counts().items():
        rows.append({
            "metric_group": "pricing_anchor_preference",
            "metric": str(pref),
            "value": int(count),
            "notes": "Counts by how the salary model and market band align after reconciliation.",
        })
    return pd.DataFrame(rows)


def build_rule_reference() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "rule_group": "forecast_support_tier",
            "rule_name": "strong_upside_support",
            "logic": "Current Deliverable 1 overlay is upside-leaning, confidence is at least medium, performance tier is upper, and the overlay does not directly conflict with the current market band.",
            "effect": "Can promote disciplined or waiting cases one bucket, but only when scarcity and comp support also cooperate.",
        },
        {
            "rule_group": "forecast_support_tier",
            "rule_name": "neutral_support",
            "logic": "Current Deliverable 1 overlay is neutral-leaning with at least medium confidence.",
            "effect": "Usually keeps the current market band intact rather than stretching or discounting it.",
        },
        {
            "rule_group": "forecast_support_tier",
            "rule_name": "downside_caution",
            "logic": "Current Deliverable 1 overlay leans bust/downside or explicitly says the market premium is not supported.",
            "effect": "Demotes aggressive early commitment by one bucket.",
        },
        {
            "rule_group": "interim_extension_bucket",
            "rule_name": "offer_now",
            "logic": "Requires a reliable market band, strong upside support, strong comp support, scarce or selective-premium role context, and no major ambiguity or anchor-width warning.",
            "effect": "Use the current band actively: open inside the band, target fair value, and keep the walk-away max available.",
        },
        {
            "rule_group": "interim_extension_bucket",
            "rule_name": "offer_now_disciplined_band",
            "logic": "Extension-worthy now, but evidence is not strong enough to justify paying all the way to the ceiling yet.",
            "effect": "Open at the protected side, aim around fair value, and avoid chasing the walk-away max early.",
        },
        {
            "rule_group": "interim_extension_bucket",
            "rule_name": "wait_and_save_flexibility",
            "logic": "Current evidence supports keeping the player in range, but the timing value of waiting is still meaningful.",
            "effect": "Use only the protected price as an early anchor and preserve optionality for later information.",
        },
        {
            "rule_group": "interim_extension_bucket",
            "rule_name": "avoid_overcommitting",
            "logic": "Comp support, market anchor quality, or forecast context do not justify a meaningful early commitment.",
            "effect": "Do not initiate aggressive extension talks off current evidence.",
        },
    ])


def build_markdown(table: pd.DataFrame, summary: pd.DataFrame, case_table: pd.DataFrame, metadata: dict) -> str:
    lines = [
        "# Deliverable 3 Interim Extension Guidance",
        "",
        "This export assembles the forecast-adjusted **interim** extension-guidance layer from the Phase 7 reconciliation table.",
        "The status remains **interim_with_current_D1_macro_forecast** and must not be treated as the final extension recommendation until durability and richer uncertainty outputs are integrated.",
        "",
        "## Inputs",
        "",
        f"- `deliverable3_salary_reconciliation_table.csv`: `{metadata['inputs']['reconciliation_table']}`",
        "",
        "## Case-study snippets",
        "",
    ]

    for _, row in case_table.iterrows():
        lines.extend([
            f"### {row['PLAYER_NAME']} — {row['interim_extension_label']}",
            "",
            f"- **Player:** {row['PLAYER_NAME']}",
            f"- **Draft year:** {int(row['draft_year']) if not pd.isna(row['draft_year']) else 'N/A'}",
            f"- **Hybrid archetype:** {row.get('hybrid_archetype_label', 'N/A')}",
            f"- **Interim stance:** {row['interim_extension_label']}",
            f"- **Market band:** {row.get('market_anchor_band_text', 'Comp-based market anchor not supported.')}",
            f"- **Interim posture:** Open around {row.get('interim_open_pct_text', 'N/A')}, target around {row.get('interim_target_pct_text', 'N/A')}, hard max around {row.get('interim_hard_max_pct_text', 'N/A')}.",
            f"- **Forecast support:** {row.get('forecast_support_tier', 'N/A')}",
            f"- **Why:** {row.get('interim_extension_reason', '')}",
            f"- **Status note:** {row.get('interim_status_text', '')}",
            "",
        ])

    lines.extend([
        "## Summary metrics",
        "",
        "| Metric group | Metric | Value | Notes |",
        "|---|---|---:|---|",
    ])

    for _, r in summary.iterrows():
        lines.append(f"| {r['metric_group']} | {r['metric']} | {r['value']} | {r['notes']} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 8 — build interim forecast-adjusted extension guidance.")
    parser.add_argument("--project-root", default=".", help="Project root containing Output/, src/, and source CSVs.")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_dir = get_output_dir(project_root)
    log_path = output_dir / "salary_decision_support_workflow_log.txt"

    reconciliation_path = find_existing_path(project_root, [
        "Output/Salary%20Decision%20Support/deliverable3_salary_reconciliation_table.csv",
        "Output/Salary Decision Support/deliverable3_salary_reconciliation_table.csv",
        "deliverable3_salary_reconciliation_table.csv",
    ])

    recon = pd.read_csv(reconciliation_path)
    table = build_inputs_table(recon)
    cards = build_cards(table)
    case_table = cards[cards["case_study_priority"] == "demo_holdout_priority"].copy()
    if case_table.empty:
        case_table = cards[cards["PLAYER_NAME"].isin(DEMO_PLAYERS)].copy()
    if case_table.empty:
        case_table = cards.head(2).copy()

    summary = build_summary(table)
    rule_reference = build_rule_reference()

    metadata = {
        "script": "08_build_interim_extension_guidance.py",
        "project_root": str(project_root),
        "inputs": {
            "reconciliation_table": str(reconciliation_path),
        },
        "outputs": [
            "deliverable3_interim_extension_guidance_table.csv",
            "deliverable3_interim_extension_guidance_cards.csv",
            "deliverable3_interim_extension_case_study_table.csv",
            "deliverable3_interim_extension_summary.csv",
            "deliverable3_interim_extension_summary.md",
            "deliverable3_interim_extension_rule_reference.csv",
        ],
        "notes": [
            "This phase upgrades the Block-2-only provisional stance into an interim forecast-adjusted guidance layer.",
            "The current Deliverable 1 macro forecast table is treated as a forecast overlay, not as the final fully locked Deliverable 1 handoff.",
            "Durability / availability risk and richer uncertainty outputs are still pending, so this phase must not be presented as final extension guidance.",
        ],
        "counts": {
            "players": int(len(table)),
            "forecast_overlay_rows": int(table["forecast_overlay_available_flag"].fillna(0).astype(int).sum()),
            "offer_now_rows": int((table["interim_extension_bucket"] == "offer_now").sum()),
            "demo_case_rows": int(len(case_table)),
        },
    }

    table.to_csv(output_dir / "deliverable3_interim_extension_guidance_table.csv", index=False)
    cards.to_csv(output_dir / "deliverable3_interim_extension_guidance_cards.csv", index=False)
    case_table.to_csv(output_dir / "deliverable3_interim_extension_case_study_table.csv", index=False)
    summary.to_csv(output_dir / "deliverable3_interim_extension_summary.csv", index=False)
    rule_reference.to_csv(output_dir / "deliverable3_interim_extension_rule_reference.csv", index=False)

    md_text = build_markdown(cards, summary, case_table, metadata)
    (output_dir / "deliverable3_interim_extension_summary.md").write_text(md_text, encoding="utf-8")
    (output_dir / "deliverable3_interim_extension_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    log_text = f"""
Phase 8 complete — interim forecast-adjusted extension guidance assembled.

Inputs used:
- reconciliation_table: {reconciliation_path}

Main outputs:
- deliverable3_interim_extension_guidance_table.csv
- deliverable3_interim_extension_guidance_cards.csv
- deliverable3_interim_extension_case_study_table.csv
- deliverable3_interim_extension_summary.csv
- deliverable3_interim_extension_rule_reference.csv

Workflow logic:
- treated the Phase 7 reconciliation table as the merge-ready bridge between salary anchors, the baseline salary model, and the current Deliverable 1 macro forecast overlay,
- converted the current Deliverable 1 overlay into conservative forecast-support tiers rather than pretending full uncertainty and durability outputs already exist,
- upgraded provisional Block-2-only stances into interim extension buckets with negotiation posture fields,
- kept Trae Young, Nikola Vučević, and Andrew Bynum reserved in the case-study export when present.

Important boundary:
- the resulting guidance remains interim_with_current_D1_macro_forecast and must not be presented as the final extension recommendation.
""".strip()
    append_workflow_log(log_path, log_text)


if __name__ == "__main__":
    main()
