from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from datetime import datetime
from difflib import get_close_matches

import numpy as np
import pandas as pd


UPPER_CLIP = 0.35
DEMO_PLAYERS = ["Trae Young", "Nikola Vucevic", "Nikola Vučević"]
ALIAS_NORMALIZED = {
    "nikolavuevi": "nikolavucevic",
    "nikolajoki": "nikolajokic",
}


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


def find_existing_path(project_root: Path, candidates: list[str], required: bool = True) -> Path | None:
    for candidate in candidates:
        path = project_root / candidate
        if path.exists():
            return path
    for candidate in candidates:
        basename = Path(candidate).name
        matches = sorted(project_root.rglob(basename), key=lambda p: (len(p.parts), str(p)))
        if matches:
            return matches[0]
    if required:
        raise FileNotFoundError(f"Could not locate any of: {candidates}")
    return None


def normalize_name(name: object) -> str:
    if pd.isna(name):
        return ""
    text = str(name).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", "", text)
    text = ALIAS_NORMALIZED.get(text, text)
    return text


def load_inputs(project_root: Path) -> dict[str, Path | None]:
    return {
        "baseline_predictions": find_existing_path(project_root, [
            "Output/Salary%20Decision%20Support/year5_salary_baseline_selected_model_predictions.csv",
            "Output/Salary Decision Support/year5_salary_baseline_selected_model_predictions.csv",
            "year5_salary_baseline_selected_model_predictions.csv",
        ]),
        "market_anchor_band": find_existing_path(project_root, [
            "Output/Salary%20Decision%20Support/salary_market_anchor_band_table.csv",
            "Output/Salary Decision Support/salary_market_anchor_band_table.csv",
            "salary_market_anchor_band_table.csv",
        ]),
        "decision_inputs": find_existing_path(project_root, [
            "Output/Salary%20Decision%20Support/salary_provisional_decision_inputs_table.csv",
            "Output/Salary Decision Support/salary_provisional_decision_inputs_table.csv",
            "salary_provisional_decision_inputs_table.csv",
        ]),
        "forecast_macro": find_existing_path(project_root, [
            "final_lstm_predictions_all_players_with_class_and_performance.csv",
            "Output/Player Archetype Analysis/final_lstm_predictions_all_players_with_class_and_performance.csv",
        ], required=False),
    }


def apply_alignment(row: pd.Series) -> str:
    pred = row.get("predicted_year5_salary_cap_pct")
    protected = row.get("protected_price_cap_pct")
    fair = row.get("fair_price_cap_pct")
    walk = row.get("walk_away_max_cap_pct")
    market_ok = row.get("market_anchor_supported_flag", 0)
    if pd.isna(pred):
        return "missing_model"
    if not market_ok or pd.isna(protected) or pd.isna(fair) or pd.isna(walk):
        return "model_only_no_market_anchor"
    if pred < protected:
        return "below_protected"
    if pred <= fair:
        return "between_protected_and_fair"
    if pred <= walk:
        return "between_fair_and_walkaway"
    return "above_walkaway"



def anchor_preference(row: pd.Series) -> str:
    pred = row.get("predicted_year5_salary_cap_pct")
    market_ok = row.get("market_anchor_supported_flag", 0)
    support = str(row.get("comp_support_bucket", "unknown"))
    align = row.get("model_market_alignment_bucket", "unknown")
    if pd.isna(pred) and market_ok:
        return "market_anchor_only"
    if not market_ok and not pd.isna(pred):
        return "baseline_model_only"
    if not market_ok and pd.isna(pred):
        return "insufficient_pricing_support"
    if align in {"between_protected_and_fair", "between_fair_and_walkaway"} and support in {"strong", "moderate"}:
        return "market_band_confirmed_by_model"
    if align == "below_protected":
        return "model_discount_vs_market_review"
    if align == "above_walkaway":
        return "model_premium_vs_market_review"
    return "mixed_review"



def derive_forecast_fields(forecast_df: pd.DataFrame) -> pd.DataFrame:
    f = forecast_df.copy()
    f["forecast_name_key"] = f["PLAYER_NAME"].map(normalize_name)
    f["forecast_outlook_bucket"] = f["predicted_class"].map({
        "Sleeper": "upside_leaning",
        "Neutral": "neutral_leaning",
        "Bust": "downside_leaning",
    }).fillna("unknown")

    f["forecast_confidence_proxy"] = np.select(
        [f["max_pred_prob"] >= 0.80, f["max_pred_prob"] >= 0.60],
        ["high_class_confidence", "medium_class_confidence"],
        default="low_class_confidence",
    )

    q = f["predicted_performance_score_per_game_5to7_avg"].quantile([0.25, 0.75]).to_dict()
    q25 = float(q.get(0.25, f["predicted_performance_score_per_game_5to7_avg"].median()))
    q75 = float(q.get(0.75, f["predicted_performance_score_per_game_5to7_avg"].median()))

    def perf_tier(x: float) -> str:
        if pd.isna(x):
            return "unknown"
        if x <= q25:
            return "lower_forecast_tier"
        if x >= q75:
            return "upper_forecast_tier"
        return "middle_forecast_tier"

    f["forecast_performance_tier"] = f["predicted_performance_score_per_game_5to7_avg"].map(perf_tier)
    return f



def merge_forecast_overlay(core: pd.DataFrame, forecast_path: Path | None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if forecast_path is None or not forecast_path.exists():
        core = core.copy()
        core["forecast_overlay_available_flag"] = 0
        core["forecast_merge_method"] = "forecast_file_missing"
        core["forecast_outlook_bucket"] = "forecast_unavailable"
        core["forecast_confidence_proxy"] = "forecast_unavailable"
        core["forecast_performance_tier"] = "forecast_unavailable"
        core["predicted_class"] = np.nan
        core["prob_Bust"] = np.nan
        core["prob_Neutral"] = np.nan
        core["prob_Sleeper"] = np.nan
        core["predicted_performance_score_per_game_5to7_avg"] = np.nan
        audit = pd.DataFrame([{
            "forecast_merge_method": "forecast_file_missing",
            "players": int(len(core)),
            "notes": "No Deliverable 1 macro forecast table was available for overlay.",
        }])
        metadata = {"forecast_overlay_used": False, "forecast_rows": 0, "matched_players": 0}
        return core, audit, metadata

    f = derive_forecast_fields(pd.read_csv(forecast_path))
    core = core.copy()
    core["forecast_name_key"] = core["PLAYER_NAME"].map(normalize_name)

    exact = core.merge(
        f.drop_duplicates("forecast_name_key"),
        on="forecast_name_key",
        how="left",
        suffixes=("", "_forecast"),
    )
    exact["forecast_merge_method"] = np.where(exact["predicted_class"].notna(), "exact_normalized_name", "unmatched_after_exact")

    unmatched_mask = exact["predicted_class"].isna()
    first_name_groups: dict[str, list[str]] = {}
    for key in f["forecast_name_key"].dropna().unique().tolist():
        first = re.match(r"[a-z]+", key)
        prefix = first.group(0) if first else key[:5]
        first_name_groups.setdefault(prefix, []).append(key)

    fuzzy_rows = []
    for idx, row in exact.loc[unmatched_mask, ["forecast_name_key", "PLAYER_NAME"]].iterrows():
        key = row["forecast_name_key"]
        first = re.match(r"[a-z]+", key)
        prefix = first.group(0) if first else key[:5]
        candidates = first_name_groups.get(prefix, [])
        match = get_close_matches(key, candidates, n=1, cutoff=0.88)
        fuzzy_rows.append((idx, match[0] if match else None))

    if fuzzy_rows:
        f_indexed = f.set_index("forecast_name_key")
        for idx, matched_key in fuzzy_rows:
            if matched_key is None:
                continue
            rec = f_indexed.loc[matched_key]
            if isinstance(rec, pd.DataFrame):
                rec = rec.iloc[0]
            for col in [
                "PLAYER_NAME_forecast", "actual_class", "predicted_class", "prob_Bust", "prob_Neutral",
                "prob_Sleeper", "actual_performance_score_per_game_5to7_avg",
                "predicted_performance_score_per_game_5to7_avg", "performance_pred_error", "label_known",
                "is_correct", "max_pred_prob", "forecast_outlook_bucket", "forecast_confidence_proxy",
                "forecast_performance_tier"
            ]:
                if col in exact.columns:
                    exact.at[idx, col] = rec.get(col.replace("_forecast", ""), rec.get(col, np.nan))
                else:
                    exact.at[idx, col] = rec.get(col.replace("_forecast", ""), rec.get(col, np.nan))
            exact.at[idx, "forecast_merge_method"] = "fuzzy_same_first_name"

    exact["forecast_overlay_available_flag"] = exact["predicted_class"].notna().astype(int)
    exact["forecast_outlook_bucket"] = exact["forecast_outlook_bucket"].fillna("forecast_unavailable")
    exact["forecast_confidence_proxy"] = exact["forecast_confidence_proxy"].fillna("forecast_unavailable")
    exact["forecast_performance_tier"] = exact["forecast_performance_tier"].fillna("forecast_unavailable")

    audit = exact["forecast_merge_method"].value_counts(dropna=False).rename_axis("forecast_merge_method").reset_index(name="players")
    audit["notes"] = audit["forecast_merge_method"].map({
        "exact_normalized_name": "Direct normalized name match between salary backbone and Deliverable 1 macro forecast table.",
        "fuzzy_same_first_name": "Controlled fallback merge using close-match search within the same first-name bucket; review if used heavily.",
        "unmatched_after_exact": "No forecast row was available after exact or controlled fallback matching.",
    }).fillna("")

    metadata = {
        "forecast_overlay_used": True,
        "forecast_rows": int(len(f)),
        "matched_players": int(exact["forecast_overlay_available_flag"].sum()),
        "exact_matches": int((exact["forecast_merge_method"] == "exact_normalized_name").sum()),
        "fuzzy_matches": int((exact["forecast_merge_method"] == "fuzzy_same_first_name").sum()),
        "unmatched_players": int((exact["forecast_merge_method"] == "unmatched_after_exact").sum()),
    }
    return exact, audit, metadata



def add_reconciliation_logic(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["predicted_year5_salary_cap_pct"] = out["predicted_year5_salary_cap_pct"].clip(lower=0.0, upper=UPPER_CLIP)
    out["baseline_vs_protected_gap"] = out["predicted_year5_salary_cap_pct"] - out["protected_price_cap_pct"]
    out["baseline_vs_fair_gap"] = out["predicted_year5_salary_cap_pct"] - out["fair_price_cap_pct"]
    out["baseline_vs_walkaway_gap"] = out["predicted_year5_salary_cap_pct"] - out["walk_away_max_cap_pct"]
    out["baseline_vs_fair_gap_abs"] = out["baseline_vs_fair_gap"].abs()
    out["baseline_vs_fair_gap_pct"] = out["baseline_vs_fair_gap_abs"] / out["fair_price_cap_pct"]
    out.loc[out["fair_price_cap_pct"].isna() | (out["fair_price_cap_pct"] == 0), "baseline_vs_fair_gap_pct"] = np.nan
    out["model_market_alignment_bucket"] = out.apply(apply_alignment, axis=1)
    out["pricing_anchor_preference"] = out.apply(anchor_preference, axis=1)
    out["market_model_midpoint_cap_pct"] = out[["fair_price_cap_pct", "predicted_year5_salary_cap_pct"]].mean(axis=1)
    out.loc[out[["fair_price_cap_pct", "predicted_year5_salary_cap_pct"]].isna().all(axis=1), "market_model_midpoint_cap_pct"] = np.nan

    def overlay_signal(row: pd.Series) -> str:
        outlook = row.get("forecast_outlook_bucket", "forecast_unavailable")
        conf = row.get("forecast_confidence_proxy", "forecast_unavailable")
        align = row.get("model_market_alignment_bucket", "unknown")
        if outlook == "forecast_unavailable":
            return "no_forecast_overlay"
        if align == "above_walkaway" and outlook != "upside_leaning":
            return "premium_not_supported_by_forecast"
        if align == "below_protected" and outlook == "downside_leaning":
            return "discount_supported_by_forecast"
        if align in {"between_protected_and_fair", "between_fair_and_walkaway"} and outlook == "upside_leaning":
            return "within_band_with_upside_support"
        if align in {"between_protected_and_fair", "between_fair_and_walkaway"} and outlook == "neutral_leaning":
            return "within_band_neutral_forecast"
        if align == "model_only_no_market_anchor" and outlook != "forecast_unavailable":
            return "model_plus_forecast_without_market_anchor"
        if outlook == "downside_leaning" and conf == "high_class_confidence":
            return "forecast_downside_caution"
        return "mixed_forecast_signal"

    out["forecast_overlay_signal"] = out.apply(overlay_signal, axis=1)

    def interim_note(row: pd.Series) -> str:
        pref = row.get("pricing_anchor_preference", "unknown")
        forecast_signal = row.get("forecast_overlay_signal", "no_forecast_overlay")
        if pref == "market_band_confirmed_by_model":
            base = "Market anchor and baseline model are directionally aligned."
        elif pref == "model_discount_vs_market_review":
            base = "Baseline model sits below the protected market price; review whether comps are too optimistic."
        elif pref == "model_premium_vs_market_review":
            base = "Baseline model sits above the walk-away max; review whether the market anchor is too conservative or the model is overstating value."
        elif pref == "baseline_model_only":
            base = "Market anchor is missing, so the baseline salary model is the only quantitative price reference."
        elif pref == "market_anchor_only":
            base = "Baseline model support is missing, so the comp-market anchor remains the main price reference."
        else:
            base = "Market and model evidence should be reviewed together before any pricing stance is hardened."

        extra = {
            "within_band_with_upside_support": "Deliverable 1 macro forecast leans positive, which supports defending at least the existing band rather than discounting it.",
            "within_band_neutral_forecast": "Deliverable 1 macro forecast is neutral, so the existing band remains a disciplined starting point.",
            "premium_not_supported_by_forecast": "The available Deliverable 1 macro forecast does not support chasing a premium above the current walk-away band.",
            "discount_supported_by_forecast": "The available Deliverable 1 macro forecast reinforces a discount-oriented negotiation posture.",
            "model_plus_forecast_without_market_anchor": "Forecast context helps, but missing comp-market support means this still is not a final recommendation.",
            "forecast_downside_caution": "Forecast classification adds extra caution, though uncertainty and durability are still missing.",
            "mixed_forecast_signal": "Forecast context is mixed and should not overrule the current salary board by itself.",
            "no_forecast_overlay": "No Deliverable 1 macro forecast overlay was available here.",
        }.get(forecast_signal, "")
        return (base + " " + extra).strip()

    out["interim_reconciliation_note"] = out.apply(interim_note, axis=1)
    out["reconciliation_status"] = np.where(
        out["forecast_overlay_available_flag"].eq(1),
        "interim_with_macro_forecast_overlay",
        "interim_without_forecast_overlay",
    )
    return out



def build_summary(df: pd.DataFrame, forecast_meta: dict) -> pd.DataFrame:
    rows = [
        {
            "metric_group": "workflow_status",
            "metric": "players_total",
            "value": int(len(df)),
            "notes": "Total player-level rows in the Phase 7 reconciliation layer.",
        },
        {
            "metric_group": "workflow_status",
            "metric": "market_anchor_supported",
            "value": int(df["market_anchor_supported_flag"].fillna(0).sum()),
            "notes": "Players with a supported comp-market anchor band carried forward from Phase 2.",
        },
        {
            "metric_group": "workflow_status",
            "metric": "forecast_overlay_matched_players",
            "value": int(forecast_meta.get("matched_players", 0)),
            "notes": "Players successfully matched to the Deliverable 1 macro forecast table.",
        },
    ]
    for col, group in [
        ("model_market_alignment_bucket", "alignment_bucket"),
        ("pricing_anchor_preference", "pricing_anchor_preference"),
        ("forecast_outlook_bucket", "forecast_outlook_bucket"),
        ("forecast_overlay_signal", "forecast_overlay_signal"),
    ]:
        vc = df[col].fillna("missing").value_counts(dropna=False)
        for key, val in vc.items():
            rows.append({
                "metric_group": group,
                "metric": str(key),
                "value": int(val),
                "notes": f"Phase 7 count for {group.replace('_', ' ')}.",
            })
    return pd.DataFrame(rows)



def write_markdown(path: Path, summary_df: pd.DataFrame, case_df: pd.DataFrame, forecast_meta: dict):
    lines = [
        "# Deliverable 3 Salary Reconciliation Layer",
        "",
        "This phase reconciles the baseline salary model with the comp-market anchor band and optionally overlays the current Deliverable 1 macro forecast table.",
        "",
        "Important boundary: the output remains **interim**. It is not the final extension recommendation because full Deliverable 1 uncertainty, durability, and merge-safe identifiers are still not fully integrated.",
        "",
        "## Summary metrics",
        "",
        summary_df.to_markdown(index=False),
    ]
    if case_df is not None and not case_df.empty:
        cols = [c for c in [
            "PLAYER_NAME", "draft_year", "decision_card_label", "decision_card_band_text",
            "predicted_year5_salary_cap_pct", "predicted_class", "max_pred_prob",
            "pricing_anchor_preference", "forecast_overlay_signal"
        ] if c in case_df.columns]
        lines += ["", "## Case-study extract", "", case_df[cols].to_markdown(index=False)]
    lines += [
        "",
        "## Forecast overlay note",
        "",
        f"Forecast overlay matched players: **{forecast_meta.get('matched_players', 0)}** out of the salary backbone.",
        "The currently available Deliverable 1 macro table is useful as an overlay because it contains class probabilities and a predicted performance score, but it still should not be treated as full final Deliverable 1 integration by itself.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def main():
    parser = argparse.ArgumentParser(description="Phase 7 — build salary-model versus market reconciliation layer.")
    parser.add_argument("--project-root", type=str, default=".")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_dir = get_output_dir(project_root)
    log_path = out_dir / "salary_decision_support_workflow_log.txt"
    paths = load_inputs(project_root)

    baseline = pd.read_csv(paths["baseline_predictions"])
    market = pd.read_csv(paths["market_anchor_band"])
    decision = pd.read_csv(paths["decision_inputs"])

    keep_market = [
        "PLAYER_ID", "protected_price_cap_pct", "fair_price_cap_pct", "walk_away_max_cap_pct",
        "anchor_band_width", "anchor_band_relative_width", "market_anchor_supported_flag",
        "market_anchor_band_text", "historical_year5_salary_cap_pct_observed", "historical_year5_salary_observed_flag",
    ]
    keep_decision = [
        "PLAYER_ID", "PLAYER_NAME", "draft_year", "macro_archetype", "shot_style_subtype",
        "hybrid_archetype_label", "scarcity_tier", "scarcity_wording", "prototype_fit_ambiguity",
        "ambiguity_band", "identity_drift_class", "drift_signal", "realistic_comp_list",
        "realistic_comp_similarity_mean", "comp_salary_match_count", "comp_salary_match_rate",
        "comp_salary_anchor_support", "detail_effective_comp_n", "detail_same_macro_share",
        "detail_anchor_std_weighted", "comp_support_bucket", "anchor_width_band", "decision_card_band_text",
        "decision_card_support_text", "provisional_action_bucket", "provisional_action_reason",
        "decision_card_label", "decision_status", "supporting_shot_style_explanation",
        "block2_market_context_interpretation", "holdout_for_case_study", "holdout_reason",
    ]
    keep_baseline = [
        "PLAYER_ID", "PLAYER_NAME", "draft_year", "year5_salary_cap_pct", "target_observed_flag",
        "training_eligible_flag", "holdout_for_case_study", "holdout_reason", "split_group_draft_year",
        "modeling_row_status", "baseline_selected_model", "predicted_year5_salary_cap_pct", "prediction_status",
    ]

    core = baseline[keep_baseline].merge(market[keep_market], on="PLAYER_ID", how="left")
    core = core.merge(decision[keep_decision], on=["PLAYER_ID", "PLAYER_NAME", "draft_year"], how="left", suffixes=("", "_decision"))

    core, merge_audit, forecast_meta = merge_forecast_overlay(core, paths["forecast_macro"])
    recon = add_reconciliation_logic(core)

    summary = build_summary(recon, forecast_meta)
    forecast_support = recon["forecast_overlay_available_flag"].value_counts(dropna=False).rename_axis("forecast_overlay_available_flag").reset_index(name="players")
    case_table = recon[recon["holdout_for_case_study"].fillna(0).astype(int).eq(1)].copy()
    if case_table.empty:
        demo_keys = {normalize_name(n) for n in DEMO_PLAYERS}
        case_table = recon[recon["PLAYER_NAME"].map(normalize_name).isin(demo_keys)].copy()

    out_table = out_dir / "deliverable3_salary_reconciliation_table.csv"
    out_summary = out_dir / "deliverable3_salary_reconciliation_summary.csv"
    out_summary_md = out_dir / "deliverable3_salary_reconciliation_summary.md"
    out_merge_audit = out_dir / "deliverable3_forecast_overlay_merge_audit.csv"
    out_forecast_support = out_dir / "deliverable3_forecast_overlay_support_breakdown.csv"
    out_case = out_dir / "deliverable3_salary_reconciliation_case_study_table.csv"
    out_meta = out_dir / "deliverable3_salary_reconciliation_metadata.json"

    recon.to_csv(out_table, index=False)
    summary.to_csv(out_summary, index=False)
    merge_audit.to_csv(out_merge_audit, index=False)
    forecast_support.to_csv(out_forecast_support, index=False)
    case_table.to_csv(out_case, index=False)
    write_markdown(out_summary_md, summary, case_table, forecast_meta)

    meta = {
        "script": "07_build_salary_reconciliation_layer.py",
        "project_root": str(project_root),
        "inputs": {k: (str(v) if v is not None else None) for k, v in paths.items()},
        "outputs": [
            out_table.name,
            out_summary.name,
            out_summary_md.name,
            out_merge_audit.name,
            out_forecast_support.name,
            out_case.name,
        ],
        "forecast_overlay": forecast_meta,
        "notes": [
            "This phase reconciles the baseline salary model with the comp-market anchor band.",
            "The Deliverable 1 macro forecast table is used only as an interim overlay when available.",
            "The resulting layer is not the final extension recommendation because durability and explicit uncertainty are still missing.",
        ],
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log_text = f"""
Phase 7 complete — salary model, market anchor, and macro forecast overlay reconciled.

Inputs used:
- baseline_predictions: {paths['baseline_predictions']}
- market_anchor_band: {paths['market_anchor_band']}
- decision_inputs: {paths['decision_inputs']}
- forecast_macro: {paths['forecast_macro']}

Main outputs:
- {out_table.name}
- {out_summary.name}
- {out_merge_audit.name}
- {out_case.name}

Workflow logic:
- reconciled the baseline Year-5 salary model against the protected / fair / walk-away comp-market band,
- classified alignment buckets such as below_protected, within_band, and above_walkaway,
- created pricing-anchor preference flags for market-only, model-only, and review cases,
- used the current Deliverable 1 macro forecast table only as an interim overlay when a merge-safe match was available,
- preserved Trae Young and Nikola Vučević as case-study rows when available.

Important boundary:
- the resulting table remains interim and should not be presented as final extension guidance because uncertainty, durability, and stronger player-keyed Deliverable 1 integration are still pending.
""".strip()
    append_workflow_log(log_path, log_text)


if __name__ == "__main__":
    main()
