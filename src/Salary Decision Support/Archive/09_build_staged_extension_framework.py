from __future__ import annotations

import argparse
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DEMO_PLAYERS = ["Trae Young", "Nikola Vučević", "Nikola Vucevic"]


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


def normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


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


def append_workflow_log(log_path: Path, text_block: str) -> None:
    ensure_dir(log_path.parent)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n{'=' * 80}\n")
        fh.write(f"[{timestamp}]\n")
        fh.write(text_block.rstrip() + "\n")


def pct_text(x: object) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{float(x) * 100:.1f}%"


def safe_float(x: object) -> float:
    if pd.isna(x):
        return np.nan
    return float(x)


def confidence_band(score: object) -> str:
    x = safe_float(score)
    if pd.isna(x):
        return "unknown_confidence"
    if x >= 0.85:
        return "high_confidence"
    if x >= 0.65:
        return "medium_confidence"
    return "low_confidence"


def uncertainty_band(x: object) -> str:
    z = safe_float(x)
    if pd.isna(z):
        return "unknown_uncertainty"
    if z <= 0.15:
        return "low_uncertainty"
    if z <= 0.35:
        return "medium_uncertainty"
    return "high_uncertainty"


def outlook_bucket(pred_class: str, sleeper: object, neutral: object, bust: object) -> str:
    pred = str(pred_class or "").strip().lower()
    s = safe_float(sleeper)
    b = safe_float(bust)
    if pred == "sleeper":
        return "upside_leaning"
    if pred == "bust":
        return "downside_leaning"
    if pred == "neutral":
        return "neutral_leaning"
    if not pd.isna(s) and not pd.isna(b):
        if s - b >= 0.20:
            return "upside_leaning"
        if b - s >= 0.20:
            return "downside_leaning"
    return "mixed_or_unknown"


def performance_tier(pred_perf: object, q33: float, q67: float) -> str:
    x = safe_float(pred_perf)
    if pd.isna(x):
        return "unknown_perf_tier"
    if x >= q67:
        return "upper_forecast_tier"
    if x >= q33:
        return "middle_forecast_tier"
    return "lower_forecast_tier"


def adjustment_mode(row: pd.Series) -> str:
    support = str(row.get("forecast_support_status", ""))
    if support != "test_prediction_available":
        if support == "classification_or_truth_only":
            return "classification_context_only"
        return "block2_only"
    outlook = str(row.get("forecast_outlook_bucket", ""))
    conf = str(row.get("forecast_confidence_band", ""))
    uncert = str(row.get("forecast_uncertainty_band", ""))
    perf = str(row.get("forecast_performance_tier", ""))
    if outlook == "upside_leaning" and conf in {"high_confidence", "medium_confidence"} and uncert != "high_uncertainty":
        if perf == "upper_forecast_tier":
            return "upside_push_within_band"
        return "mild_upside_push"
    if outlook == "downside_leaning" and conf in {"high_confidence", "medium_confidence"}:
        return "downside_caution_within_band"
    if outlook == "neutral_leaning" and uncert == "low_uncertainty":
        return "hold_band_with_neutral_support"
    return "hold_band"


def adjusted_stance(row: pd.Series) -> str:
    base = str(row.get("provisional_action_bucket", "wait_and_save_flexibility"))
    mode = str(row.get("forecast_adjustment_mode", "block2_only"))
    comp = str(row.get("comp_support_bucket", ""))
    scarcity = str(row.get("scarcity_tier", ""))
    ambiguity = str(row.get("ambiguity_band", ""))
    width = str(row.get("anchor_width_band", ""))
    if mode in {"block2_only", "classification_context_only", "hold_band", "hold_band_with_neutral_support"}:
        return base
    if mode == "upside_push_within_band":
        if base == "offer_now_disciplined_band" and comp == "strong" and scarcity in {"scarce", "selective_premium"} and ambiguity != "high_ambiguity" and width in {"narrow", "balanced"}:
            return "offer_now"
        if base == "wait_and_save_flexibility" and comp in {"strong", "moderate"} and scarcity in {"scarce", "selective_premium", "replaceable_middle"}:
            return "offer_now_disciplined_band"
        return base
    if mode == "mild_upside_push":
        if base == "wait_and_save_flexibility" and comp == "strong" and scarcity in {"scarce", "selective_premium"}:
            return "offer_now_disciplined_band"
        return base
    if mode == "downside_caution_within_band":
        if base == "offer_now":
            return "offer_now_disciplined_band"
        if base == "offer_now_disciplined_band":
            return "wait_and_save_flexibility"
        if base == "wait_and_save_flexibility" and comp in {"weak", "insufficient"}:
            return "avoid_overcommitting"
        return base
    return base


def negotiation_values(row: pd.Series) -> tuple[float, float, float]:
    p = safe_float(row.get("protected_price_cap_pct"))
    f = safe_float(row.get("fair_price_cap_pct"))
    w = safe_float(row.get("walk_away_max_cap_pct"))
    if pd.isna(p) and pd.isna(f) and pd.isna(w):
        return (np.nan, np.nan, np.nan)
    if pd.isna(p):
        p = f if not pd.isna(f) else w
    if pd.isna(f):
        f = p if not pd.isna(p) else w
    if pd.isna(w):
        w = f if not pd.isna(f) else p
    stance = str(row.get("staged_extension_stance", "wait_and_save_flexibility"))
    mode = str(row.get("forecast_adjustment_mode", "block2_only"))
    if stance == "offer_now":
        open_x = max(p, f - 0.25 * (w - p))
        target_x = f
        hard_x = w
    elif stance == "offer_now_disciplined_band":
        open_x = p + 0.40 * (f - p)
        target_x = f
        hard_x = w
    elif stance == "wait_and_save_flexibility":
        open_x = p
        target_x = p + 0.50 * (f - p)
        hard_x = f
    else:
        open_x = p
        target_x = p
        hard_x = min(f, p + 0.25 * max(w - p, 0))
    if mode == "downside_caution_within_band":
        target_x = min(target_x, f)
        hard_x = min(hard_x, f)
    return (float(open_x), float(target_x), float(hard_x))


def stage_status(row: pd.Series) -> str:
    if int(row.get("supported_forecast_adjustment_flag", 0)) == 1:
        return "forecast_adjusted_supported_subset"
    if int(row.get("forecast_probabilities_available_flag", 0)) == 1:
        return "block1_classification_overlay_only"
    return "block2_only_provisional"


def reason_text(row: pd.Series) -> str:
    stance = str(row.get("staged_extension_stance", ""))
    comp = str(row.get("comp_support_bucket", ""))
    scarcity = str(row.get("scarcity_tier", ""))
    mode = str(row.get("forecast_adjustment_mode", ""))
    conf = str(row.get("forecast_confidence_band", ""))
    uncert = str(row.get("forecast_uncertainty_band", ""))
    base = str(row.get("provisional_action_bucket", ""))
    if mode == "block2_only":
        return f"Remain {base}: no official Block 1 forecast support was merged, so the stance stays anchored in comp-market evidence only."
    if mode == "classification_context_only":
        return "Keep the Block 2 posture: class probabilities are available, but no supported performance forecast is available for this row, so forecast use stays descriptive rather than action-changing."
    if mode == "upside_push_within_band":
        return f"Forecast support is positive and reasonably confident, so the posture is allowed to move one step more aggressive within the existing market band; comp support remains {comp} and scarcity remains {scarcity}."
    if mode == "mild_upside_push":
        return "Forecast support leans positive, but not strongly enough to expand the band; the posture can become slightly more proactive while staying disciplined."
    if mode == "downside_caution_within_band":
        return f"Forecast support leans negative with {conf} and {uncert}, so the stance is pulled one step more cautious inside the existing band rather than discounted beyond it."
    if mode in {"hold_band", "hold_band_with_neutral_support"}:
        return "Forecast support is neutral or mixed, so the existing market band is retained without additional premium or discount."
    return f"Staged guidance remains {stance}."


def build_card_text(row: pd.Series) -> dict[str, str]:
    title = f"{row['PLAYER_NAME']} — {row.get('staged_extension_stance_label','')}"
    market = (
        f"Protected {pct_text(row.get('protected_price_cap_pct'))}, fair {pct_text(row.get('fair_price_cap_pct'))}, walk-away {pct_text(row.get('walk_away_max_cap_pct'))}. "
        f"Open at {pct_text(row.get('staged_open_cap_pct'))}, target {pct_text(row.get('staged_target_cap_pct'))}, hard max {pct_text(row.get('staged_hard_max_cap_pct'))}."
    )
    forecast = (
        f"Forecast support status: {row.get('forecast_support_status','none')}. "
        f"Outlook {row.get('forecast_outlook_bucket','unknown')}, confidence {row.get('forecast_confidence_band','unknown')}, uncertainty {row.get('forecast_uncertainty_band','unknown')}."
    )
    status = (
        f"Status: {row.get('deliverable3_framework_status','')}. "
        f"This is a staged decision-support output, not a fully validated final extension recommendation for the whole cohort."
    )
    return {
        "card_title": title,
        "card_market_text": market,
        "card_forecast_text": forecast,
        "card_reason_text": row.get("staged_extension_reason", ""),
        "card_status_text": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build staged final Deliverable 3 extension framework")
    parser.add_argument("--project-root", default=".", help="Project root path")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_dir = get_output_dir(project_root)

    reconciliation_path = find_existing_path(project_root, [
        "Output/Salary Decision Support/deliverable3_salary_reconciliation_table.csv",
        "Output/Salary%20Decision%20Support/deliverable3_salary_reconciliation_table.csv",
        "deliverable3_salary_reconciliation_table.csv",
    ])
    block1_handoff_path = find_existing_path(project_root, [
        "Output/Salary Decision Support/deliverable3_block1_forecast_handoff.csv",
        "Output/Salary%20Decision%20Support/deliverable3_block1_forecast_handoff.csv",
        "deliverable3_block1_forecast_handoff.csv",
    ])
    case_study_input_path = find_existing_path(project_root, [
        "Output/Salary Decision Support/deliverable3_block1_case_study_examples.csv",
        "Output/Salary%20Decision%20Support/deliverable3_block1_case_study_examples.csv",
        "deliverable3_block1_case_study_examples.csv",
    ])

    recon = pd.read_csv(reconciliation_path)
    block1 = pd.read_csv(block1_handoff_path)
    block1_case = pd.read_csv(case_study_input_path)

    recon["player_name_key"] = recon["PLAYER_NAME"].map(normalize_name)
    block1["player_name_key"] = block1["PLAYER_NAME"].map(normalize_name)

    keep_block1 = [
        "PLAYER_ID", "PLAYER_NAME", "player_name_key", "draft_year", "predicted_class", "actual_class",
        "label_known", "is_correct", "sleeper_probability", "neutral_probability", "bust_probability",
        "confidence_rating", "confidence_score", "uncertainty_proxy", "predicted_years5_7_perf_score_component_mean",
        "actual_years5_7_perf_score_component_mean", "performance_pred_error_component_mean",
        "actual_years5_7_games_avg", "actual_years5_7_minutes_avg", "actual_years5_7_points_avg",
        "actual_years5_7_rebounds_avg", "actual_years5_7_assists_avg", "performance_test_prediction_available",
        "forecast_support_flag", "classification_support_flag", "block1_ready_flag", "deliverable3_block1_note",
    ]
    block1 = block1[keep_block1].copy()

    # Drop any stale forecast-side fields inherited from earlier reconciliation layers.
    # The refreshed Block 1 handoff must be authoritative for all forecast-related columns
    # in the final Deliverable 3 assembly.
    stale_block1_fields = [
        "predicted_class",
        "actual_class",
        "label_known",
        "is_correct",
        "sleeper_probability",
        "neutral_probability",
        "bust_probability",
        "confidence_rating",
        "confidence_score",
        "uncertainty_proxy",
        "predicted_years5_7_perf_score_component_mean",
        "actual_years5_7_perf_score_component_mean",
        "performance_pred_error_component_mean",
        "actual_years5_7_games_avg",
        "actual_years5_7_minutes_avg",
        "actual_years5_7_points_avg",
        "actual_years5_7_rebounds_avg",
        "actual_years5_7_assists_avg",
        "performance_test_prediction_available",
        "forecast_support_flag",
        "classification_support_flag",
        "block1_ready_flag",
        "deliverable3_block1_note",
        "forecast_probabilities_available_flag",
        "forecast_component_prediction_available_flag",
        "supported_forecast_adjustment_flag",
        "classification_overlay_only_flag",
        "forecast_adjustment_evaluation_subset_flag",
        "forecast_support_status",
        "forecast_outlook_bucket",
        "forecast_confidence_band",
        "forecast_uncertainty_band",
        "forecast_performance_tier",
        "forecast_adjustment_mode",
        "block1_support_note",
    ]

    core = recon.copy().drop(columns=stale_block1_fields, errors="ignore")
    core["block1_merge_method"] = "no_block1_support"

    id_counts = block1.loc[block1["PLAYER_ID"].notna(), "PLAYER_ID"].value_counts()
    unique_ids = set(id_counts[id_counts == 1].index.tolist())
    block1_by_id = block1[block1["PLAYER_ID"].isin(unique_ids)].copy()
    merge_cols = [c for c in block1_by_id.columns if c not in {"PLAYER_NAME", "draft_year", "player_name_key"}]
    core = core.merge(block1_by_id[merge_cols], on="PLAYER_ID", how="left", suffixes=("", "_block1"))
    core.loc[core["predicted_class"].notna(), "block1_merge_method"] = "exact_player_id"

    name_map = block1.drop(columns=["PLAYER_ID"]).copy()
    name_map = name_map.rename(columns={"PLAYER_NAME": "PLAYER_NAME_block1", "draft_year": "draft_year_block1"})
    core = core.merge(name_map, on="player_name_key", how="left", suffixes=("", "_namefill"))
    fill_pairs = {
        "predicted_class_namefill": "predicted_class",
        "actual_class_namefill": "actual_class",
        "label_known_namefill": "label_known",
        "is_correct_namefill": "is_correct",
        "sleeper_probability_namefill": "sleeper_probability",
        "neutral_probability_namefill": "neutral_probability",
        "bust_probability_namefill": "bust_probability",
        "confidence_rating_namefill": "confidence_rating",
        "confidence_score_namefill": "confidence_score",
        "uncertainty_proxy_namefill": "uncertainty_proxy",
        "predicted_years5_7_perf_score_component_mean_namefill": "predicted_years5_7_perf_score_component_mean",
        "actual_years5_7_perf_score_component_mean_namefill": "actual_years5_7_perf_score_component_mean",
        "performance_pred_error_component_mean_namefill": "performance_pred_error_component_mean",
        "actual_years5_7_games_avg_namefill": "actual_years5_7_games_avg",
        "actual_years5_7_minutes_avg_namefill": "actual_years5_7_minutes_avg",
        "actual_years5_7_points_avg_namefill": "actual_years5_7_points_avg",
        "actual_years5_7_rebounds_avg_namefill": "actual_years5_7_rebounds_avg",
        "actual_years5_7_assists_avg_namefill": "actual_years5_7_assists_avg",
        "performance_test_prediction_available_namefill": "performance_test_prediction_available",
        "forecast_support_flag_namefill": "forecast_support_flag",
        "classification_support_flag_namefill": "classification_support_flag",
        "block1_ready_flag_namefill": "block1_ready_flag",
        "deliverable3_block1_note_namefill": "deliverable3_block1_note",
    }
    before_name_merge = core["predicted_class"].notna()
    for src, dst in fill_pairs.items():
        if src in core.columns:
            core[dst] = core[dst].where(core[dst].notna(), core[src])
    core.loc[(~before_name_merge) & core["predicted_class"].notna(), "block1_merge_method"] = "normalized_name_fallback"
    core = core.drop(columns=[c for c in core.columns if c.endswith("_namefill") or c in {"PLAYER_NAME_block1", "draft_year_block1"}], errors="ignore")

    core["forecast_probabilities_available_flag"] = core["predicted_class"].notna().astype(int)
    core["forecast_component_prediction_available_flag"] = core["performance_test_prediction_available"].fillna(0).astype(float).astype(int)
    core["supported_forecast_adjustment_flag"] = ((core["forecast_component_prediction_available_flag"] == 1) & (core["market_anchor_supported_flag"].fillna(0).astype(int) == 1)).astype(int)
    core["classification_overlay_only_flag"] = ((core["forecast_probabilities_available_flag"] == 1) & (core["supported_forecast_adjustment_flag"] == 0)).astype(int)
    core["salary_target_available_flag"] = core["target_observed_flag"].fillna(0).astype(int)
    core["salary_model_training_sample_flag"] = ((core["training_eligible_flag"].fillna(0).astype(int) == 1) & (core["holdout_for_case_study"].fillna(0).astype(int) == 0)).astype(int)
    core["forecast_adjustment_evaluation_subset_flag"] = core["supported_forecast_adjustment_flag"]

    core["forecast_support_status"] = np.select(
        [
            core["supported_forecast_adjustment_flag"] == 1,
            core["classification_overlay_only_flag"] == 1,
        ],
        ["test_prediction_available", "classification_or_truth_only"],
        default="no_block1_support",
    )

    q33 = core.loc[core["supported_forecast_adjustment_flag"] == 1, "predicted_years5_7_perf_score_component_mean"].quantile(0.33)
    q67 = core.loc[core["supported_forecast_adjustment_flag"] == 1, "predicted_years5_7_perf_score_component_mean"].quantile(0.67)
    if pd.isna(q33):
        q33 = core["predicted_years5_7_perf_score_component_mean"].quantile(0.33)
    if pd.isna(q67):
        q67 = core["predicted_years5_7_perf_score_component_mean"].quantile(0.67)

    core["forecast_outlook_bucket"] = core.apply(lambda r: outlook_bucket(r.get("predicted_class"), r.get("sleeper_probability"), r.get("neutral_probability"), r.get("bust_probability")), axis=1)
    core["forecast_confidence_band"] = core["confidence_score"].map(confidence_band)
    core["forecast_uncertainty_band"] = core["uncertainty_proxy"].map(uncertainty_band)
    core["forecast_performance_tier"] = core["predicted_years5_7_perf_score_component_mean"].map(lambda x: performance_tier(x, q33, q67))
    core["forecast_adjustment_mode"] = core.apply(adjustment_mode, axis=1)
    core["staged_extension_stance"] = core.apply(adjusted_stance, axis=1)
    core["staged_extension_reason"] = core.apply(reason_text, axis=1)
    core["deliverable3_framework_status"] = core.apply(stage_status, axis=1)
    core[["staged_open_cap_pct", "staged_target_cap_pct", "staged_hard_max_cap_pct"]] = pd.DataFrame(core.apply(negotiation_values, axis=1).tolist(), index=core.index)

    label_map = {
        "offer_now": "Offer now",
        "offer_now_disciplined_band": "Offer now at disciplined price",
        "wait_and_save_flexibility": "Wait and save flexibility",
        "avoid_overcommitting": "Avoid overcommitting",
    }
    core["staged_extension_stance_label"] = core["staged_extension_stance"].map(label_map).fillna(core["staged_extension_stance"])
    core["staged_band_text"] = core.apply(lambda r: f"Protected {pct_text(r.get('protected_price_cap_pct'))}, fair {pct_text(r.get('fair_price_cap_pct'))}, walk-away {pct_text(r.get('walk_away_max_cap_pct'))}; open {pct_text(r.get('staged_open_cap_pct'))}, target {pct_text(r.get('staged_target_cap_pct'))}, hard max {pct_text(r.get('staged_hard_max_cap_pct'))}.", axis=1)
    core["block1_support_note"] = core.apply(lambda r: "Forecast-adjustment support available from Block 1 handoff." if int(r.get("supported_forecast_adjustment_flag", 0)) == 1 else ("Only class probabilities / truth-side context available from Block 1; stance remains staging-only." if int(r.get("classification_overlay_only_flag", 0)) == 1 else "No official Block 1 support merged; remain Block 2-only."), axis=1)

    card_df = core[[
        "PLAYER_ID", "PLAYER_NAME", "draft_year", "hybrid_archetype_label", "macro_archetype", "shot_style_subtype",
        "staged_extension_stance", "staged_extension_stance_label", "staged_band_text", "staged_extension_reason",
        "deliverable3_framework_status", "forecast_support_status", "forecast_outlook_bucket", "forecast_confidence_band",
        "forecast_uncertainty_band", "comp_support_bucket", "scarcity_tier", "identity_drift_class", "block1_support_note"
    ]].copy()
    card_texts = core.apply(build_card_text, axis=1, result_type="expand")
    card_df = pd.concat([card_df, card_texts], axis=1)

    case_study_keys = {normalize_name(x) for x in DEMO_PLAYERS}
    core["case_study_flag"] = core["PLAYER_NAME"].map(normalize_name).isin(case_study_keys).astype(int)
    case_df = core[core["case_study_flag"] == 1].copy()
    if case_df.empty:
        block1_case["player_name_key"] = block1_case["PLAYER_NAME"].map(normalize_name)
        case_df = core[core["player_name_key"].isin(block1_case["player_name_key"])].copy()
    case_df = case_df[[
        "PLAYER_ID", "PLAYER_NAME", "draft_year", "hybrid_archetype_label", "macro_archetype", "shot_style_subtype",
        "provisional_action_bucket", "staged_extension_stance", "staged_extension_stance_label",
        "protected_price_cap_pct", "fair_price_cap_pct", "walk_away_max_cap_pct",
        "staged_open_cap_pct", "staged_target_cap_pct", "staged_hard_max_cap_pct",
        "predicted_class", "actual_class", "sleeper_probability", "neutral_probability", "bust_probability",
        "confidence_rating", "confidence_score", "forecast_support_status", "forecast_adjustment_mode",
        "predicted_years5_7_perf_score_component_mean", "actual_years5_7_perf_score_component_mean",
        "performance_pred_error_component_mean", "comp_support_bucket", "scarcity_tier", "identity_drift_class",
        "deliverable3_framework_status", "staged_extension_reason", "block1_support_note"
    ]].sort_values(["PLAYER_NAME", "draft_year"])

    supported = core[core["forecast_adjustment_evaluation_subset_flag"] == 1].copy()
    if not supported.empty:
        supported["perf_sq_err"] = supported["performance_pred_error_component_mean"].astype(float) ** 2
        supported["abs_perf_err"] = supported["performance_pred_error_component_mean"].abs()
        acc = supported.loc[supported["label_known"] == 1, "is_correct"].mean()
        rmse = float(np.sqrt(supported["perf_sq_err"].mean()))
        mae = float(supported["abs_perf_err"].mean())
    else:
        acc = np.nan
        rmse = np.nan
        mae = np.nan

    status_counts = core["deliverable3_framework_status"].value_counts(dropna=False).rename_axis("framework_status").reset_index(name="player_count")
    stance_counts = core["staged_extension_stance"].value_counts(dropna=False).rename_axis("framework_status").reset_index(name="player_count")
    stance_counts["metric_group"] = "staged_extension_stance"
    status_counts["metric_group"] = "framework_status"
    support_counts = core["forecast_support_status"].value_counts(dropna=False).rename_axis("framework_status").reset_index(name="player_count")
    support_counts["metric_group"] = "forecast_support_status"
    matrix = pd.concat([status_counts, stance_counts, support_counts], ignore_index=True)[["metric_group", "framework_status", "player_count"]]

    supported_summary = pd.DataFrame([
        ("supported_subset_rows", len(supported)),
        ("supported_subset_with_salary_target", int(supported["salary_target_available_flag"].sum()) if not supported.empty else 0),
        ("supported_subset_label_accuracy", acc),
        ("supported_subset_perf_rmse", rmse),
        ("supported_subset_perf_mae", mae),
    ], columns=["metric", "value"])

    crosstab = pd.crosstab(supported["actual_class"].fillna("unknown"), supported["staged_extension_stance_label"].fillna("unknown")).reset_index().rename(columns={"actual_class": "actual_class"})

    rule_reference = pd.DataFrame([
        ("block2_only_provisional", "No official Block 1 support merged; keep Block 2 action logic only."),
        ("block1_classification_overlay_only", "Official class probabilities merged, but no supported performance forecast row; use only as descriptive overlay."),
        ("forecast_adjusted_supported_subset", "Official Block 1 performance forecast row available; allow staged action changes within the existing market band."),
        ("upside_push_within_band", "Only move one action step more aggressive; never expand beyond walk-away max."),
        ("downside_caution_within_band", "Only move one action step more cautious; do not impose a new discount model."),
    ], columns=["rule_name", "rule_text"])

    summary = pd.DataFrame([
        ("framework_rows", len(core), "Total player rows in the staged Deliverable 3 framework."),
        ("market_anchor_supported_rows", int(core["market_anchor_supported_flag"].fillna(0).sum()), "Rows with supported protected/fair/walk-away anchor band."),
        ("forecast_probabilities_available_rows", int(core["forecast_probabilities_available_flag"].sum()), "Rows with official Block 1 class probabilities merged."),
        ("supported_forecast_adjustment_rows", int(core["supported_forecast_adjustment_flag"].sum()), "Rows eligible for supported forecast-adjusted logic."),
        ("salary_model_training_sample_rows", int(core["salary_model_training_sample_flag"].sum()), "Rows eligible for salary-model training after case-study holdout exclusion."),
    ], columns=["metric", "value", "notes"])

    summary_md = ["# Deliverable 3 Staged Final Framework", "", "This export completes Deliverable 3 as a staged salary-decision-support system rather than as a single fully validated final recommendation layer.", "", "## What is complete now", "- Block 2 market-anchor logic remains the pricing backbone.", "- Salary-model and market-band reconciliation are preserved.", "- Official Block 1 forecast support is now merged through the finalized handoff file.", "- Forecast-adjusted posture is only activated on the supported subset with test-set performance forecast support.", "", "## What stays conditional", "- Full-cohort final extension guidance remains unvalidated.", "- Durability / availability risk is still not integrated as a separate resolved module.", "- The current uncertainty input is still the Block 1 confidence proxy, not a richer interval package.", "", "## Core counts", "", "| Metric | Value | Notes |", "|---|---:|---|"]
    for _, row in summary.iterrows():
        summary_md.append(f"| {row['metric']} | {row['value']} | {row['notes']} |")
    summary_md += ["", "### Supported forecast-adjustment evaluation subset", "", "| Metric | Value |", "|---|---:|"]
    for _, row in supported_summary.iterrows():
        summary_md.append(f"| {row['metric']} | {row['value']} |")

    framework_path = output_dir / "deliverable3_staged_final_framework_table.csv"
    cards_path = output_dir / "deliverable3_staged_final_guidance_cards.csv"
    case_path = output_dir / "deliverable3_final_case_study_table.csv"
    matrix_path = output_dir / "deliverable3_framework_status_matrix.csv"
    supported_path = output_dir / "deliverable3_forecast_adjustment_supported_subset.csv"
    supported_summary_path = output_dir / "deliverable3_forecast_adjustment_supported_subset_summary.csv"
    supported_crosstab_path = output_dir / "deliverable3_forecast_adjustment_supported_subset_crosstab.csv"
    summary_path = output_dir / "deliverable3_staged_final_summary.csv"
    summary_md_path = output_dir / "deliverable3_staged_final_summary.md"
    rule_ref_path = output_dir / "deliverable3_final_rule_reference.csv"
    metadata_path = output_dir / "deliverable3_staged_final_metadata.json"

    core.to_csv(framework_path, index=False)
    card_df.to_csv(cards_path, index=False)
    case_df.to_csv(case_path, index=False)
    matrix.to_csv(matrix_path, index=False)
    supported.to_csv(supported_path, index=False)
    supported_summary.to_csv(supported_summary_path, index=False)
    crosstab.to_csv(supported_crosstab_path, index=False)
    summary.to_csv(summary_path, index=False)
    summary_md_path.write_text("\n".join(summary_md), encoding="utf-8")
    rule_reference.to_csv(rule_ref_path, index=False)

    metadata = {
        "script": "09_build_staged_extension_framework.py",
        "project_root": str(project_root),
        "inputs": {
            "reconciliation_table": str(reconciliation_path),
            "block1_forecast_handoff": str(block1_handoff_path),
            "block1_case_study_examples": str(case_study_input_path),
        },
        "outputs": [
            framework_path.name,
            cards_path.name,
            case_path.name,
            matrix_path.name,
            supported_path.name,
            supported_summary_path.name,
            supported_crosstab_path.name,
            summary_path.name,
            summary_md_path.name,
            rule_ref_path.name,
        ],
        "counts": {
            "framework_rows": int(len(core)),
            "forecast_probabilities_available_rows": int(core["forecast_probabilities_available_flag"].sum()),
            "supported_forecast_adjustment_rows": int(core["supported_forecast_adjustment_flag"].sum()),
            "salary_model_training_sample_rows": int(core["salary_model_training_sample_flag"].sum()),
        },
        "notes": [
            "This phase consumes the official Deliverable 3 Block 1 handoff rather than raw Deliverable 1 files.",
            "Forecast-adjusted posture is only activated on the supported subset with test prediction support.",
            "Rows without supported forecast adjustment remain staged as Block 2-only or classification-overlay-only outputs.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    log_path = output_dir / "salary_decision_support_workflow_log.txt"
    append_workflow_log(
        log_path,
        f"""
Phase 9 complete — staged final Deliverable 3 extension framework assembled.

Inputs used:
- reconciliation_table: {reconciliation_path}
- block1_forecast_handoff: {block1_handoff_path}
- block1_case_study_examples: {case_study_input_path}

Workflow logic:
- replaced earlier ad hoc forecast overlay fields with the official Deliverable 3 Block 1 handoff,
- created explicit flags separating salary-model training sample, forecast-adjustment evaluation subset, and unsupported rows,
- activated forecast-adjusted stance changes only on the supported subset with official test performance predictions,
- kept the broader cohort in staged Block 2-only or classification-overlay-only status when stronger Block 1 support was not available,
- assembled final stakeholder-facing case-study outputs for Trae Young and Nikola Vučević.

Main outputs:
- {framework_path.name}
- {cards_path.name}
- {case_path.name}
- {matrix_path.name}
- {supported_path.name}
- {supported_summary_path.name}
- {supported_crosstab_path.name}
- {summary_path.name}
- {summary_md_path.name}

Important boundary:
- this phase completes Deliverable 3 as a staged final system, not as a fully validated final recommendation layer for the whole cohort.
- durability / availability risk is still pending as a separate module.
        """,
    )


if __name__ == "__main__":
    main()
