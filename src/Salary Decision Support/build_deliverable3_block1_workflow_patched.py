from __future__ import annotations

import argparse
import difflib
import sys
import unicodedata
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from salary_workflow_utils import (
    DEMO_PLAYERS,
    append_workflow_log,
    ensure_dir,
    inventory_markdown,
    records_to_rows,
    resolve_all_expected_files,
    summarize_inventory,
    write_json,
)

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


CLASS_PROB_FILE = "final_lstm_predictions_all_players.csv"
SPLIT_SCORE_FILE = "player_train_test_split_with_score.csv"
PERF_TEST_FILE = "player_performance_testset_forecast_result.csv"
ALL_STATS_FILE = "all_player_stats_1999-2025.csv"
RESID_LABEL_FILE = "player_residual_labels_with_majority_vote.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Deliverable 3 Block 1 handoff tables from current Deliverable 1 outputs."
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
        "--data-dir",
        type=Path,
        default=None,
        help="Optional override for data/Salary Decision Support.",
    )
    return parser.parse_args()


def normalize_name(name: object) -> str:
    if pd.isna(name):
        return ""
    text = str(name).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    cleaned = []
    prev_space = False
    for ch in text:
        if ch.isalnum():
            cleaned.append(ch)
            prev_space = False
        else:
            if not prev_space:
                cleaned.append(" ")
            prev_space = True
    return " ".join("".join(cleaned).split())


def make_season_label(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def season_start_from_any_label(value: object) -> Optional[int]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if len(text) >= 7 and text[:4].isdigit() and text[4] == '-':
        return int(text[:4])
    try:
        dt = pd.to_datetime(text, format="%b-%y")
        return int(dt.year if dt.month >= 10 else dt.year - 1)
    except Exception:
        return None


def parse_window(window: str) -> tuple[int, int]:
    start, end = str(window).split("-")
    return int(start), int(end)


def season_labels_for_window(window: str) -> list[str]:
    start, end = parse_window(window)
    return [make_season_label(y) for y in range(start, end)]


def earliest_window(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    temp["input_start_year"] = temp["INPUT_SEASON"].astype(str).str.slice(0, 4).astype(int)
    temp = temp.sort_values(["PLAYER_ID", "input_start_year", "FORECAST_SEASON", "INPUT_SEASON"])
    return temp.groupby("PLAYER_ID", as_index=False).head(1).drop(columns=["input_start_year"])


def confidence_to_rating(score: float) -> str:
    if pd.isna(score):
        return "Unknown"
    if score >= 0.80:
        return "High"
    if score >= 0.60:
        return "Medium"
    return "Low"


def resolve_required_csv(project_root: Path, data_dir: Path, filename: str) -> Path:
    candidates = [
        project_root / filename,
        data_dir / filename,
    ]
    existing = [p for p in candidates if p.exists()]
    if existing:
        # Prefer the project-root copy when present so refreshed handoff inputs can
        # override older copies under data/. Otherwise choose the most recently
        # modified available file.
        root_copy = project_root / filename
        if root_copy.exists():
            return root_copy
        existing.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return existing[0]

    matches = sorted(project_root.rglob(filename), key=lambda p: (p.stat().st_mtime if p.exists() else 0), reverse=True)
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Missing required input: {filename} (searched project root and data dir)")


def load_required_csv(project_root: Path, data_dir: Path, filename: str) -> tuple[pd.DataFrame, Path]:
    path = resolve_required_csv(project_root, data_dir, filename)
    return pd.read_csv(path), path


def build_identity_table(split_score: pd.DataFrame, all_stats: pd.DataFrame) -> pd.DataFrame:
    identity = earliest_window(split_score[["PLAYER_ID", "PLAYER_NAME", "INPUT_SEASON", "FORECAST_SEASON"]].drop_duplicates())
    identity["player_name_key"] = identity["PLAYER_NAME"].map(normalize_name)
    identity["draft_year_from_window"] = identity["INPUT_SEASON"].astype(str).str.slice(0, 4).astype(int)

    draft_lookup = (
        all_stats.assign(
            draft_year=all_stats["season"].map(season_start_from_any_label),
            player_name_key=all_stats["PLAYER_NAME"].map(normalize_name),
        )
        .groupby(["PLAYER_ID", "player_name_key"], as_index=False)
        .agg(
            PLAYER_NAME_stats=("PLAYER_NAME", "first"),
            draft_year_stats=("draft_year", "min"),
        )
    )

    identity = identity.merge(
        draft_lookup[["PLAYER_ID", "draft_year_stats"]],
        on="PLAYER_ID",
        how="left",
    )
    identity["draft_year"] = identity["draft_year_stats"].fillna(identity["draft_year_from_window"]).astype("Int64")
    identity["identity_source"] = np.where(identity["draft_year_stats"].notna(), "player_id_all_stats", "input_window_fallback")
    return identity.drop(columns=["draft_year_stats"])


def build_actual_component_table(split_score: pd.DataFrame) -> pd.DataFrame:
    base = earliest_window(split_score.copy())
    score_cols = [
        "Scoring_Metric_Score",
        "Efficiency_Metric_Score",
        "Playmaking_Metric_Score",
        "Defense_Metric_Score",
        "Control_Metric_Score",
    ]
    rename_map = {
        "Scoring_Metric_Score": "actual_scoring_metric_score",
        "Efficiency_Metric_Score": "actual_efficiency_metric_score",
        "Playmaking_Metric_Score": "actual_playmaking_metric_score",
        "Defense_Metric_Score": "actual_defense_metric_score",
        "Control_Metric_Score": "actual_control_metric_score",
    }
    out = base[["PLAYER_ID", "PLAYER_NAME", "INPUT_SEASON", "FORECAST_SEASON", "SPLIT"] + score_cols].rename(columns=rename_map)
    actual_metric_cols = list(rename_map.values())
    out["actual_years5_7_perf_score_component_mean"] = out[actual_metric_cols].mean(axis=1)
    out = out.rename(columns={"SPLIT": "holdout_split_tag"})
    out["player_name_key"] = out["PLAYER_NAME"].map(normalize_name)
    return out


def build_actual_boxscore_targets(identity: pd.DataFrame, all_stats: pd.DataFrame) -> pd.DataFrame:
    stats = all_stats.copy()
    stats["season"] = stats["season"].astype(str)

    rows = []
    for row in identity.itertuples(index=False):
        seasons = season_labels_for_window(row.FORECAST_SEASON)
        g = stats[stats["PLAYER_ID"] == row.PLAYER_ID]
        g = g[g["season"].isin(seasons)]
        season_count = len(seasons)
        rows.append(
            {
                "PLAYER_ID": row.PLAYER_ID,
                "player_name_key": row.player_name_key,
                "actual_years5_7_games_avg": g["GP"].sum() / season_count if season_count else np.nan,
                "actual_years5_7_minutes_avg": g["MIN"].sum() / season_count if season_count else np.nan,
                "actual_years5_7_points_avg": g["PTS"].sum() / season_count if season_count else np.nan,
                "actual_years5_7_rebounds_avg": g["REB"].sum() / season_count if season_count else np.nan,
                "actual_years5_7_assists_avg": g["AST"].sum() / season_count if season_count else np.nan,
                "boxscore_target_support_flag": "ok" if not g.empty else "missing_all_stats_window",
            }
        )
    return pd.DataFrame(rows)


def build_test_prediction_table(perf_test: pd.DataFrame) -> pd.DataFrame:
    pred = perf_test.copy()
    pred["player_name_key"] = pred["PLAYER_NAME"].map(normalize_name)
    pred = pred.rename(
        columns={
            "Scoring_Score_Predicted": "predicted_scoring_metric_score",
            "Efficiency_Score_Predicted": "predicted_efficiency_metric_score",
            "Playmaking_Score_Predicted": "predicted_playmaking_metric_score",
            "Defense_Score_Predicted": "predicted_defense_metric_score",
            "Control_Score_Predicted": "predicted_control_metric_score",
            "Scoring_Score_Actual": "test_actual_scoring_metric_score",
            "Efficiency_Score_Actual": "test_actual_efficiency_metric_score",
            "Playmaking_Score_Actual": "test_actual_playmaking_metric_score",
            "Defense_Score_Actual": "test_actual_defense_metric_score",
            "Control_Score_Actual": "test_actual_control_metric_score",
        }
    )
    pred_metric_cols = [
        "predicted_scoring_metric_score",
        "predicted_efficiency_metric_score",
        "predicted_playmaking_metric_score",
        "predicted_defense_metric_score",
        "predicted_control_metric_score",
    ]
    act_metric_cols = [
        "test_actual_scoring_metric_score",
        "test_actual_efficiency_metric_score",
        "test_actual_playmaking_metric_score",
        "test_actual_defense_metric_score",
        "test_actual_control_metric_score",
    ]
    pred["predicted_years5_7_perf_score_component_mean"] = pred[pred_metric_cols].mean(axis=1)
    pred["test_actual_years5_7_perf_score_component_mean"] = pred[act_metric_cols].mean(axis=1)
    pred["performance_pred_error_component_mean"] = (
        pred["predicted_years5_7_perf_score_component_mean"] - pred["test_actual_years5_7_perf_score_component_mean"]
    )
    pred["performance_test_prediction_available"] = 1
    keep = [
        "PLAYER_ID",
        "PLAYER_NAME",
        "player_name_key",
        "predicted_years5_7_perf_score_component_mean",
        "test_actual_years5_7_perf_score_component_mean",
        "performance_pred_error_component_mean",
        "performance_test_prediction_available",
    ] + pred_metric_cols
    return pred[keep]


def build_classification_table(class_probs: pd.DataFrame) -> pd.DataFrame:
    df = class_probs.copy()
    df["player_name_key"] = df["PLAYER_NAME"].map(normalize_name)
    df = df.rename(
        columns={
            "prob_Sleeper": "sleeper_probability",
            "prob_Neutral": "neutral_probability",
            "prob_Bust": "bust_probability",
            "confidence": "confidence_rating",
            "max_pred_prob": "confidence_score",
        }
    )
    if "confidence_rating" not in df.columns:
        df["confidence_rating"] = df["confidence_score"].map(confidence_to_rating)
    df["uncertainty_proxy"] = 1.0 - df["confidence_score"].astype(float)
    df["classification_support_flag"] = np.where(df["label_known"].fillna(0).astype(int) == 1, "label_known", "prediction_only")
    return df[[
        "PLAYER_NAME", "player_name_key", "predicted_class", "sleeper_probability", "neutral_probability",
        "bust_probability", "actual_class", "label_known", "is_correct", "confidence_score",
        "confidence_rating", "uncertainty_proxy", "classification_support_flag"
    ]]




def fuzzy_identity_lookup(unresolved: pd.DataFrame, identity: pd.DataFrame, threshold: float = 0.88) -> pd.DataFrame:
    """Attempt a conservative fuzzy fallback on normalized names for mojibake cases."""
    if unresolved.empty:
        return unresolved
    identity_small = identity[["PLAYER_ID", "player_name_key", "draft_year", "INPUT_SEASON", "FORECAST_SEASON", "identity_source"]].drop_duplicates()
    keys = identity_small["player_name_key"].dropna().unique().tolist()
    records = []
    for row in unresolved.itertuples(index=False):
        key = getattr(row, "player_name_key")
        match = difflib.get_close_matches(key, keys, n=1, cutoff=threshold)
        if match:
            ident = identity_small[identity_small["player_name_key"] == match[0]].iloc[0].to_dict()
            base = row._asdict()
            base.update(ident)
            base["identity_source"] = f"fuzzy_name_fallback:{match[0]}"
            records.append(base)
        else:
            records.append(row._asdict())
    return pd.DataFrame(records)

def build_join_audit(block1: pd.DataFrame, identity: pd.DataFrame, class_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"check_name": "classification_rows", "value": len(class_df)},
        {"check_name": "identity_rows", "value": len(identity)},
        {"check_name": "block1_rows", "value": len(block1)},
        {"check_name": "classification_with_player_id", "value": int(block1["PLAYER_ID"].notna().sum())},
        {"check_name": "classification_without_player_id", "value": int(block1["PLAYER_ID"].isna().sum())},
        {"check_name": "performance_predictions_available", "value": int(block1["performance_test_prediction_available"].fillna(0).sum())},
        {"check_name": "actual_component_targets_available", "value": int(block1["actual_years5_7_perf_score_component_mean"].notna().sum())},
        {"check_name": "actual_boxscore_targets_available", "value": int(block1["actual_years5_7_games_avg"].notna().sum())},
        {"check_name": "confidence_high_count", "value": int((block1["confidence_rating"] == "High").sum())},
        {"check_name": "confidence_medium_count", "value": int((block1["confidence_rating"] == "Medium").sum())},
        {"check_name": "confidence_low_count", "value": int((block1["confidence_rating"] == "Low").sum())},
        {"check_name": "missing_identity_name_keys", "value": int((~class_df["player_name_key"].isin(identity["player_name_key"])).sum())},
        {"check_name": "test_predictions_name_keys", "value": pred_df["player_name_key"].nunique()},
    ]
    return pd.DataFrame(rows)


def build_case_study_table(block1: pd.DataFrame) -> pd.DataFrame:
    keys = {normalize_name(name) for name in DEMO_PLAYERS}
    out = block1[block1["player_name_key"].isin(keys)].copy()
    out["case_study_flag"] = 1
    out = out.sort_values(["PLAYER_NAME", "draft_year"])
    return out


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve() if args.project_root else CURRENT_DIR
    data_dir = args.data_dir.resolve() if args.data_dir else project_root / "data"
    output_dir = args.output_dir.resolve() if args.output_dir else project_root / "Output" / "Salary Decision Support"
    output_dir.mkdir(parents=True, exist_ok=True)

    class_probs, class_probs_path = load_required_csv(project_root, data_dir, CLASS_PROB_FILE)
    split_score, split_score_path = load_required_csv(project_root, data_dir, SPLIT_SCORE_FILE)
    perf_test, perf_test_path = load_required_csv(project_root, data_dir, PERF_TEST_FILE)
    all_stats, all_stats_path = load_required_csv(project_root, data_dir, ALL_STATS_FILE)

    identity = build_identity_table(split_score, all_stats)
    class_df = build_classification_table(class_probs)
    actual_components = build_actual_component_table(split_score)
    actual_boxscore = build_actual_boxscore_targets(identity, all_stats)
    pred_df = build_test_prediction_table(perf_test)

    block1 = class_df.merge(
        identity[["PLAYER_ID", "PLAYER_NAME", "player_name_key", "draft_year", "INPUT_SEASON", "FORECAST_SEASON", "identity_source"]],
        on=["PLAYER_NAME", "player_name_key"],
        how="left",
    )
    # fallback for names that differ only by accent/punctuation or stats spelling
    unresolved = block1[block1["PLAYER_ID"].isna()].drop(columns=["PLAYER_ID", "draft_year", "INPUT_SEASON", "FORECAST_SEASON", "identity_source"])
    if not unresolved.empty:
        fallback = identity[["PLAYER_ID", "player_name_key", "draft_year", "INPUT_SEASON", "FORECAST_SEASON", "identity_source"]].drop_duplicates()
        unresolved = unresolved.merge(fallback, on="player_name_key", how="left")
        unresolved = fuzzy_identity_lookup(unresolved, identity, threshold=0.88)
        resolved = block1[block1["PLAYER_ID"].notna()]
        block1 = pd.concat([resolved, unresolved], ignore_index=True, sort=False)

    block1 = block1.merge(
        actual_components.drop(columns=["PLAYER_NAME"]),
        on=["PLAYER_ID", "player_name_key", "INPUT_SEASON", "FORECAST_SEASON"],
        how="left",
    )
    block1 = block1.merge(actual_boxscore, on=["PLAYER_ID", "player_name_key"], how="left")
    block1 = block1.merge(pred_df.drop(columns=["PLAYER_NAME"]), on=["PLAYER_ID", "player_name_key"], how="left")

    # simple support flags for Deliverable 3 Block 1 integration
    block1["forecast_support_flag"] = np.select(
        [
            block1["predicted_years5_7_perf_score_component_mean"].notna(),
            block1["actual_years5_7_perf_score_component_mean"].notna(),
        ],
        ["test_prediction_available", "actual_target_only"],
        default="classification_only",
    )
    block1["block1_ready_flag"] = np.where(block1["PLAYER_ID"].notna(), "ready_for_name_or_id_merge", "name_only_merge")
    block1["deliverable3_block1_note"] = np.where(
        block1["predicted_years5_7_perf_score_component_mean"].notna(),
        "classification probabilities plus test-set performance forecast available",
        np.where(
            block1["actual_years5_7_perf_score_component_mean"].notna(),
            "classification probabilities available; only realized target-side performance components available",
            "classification probabilities only; no linked performance component row in current split table",
        ),
    )

    # column order
    ordered_cols = [
        "PLAYER_ID", "PLAYER_NAME", "player_name_key", "draft_year", "INPUT_SEASON", "FORECAST_SEASON",
        "predicted_class", "actual_class", "label_known", "is_correct",
        "sleeper_probability", "neutral_probability", "bust_probability",
        "confidence_rating", "confidence_score", "uncertainty_proxy",
        "holdout_split_tag",
        "actual_years5_7_perf_score_component_mean",
        "predicted_years5_7_perf_score_component_mean",
        "performance_pred_error_component_mean",
        "actual_scoring_metric_score", "actual_efficiency_metric_score", "actual_playmaking_metric_score",
        "actual_defense_metric_score", "actual_control_metric_score",
        "predicted_scoring_metric_score", "predicted_efficiency_metric_score", "predicted_playmaking_metric_score",
        "predicted_defense_metric_score", "predicted_control_metric_score",
        "actual_years5_7_games_avg", "actual_years5_7_minutes_avg", "actual_years5_7_points_avg",
        "actual_years5_7_rebounds_avg", "actual_years5_7_assists_avg",
        "boxscore_target_support_flag", "performance_test_prediction_available",
        "identity_source", "classification_support_flag", "forecast_support_flag",
        "block1_ready_flag", "deliverable3_block1_note",
    ]
    for col in ordered_cols:
        if col not in block1.columns:
            block1[col] = np.nan
    block1 = block1[ordered_cols].sort_values(["PLAYER_NAME", "draft_year"], kind="stable")

    audit = build_join_audit(block1, identity, class_df, pred_df)
    case_studies = build_case_study_table(block1)

    block1_path = output_dir / "deliverable3_block1_forecast_handoff.csv"
    audit_path = output_dir / "deliverable3_block1_join_audit.csv"
    case_path = output_dir / "deliverable3_block1_case_study_examples.csv"
    log_path = output_dir / "deliverable3_block1_workflow_log.txt"

    block1.to_csv(block1_path, index=False)
    audit.to_csv(audit_path, index=False)
    case_studies.to_csv(case_path, index=False)

    log_lines = [
        "Deliverable 3 Block 1 workflow log",
        "=================================",
        "",
        f"project_root: {project_root}",
        f"data_dir: {data_dir}",
        f"output_dir: {output_dir}",
        "",
        "Inputs used:",
        f"- {class_probs_path.name}: finalized Sleeper/Neutral/Bust probabilities and confidence proxy. Resolved from {class_probs_path}.",
        f"- {split_score_path.name}: earliest-window realized Years 5-7 component-score truth by player. Resolved from {split_score_path}.",
        f"- {perf_test_path.name}: test-set performance component forecasts. Resolved from {perf_test_path}.",
        f"- {all_stats_path.name}: actual Years 5-7 games / minutes / points / rebounds / assists averages and draft-year bridge. Resolved from {all_stats_path}.",
        "",
        "Workflow choices:",
        "- One earliest INPUT_SEASON/FORECAST_SEASON window per PLAYER_ID is retained to match post-rookie extension framing.",
        "- player_name_key is an accent-insensitive normalized name key used to bridge Andrew Bynum type spelling differences.",
        "- confidence_score from the finalized LSTM probability table is treated as the current uncertainty proxy for Deliverable 3 Block 1.",
        "- predicted performance components are only available for the held-out test players in the current exported results.",
        "- non-test players still receive class probabilities and realized target-side component truth when present.",
        "",
        "Outputs written:",
        f"- {block1_path.name}",
        f"- {audit_path.name}",
        f"- {case_path.name}",
        "",
        f"Row count written: {len(block1):,}",
        f"Rows with PLAYER_ID resolved: {int(block1['PLAYER_ID'].notna().sum()):,}",
        f"Rows with test performance predictions: {int(block1['performance_test_prediction_available'].fillna(0).sum()):,}",
        f"Case-study rows: {len(case_studies):,}",
    ]
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    print(f"Wrote: {block1_path}")
    print(f"Wrote: {audit_path}")
    print(f"Wrote: {case_path}")
    print(f"Wrote: {log_path}")


if __name__ == "__main__":
    main()
