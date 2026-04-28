"""End-to-end runner for CaseStudy3.

Usage:
    python src/run_pipeline.py --data-dir ./data --visual-dir ./visual --report-dir ./report --require-gpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from baseline_model import train_baseline
from config import PipelineConfig, ensure_dirs
from data_acquisition import build_draft_cohort, build_raw_dataset
from evaluate_and_visualize import (
    compare_metrics,
    identify_case_studies,
    merge_predictions,
    plot_prediction_trajectories,
)
from feature_engineering import build_baseline_aggregates, engineer_features
from lstm_model import train_lstm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--visual-dir", type=Path, default=Path("visual"))
    p.add_argument("--report-dir", type=Path, default=Path("report"))
    p.add_argument("--require-gpu", action="store_true", help="Fail if TensorFlow cannot see a GPU.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig()

    ensure_dirs(args.data_dir, args.visual_dir, args.report_dir)

    # ------------------------
    # Step 2: Cohort
    # ------------------------
    cohort_path = args.data_dir / "raw" / "cohort_2005_2019.csv"
    cohort = build_draft_cohort(cfg, cohort_path)

    # ------------------------
    # Step 3: Raw sequential data
    # ------------------------
    raw_path = args.data_dir / "raw" / "player_gamelogs_first4.csv"
    raw_df = build_raw_dataset(cohort, cfg, raw_path)

    # ------------------------
    # Step 4: Feature engineering
    # ------------------------
    fe_df = engineer_features(raw_df)
    fe_path = args.data_dir / "intermediate" / "engineered_gamelogs.csv"
    fe_df.to_csv(fe_path, index=False)

    # ------------------------
    # Step 5: Baseline model
    # ------------------------
    base_input = build_baseline_aggregates(fe_df)
    base_input.to_csv(args.data_dir / "processed" / "baseline_input.csv", index=False)

    baseline_metrics, baseline_preds = train_baseline(base_input, cfg)
    baseline_metrics.to_csv(args.report_dir / "baseline_metrics.csv", index=False)
    baseline_preds.to_csv(args.report_dir / "baseline_predictions.csv", index=False)

    # ------------------------
    # Step 6: LSTM model (GPU-first)
    # ------------------------
    lstm_metrics, lstm_preds, models = train_lstm(fe_df, cfg, require_gpu=args.require_gpu)
    lstm_metrics.to_csv(args.report_dir / "lstm_metrics.csv", index=False)
    lstm_preds.to_csv(args.report_dir / "lstm_predictions.csv", index=False)

    # Save model weights if models exist
    if models.get("minutes_model") is not None:
        models["minutes_model"].save(args.report_dir / "lstm_minutes_model.keras")
    if models.get("winshares_model") is not None:
        models["winshares_model"].save(args.report_dir / "lstm_winshares_model.keras")

    # ------------------------
    # Step 7-8: Evaluation & stakeholder story artifacts
    # ------------------------
    comp = compare_metrics(baseline_metrics, lstm_metrics)
    comp.to_csv(args.report_dir / "model_comparison_metrics.csv", index=False)

    merged_preds = merge_predictions(baseline_preds, lstm_preds)
    merged_preds.to_csv(args.report_dir / "merged_predictions.csv", index=False)

    plot_prediction_trajectories(merged_preds, args.visual_dir)

    case_studies = identify_case_studies(merged_preds)
    case_studies.to_csv(args.report_dir / "case_studies.csv", index=False)

    # Lightweight markdown summary
    summary_md = args.report_dir / "case_study3_summary.md"
    with summary_md.open("w", encoding="utf-8") as f:
        f.write("# Case Study 3 Summary\n\n")
        f.write("## Model Comparison\n\n")
        f.write(comp.to_markdown(index=False) if not comp.empty else "No comparison generated.\n")
        f.write("\n\n## Suggested Case Studies\n\n")
        f.write(case_studies.to_markdown(index=False) if not case_studies.empty else "No candidate case studies identified.\n")


if __name__ == "__main__":
    main()
