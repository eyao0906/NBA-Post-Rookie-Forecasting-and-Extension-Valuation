"""Stage 7-8: model comparison, visuals, and stakeholder narrative snippets."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def compare_metrics(baseline_metrics: pd.DataFrame, lstm_metrics: pd.DataFrame) -> pd.DataFrame:
    b = baseline_metrics.copy()
    l = lstm_metrics.copy()
    b["Model"] = "Baseline_Ridge"
    l["Model"] = "LSTM"
    comp = pd.concat([b, l], ignore_index=True)
    return comp[["Model", "target", "rmse", "mae"]].sort_values(["target", "Model"])


def merge_predictions(baseline_preds: pd.DataFrame, lstm_preds: pd.DataFrame) -> pd.DataFrame:
    if baseline_preds.empty or lstm_preds.empty:
        return pd.DataFrame()

    b = baseline_preds[["Player_ID", "Player_Name", "Target", "Actual", "Baseline_Pred"]].copy()
    l = lstm_preds[["Player_ID", "Player_Name", "Target", "Actual", "LSTM_Pred"]].copy()

    merged = pd.merge(
        b,
        l,
        on=["Player_ID", "Player_Name", "Target", "Actual"],
        how="inner",
    )
    return merged


def plot_prediction_trajectories(merged_preds: pd.DataFrame, visual_dir: Path) -> None:
    if merged_preds.empty:
        return

    visual_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    for target, g in merged_preds.groupby("Target"):
        # Sort by actual so lines reveal ranking/relative misses
        g = g.sort_values("Actual").reset_index(drop=True)
        g["Rank"] = g.index + 1

        plt.figure(figsize=(11, 6))
        plt.plot(g["Rank"], g["Actual"], label="Actual", linewidth=2)
        plt.plot(g["Rank"], g["Baseline_Pred"], label="Baseline", alpha=0.85)
        plt.plot(g["Rank"], g["LSTM_Pred"], label="LSTM", alpha=0.85)
        plt.title(f"Year-5 {target}: Actual vs Baseline vs LSTM")
        plt.xlabel("Player Rank (sorted by actual)")
        plt.ylabel(target)
        plt.legend()
        plt.tight_layout()
        plt.savefig(visual_dir / f"trajectory_{target}.png", dpi=180)
        plt.close()


def identify_case_studies(merged_preds: pd.DataFrame) -> pd.DataFrame:
    """Identify Sleeper/Bust examples where LSTM beats baseline by a margin."""
    if merged_preds.empty:
        return pd.DataFrame()

    out = merged_preds.copy()
    out["Baseline_AbsErr"] = (out["Actual"] - out["Baseline_Pred"]).abs()
    out["LSTM_AbsErr"] = (out["Actual"] - out["LSTM_Pred"]).abs()
    out["LSTM_Improvement"] = out["Baseline_AbsErr"] - out["LSTM_AbsErr"]
    out["Baseline_Bias"] = out["Baseline_Pred"] - out["Actual"]

    sleepers = out[(out["Baseline_Bias"] < 0) & (out["LSTM_Improvement"] > 0)].copy()
    busts = out[(out["Baseline_Bias"] > 0) & (out["LSTM_Improvement"] > 0)].copy()

    sleeper_top = sleepers.sort_values("LSTM_Improvement", ascending=False).head(1)
    bust_top = busts.sort_values("LSTM_Improvement", ascending=False).head(1)

    if sleeper_top.empty and bust_top.empty:
        return pd.DataFrame()

    sleeper_top["CaseType"] = "Sleeper"
    bust_top["CaseType"] = "Bust"
    return pd.concat([sleeper_top, bust_top], ignore_index=True)
