"""Stage 4: Feature engineering for sequence-aware modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _to_minutes_decimal(min_str: str) -> float:
    """Convert 'MM:SS' string to decimal minutes."""
    if pd.isna(min_str):
        return 0.0
    if isinstance(min_str, (float, int)):
        return float(min_str)
    try:
        mm, ss = str(min_str).split(":")
        return float(mm) + float(ss) / 60.0
    except Exception:  # noqa: BLE001
        return 0.0


def engineer_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df.copy()

    df = raw_df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df = df.sort_values(["Player_ID", "GAME_DATE"]).reset_index(drop=True)

    # Minutes normalization
    if "MIN" in df.columns:
        df["Minutes"] = df["MIN"].apply(_to_minutes_decimal)
    else:
        df["Minutes"] = 0.0

    # Basic proxies (prefer direct columns if present)
    for col in ["PTS", "REB", "AST", "FGA", "FTA"]:
        if col not in df.columns:
            df[col] = 0.0

    # True Shooting Percentage (TS%) = PTS / (2 * (FGA + 0.44*FTA))
    denom = 2.0 * (df["FGA"] + 0.44 * df["FTA"])
    df["TS_Pct"] = np.where(denom > 0, df["PTS"] / denom, 0.0)

    # Required engineered columns
    df["Rest_Days"] = (
        df.groupby("Player_ID")["GAME_DATE"].diff().dt.days.fillna(0).clip(lower=0)
    )
    df["Is_B2B"] = (df["Rest_Days"] == 1).astype(int)
    df["Rolling_5G_TS"] = (
        df.groupby("Player_ID")["TS_Pct"]
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["Cumulative_Minutes"] = df.groupby("Player_ID")["Minutes"].cumsum()

    # Fill NaN in rolling/context with season avg or 0
    season_ts_mean = df.groupby(["Player_ID", "Season_Number"])["TS_Pct"].transform("mean")
    df["Rolling_5G_TS"] = df["Rolling_5G_TS"].fillna(season_ts_mean).fillna(0.0)

    for c in ["Rest_Days", "Is_B2B", "Cumulative_Minutes", "Minutes", "TS_Pct"]:
        df[c] = df[c].fillna(0.0)

    return df


def build_baseline_aggregates(fe_df: pd.DataFrame) -> pd.DataFrame:
    """Flatten sequence into player-level aggregates for baseline regression."""
    if fe_df.empty:
        return fe_df.copy()

    grouped = (
        fe_df.groupby(["Player_ID", "Player_Name", "Draft_Year", "Season_Number"], as_index=False)
        .agg(
            Avg_Minutes=("Minutes", "mean"),
            Avg_TS=("TS_Pct", "mean"),
            B2B_Count=("Is_B2B", "sum"),
            Total_Minutes=("Minutes", "sum"),
            Games=("Game_Number", "max"),
            Y5_Minutes=("Y5_Minutes", "first"),
            Y5_WinShares=("Y5_WinShares", "first"),
        )
    )

    wide = grouped.pivot_table(
        index=["Player_ID", "Player_Name", "Draft_Year", "Y5_Minutes", "Y5_WinShares"],
        columns="Season_Number",
        values=["Avg_Minutes", "Avg_TS", "B2B_Count", "Total_Minutes", "Games"],
    )

    wide.columns = [f"{m}_S{s}" for m, s in wide.columns]
    wide = wide.reset_index()
    return wide
