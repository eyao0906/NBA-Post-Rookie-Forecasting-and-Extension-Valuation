"""Stage 5: Baseline tabular model (Ridge regression default)."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from config import PipelineConfig


def _train_one_target(df: pd.DataFrame, target_col: str, cfg: PipelineConfig) -> Tuple[Dict, pd.DataFrame]:
    clean = df.dropna(subset=[target_col]).copy()
    id_cols = ["Player_ID", "Player_Name", "Draft_Year", "Y5_Minutes", "Y5_WinShares"]
    feature_cols = [c for c in clean.columns if c not in id_cols]

    if clean.empty or not feature_cols:
        return {"target": target_col, "rmse": np.nan, "mae": np.nan}, pd.DataFrame()

    X = clean[feature_cols].fillna(0.0).values
    y = clean[target_col].values

    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X,
        y,
        clean.index.values,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
    )

    model = Ridge(alpha=1.0)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    metrics = {
        "target": target_col,
        "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
        "mae": float(mean_absolute_error(y_te, pred)),
    }

    pred_df = clean.loc[idx_te, ["Player_ID", "Player_Name", target_col]].copy()
    pred_df["Baseline_Pred"] = pred
    pred_df = pred_df.rename(columns={target_col: "Actual"})
    pred_df["Target"] = target_col
    return metrics, pred_df


def train_baseline(df_wide: pd.DataFrame, cfg: PipelineConfig):
    m_minutes, p_minutes = _train_one_target(df_wide, "Y5_Minutes", cfg)
    m_ws, p_ws = _train_one_target(df_wide, "Y5_WinShares", cfg)

    metrics = pd.DataFrame([m_minutes, m_ws])
    preds = pd.concat([p_minutes, p_ws], ignore_index=True) if not p_minutes.empty or not p_ws.empty else pd.DataFrame()
    return metrics, preds
