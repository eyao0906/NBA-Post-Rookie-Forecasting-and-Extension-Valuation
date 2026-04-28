"""Stage 6: LSTM sequence model with GPU-first execution."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import PipelineConfig


SEQUENCE_FEATURES = [
    "Minutes",
    "PTS",
    "REB",
    "AST",
    "TS_Pct",
    "Rest_Days",
    "Is_B2B",
    "Rolling_5G_TS",
    "Cumulative_Minutes",
]


def configure_gpu(require_gpu: bool = True) -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if require_gpu and not gpus:
        raise RuntimeError("No GPU detected. Re-run with GPU available, or pass require_gpu=False.")

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def _pad_player_sequence(player_df: pd.DataFrame, max_len: int, feature_cols: List[str]) -> np.ndarray:
    arr = player_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    out = np.zeros((max_len, len(feature_cols)), dtype=np.float32)
    n = min(len(arr), max_len)
    out[:n, :] = arr[:n, :]
    return out


def build_lstm_tensors(fe_df: pd.DataFrame, max_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    # Keep only players with available targets for each output
    player_targets = (
        fe_df.groupby(["Player_ID", "Player_Name"], as_index=False)
        .agg(Y5_Minutes=("Y5_Minutes", "first"), Y5_WinShares=("Y5_WinShares", "first"))
    )

    players = player_targets["Player_ID"].tolist()
    seqs = []

    for pid in players:
        p = fe_df[fe_df["Player_ID"] == pid].sort_values("GAME_DATE")
        for c in SEQUENCE_FEATURES:
            if c not in p.columns:
                p[c] = 0.0
        seqs.append(_pad_player_sequence(p, max_len=max_len, feature_cols=SEQUENCE_FEATURES))

    X = np.stack(seqs, axis=0) if seqs else np.empty((0, max_len, len(SEQUENCE_FEATURES)))
    y_min = player_targets["Y5_Minutes"].to_numpy(dtype=np.float32)
    y_ws = player_targets["Y5_WinShares"].to_numpy(dtype=np.float32)
    return X, y_min, y_ws, player_targets


def _make_model(timesteps: int, n_features: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(timesteps, n_features)),
            tf.keras.layers.Masking(mask_value=0.0),
            tf.keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.0),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def _train_one_output(X: np.ndarray, y: np.ndarray, cfg: PipelineConfig) -> Dict:
    valid = ~np.isnan(y)
    Xv = X[valid]
    yv = y[valid]
    if len(yv) < 10:
        return {"rmse": np.nan, "mae": np.nan, "y_test": np.array([]), "pred": np.array([]), "idx_test": np.array([])}

    idx = np.arange(len(yv))
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        Xv, yv, idx, test_size=cfg.test_size, random_state=cfg.random_state
    )

    # Scale features across all timesteps by flattening then restoring shape
    scaler = StandardScaler()
    X_tr_2d = X_tr.reshape(-1, X_tr.shape[-1])
    X_te_2d = X_te.reshape(-1, X_te.shape[-1])
    X_tr_s = scaler.fit_transform(X_tr_2d).reshape(X_tr.shape)
    X_te_s = scaler.transform(X_te_2d).reshape(X_te.shape)

    model = _make_model(timesteps=X_tr.shape[1], n_features=X_tr.shape[2])
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)]

    model.fit(
        X_tr_s,
        y_tr,
        validation_split=0.2,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        verbose=0,
        callbacks=cb,
    )

    pred = model.predict(X_te_s, verbose=0).reshape(-1)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
        "mae": float(mean_absolute_error(y_te, pred)),
        "y_test": y_te,
        "pred": pred,
        "idx_test": idx_te,
        "model": model,
    }


def train_lstm(fe_df: pd.DataFrame, cfg: PipelineConfig, require_gpu: bool = True):
    configure_gpu(require_gpu=require_gpu)
    X, y_min, y_ws, player_targets = build_lstm_tensors(fe_df, max_len=cfg.max_games_first4)

    m_min = _train_one_output(X, y_min, cfg)
    m_ws = _train_one_output(X, y_ws, cfg)

    metrics = pd.DataFrame(
        [
            {"target": "Y5_Minutes", "rmse": m_min["rmse"], "mae": m_min["mae"]},
            {"target": "Y5_WinShares", "rmse": m_ws["rmse"], "mae": m_ws["mae"]},
        ]
    )

    preds = []
    for target, out in [("Y5_Minutes", m_min), ("Y5_WinShares", m_ws)]:
        if len(out.get("idx_test", [])) == 0:
            continue
        tmp = player_targets.iloc[out["idx_test"]][["Player_ID", "Player_Name"]].copy()
        tmp["Target"] = target
        tmp["Actual"] = out["y_test"]
        tmp["LSTM_Pred"] = out["pred"]
        preds.append(tmp)

    pred_df = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    models = {"minutes_model": m_min.get("model"), "winshares_model": m_ws.get("model")}
    return metrics, pred_df, models
