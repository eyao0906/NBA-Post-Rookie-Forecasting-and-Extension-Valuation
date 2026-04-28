from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from archetype_workflow_utils import PATHS, append_log, ensure_dirs

X_MIN, X_MAX = -250.0, 250.0
Y_MIN, Y_MAX = -60.0, 470.0
GRID_SIZE = 25
SIGMA = 1.2
MIN_SHOTS = 25


def load_shots(path: Path, cohort_path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if cohort_path is not None:
        cohort = pd.read_csv(cohort_path).rename(columns={"PERSON_ID": "PLAYER_ID", "SEASON": "DRAFT_YEAR"})
        cohort["PLAYER_ID"] = pd.to_numeric(cohort["PLAYER_ID"], errors="coerce")
        df["PLAYER_ID"] = pd.to_numeric(df["PLAYER_ID"], errors="coerce")
        df = df.merge(cohort[["PLAYER_ID", "PLAYER_NAME", "DRAFT_YEAR"]].rename(columns={"PLAYER_NAME": "COHORT_PLAYER_NAME"}), on="PLAYER_ID", how="left")
    else:
        df["COHORT_PLAYER_NAME"] = df["PLAYER_NAME"]
        df["DRAFT_YEAR"] = pd.to_numeric(df.get("draft_year"), errors="coerce")
    for col in ["PLAYER_ID", "season_num", "LOC_X", "LOC_Y", "SHOT_MADE_FLAG", "SHOT_ATTEMPTED_FLAG", "GAME_ID", "GAME_EVENT_ID"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["SHOT_ATTEMPTED_FLAG"] == 1].copy()
    df = df[df["season_num"].between(1, 4, inclusive="both")].copy()
    df = df[df["LOC_X"].between(X_MIN, X_MAX, inclusive="both") & df["LOC_Y"].between(Y_MIN, Y_MAX, inclusive="both")].copy()
    df = df.sort_values(["PLAYER_ID", "season_num", "GAME_ID", "GAME_EVENT_ID"]).drop_duplicates(["PLAYER_ID", "season_num", "GAME_ID", "GAME_EVENT_ID"])
    return df


def build_tensor(group: pd.DataFrame, x_edges: np.ndarray, y_edges: np.ndarray) -> np.ndarray:
    x = group["LOC_X"].to_numpy(dtype=np.float32)
    y = group["LOC_Y"].to_numpy(dtype=np.float32)
    made = group["SHOT_MADE_FLAG"].to_numpy(dtype=np.float32)
    attempts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    makes, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=made)
    attempts = attempts.T.astype(np.float32)
    makes = makes.T.astype(np.float32)
    n = max(len(group), 1)
    attempt_density = attempts / n
    made_density = makes / n
    smooth_attempts = gaussian_filter(attempts, sigma=SIGMA, mode="constant")
    smooth_makes = gaussian_filter(makes, sigma=SIGMA, mode="constant")
    make_rate = np.divide(smooth_makes, np.maximum(smooth_attempts, 1e-8), out=np.zeros_like(smooth_makes), where=smooth_attempts > 0)
    return np.stack([attempt_density, made_density, make_rate], axis=0).astype(np.float32)


def export_bundle(df: pd.DataFrame, cohort_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_edges = np.linspace(X_MIN, X_MAX, GRID_SIZE + 1)
    y_edges = np.linspace(Y_MIN, Y_MAX, GRID_SIZE + 1)

    counts = (
        df.groupby(["PLAYER_ID", "COHORT_PLAYER_NAME", "DRAFT_YEAR", "season_num", "season_string"], dropna=False)
        .agg(games_with_shots=("GAME_ID", "nunique"), shot_attempts=("SHOT_ATTEMPTED_FLAG", "sum"), shot_makes=("SHOT_MADE_FLAG", "sum"))
        .reset_index()
    )
    counts["fg_pct"] = counts["shot_makes"] / counts["shot_attempts"].replace(0, np.nan)
    counts["eligible_for_tensor"] = (counts["shot_attempts"] >= MIN_SHOTS).astype(int)

    rows, tensors = [], []
    tensor_idx = 0
    for keys, grp in df.groupby(["PLAYER_ID", "COHORT_PLAYER_NAME", "DRAFT_YEAR", "season_num", "season_string"], dropna=False):
        info = counts[(counts["PLAYER_ID"] == keys[0]) & (counts["season_num"] == keys[3])].iloc[0]
        if int(info["eligible_for_tensor"]) != 1:
            continue
        tensor = build_tensor(grp, x_edges, y_edges)
        tensors.append(tensor)
        rows.append(
            {
                "tensor_index": tensor_idx,
                "PLAYER_ID": keys[0],
                "PLAYER_NAME": keys[1],
                "DRAFT_YEAR": keys[2],
                "season_num": keys[3],
                "season_string": keys[4],
                "games_with_shots": int(info["games_with_shots"]),
                "shot_attempts": int(info["shot_attempts"]),
                "shot_makes": int(info["shot_makes"]),
                "fg_pct": float(info["fg_pct"]),
                "cohort_label": cohort_label,
            }
        )
        tensor_idx += 1

    tensor_index = pd.DataFrame(rows)
    tensor_arr = np.stack(tensors, axis=0) if tensors else np.empty((0, 3, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    np.savez_compressed(PATHS.archetype_output_dir / f"{cohort_label}_player_season_shot_tensors.npz", tensors=tensor_arr)
    tensor_index.to_csv(PATHS.archetype_output_dir / f"{cohort_label}_player_shot_tensor_index.csv", index=False)
    return counts, tensor_index


def aggregate_player_tensor(index_df: pd.DataFrame, tensor_npz_path: Path, out_npz_path: Path, cohort_label: str) -> pd.DataFrame:
    tensor_file = np.load(tensor_npz_path)
    tensors = tensor_file["tensors"]
    if len(index_df) == 0:
        np.savez_compressed(out_npz_path, tensors=np.empty((0, 3, GRID_SIZE, GRID_SIZE), dtype=np.float32))
        return pd.DataFrame()
    player_rows = []
    player_tensors = []
    player_tensor_index = 0
    for player_id, grp in index_df.groupby("PLAYER_ID"):
        weights = grp["shot_attempts"].to_numpy(dtype=float)
        player_tensor = np.average(tensors[grp["tensor_index"].to_numpy()], axis=0, weights=weights).astype(np.float32)
        player_tensors.append(player_tensor)
        player_rows.append(
            {
                "player_tensor_index": player_tensor_index,
                "PLAYER_ID": player_id,
                "PLAYER_NAME": grp["PLAYER_NAME"].iloc[0],
                "DRAFT_YEAR": grp["DRAFT_YEAR"].iloc[0],
                "seasons_with_tensors": int(grp["season_num"].nunique()),
                "total_shot_attempts_covered": float(grp["shot_attempts"].sum()),
                "season_tensor_source_path": str(tensor_npz_path),
                "player_tensor_path": str(out_npz_path),
                "player_tensor_index_count": int(len(grp)),
                "cohort_label": cohort_label,
                "contains_actual_player_tensor": True,
            }
        )
        player_tensor_index += 1
    np.savez_compressed(out_npz_path, tensors=np.stack(player_tensors, axis=0))
    return pd.DataFrame(player_rows)


def main() -> None:
    ensure_dirs()
    draft_shots = load_shots(PATHS.data_dir / "Shot Chart Details Raw" / "raw_shotchart_S1_to_S4_main.csv", PATHS.data_dir / "cohort_1999_2019.csv")
    hof_shots = load_shots(PATHS.data_dir / "Shot Chart Details Raw" / "raw_shotchart_S1_to_S4_hof_shotstyle.csv", PATHS.data_dir / "cohort_HOF.csv")

    draft_counts, draft_index = export_bundle(draft_shots, "draft")
    hof_counts, hof_index = export_bundle(hof_shots, "hof")

    shot_tensor_player_season = pd.concat([draft_index.assign(pool="draft"), hof_index.assign(pool="hof")], ignore_index=True)
    draft_player_tensor_npz = PATHS.archetype_output_dir / "draft_player_tensors.npz"
    hof_player_tensor_npz = PATHS.archetype_output_dir / "hof_player_tensors.npz"
    shot_tensor_player = pd.concat(
        [
            aggregate_player_tensor(draft_index, PATHS.archetype_output_dir / "draft_player_season_shot_tensors.npz", draft_player_tensor_npz, "draft").assign(pool="draft"),
            aggregate_player_tensor(hof_index, PATHS.archetype_output_dir / "hof_player_season_shot_tensors.npz", hof_player_tensor_npz, "hof").assign(pool="hof"),
        ],
        ignore_index=True,
    )

    shot_cov = draft_counts.copy()
    shot_cov["excluded_from_embedding_training_low_volume"] = 1 - shot_cov["eligible_for_tensor"]
    hof_cov = hof_counts.copy()
    hof_cov["excluded_from_embedding_training_low_volume"] = 1 - hof_cov["eligible_for_tensor"]

    season_path = PATHS.archetype_output_dir / "shot_tensor_player_season.parquet"
    player_path = PATHS.archetype_output_dir / "shot_tensor_player.parquet"
    shot_cov_path = PATHS.archetype_output_dir / "shot_coverage_audit.csv"
    hof_cov_path = PATHS.archetype_output_dir / "hof_shot_coverage_audit.csv"
    meta_path = PATHS.archetype_output_dir / "shot_tensor_metadata.json"

    shot_tensor_player_season.to_parquet(season_path, index=False)
    shot_tensor_player.to_parquet(player_path, index=False)
    shot_cov.to_csv(shot_cov_path, index=False)
    hof_cov.to_csv(hof_cov_path, index=False)
    meta_path.write_text(json.dumps({"grid_size": GRID_SIZE, "x_range": [X_MIN, X_MAX], "y_range": [Y_MIN, Y_MAX], "sigma": SIGMA, "min_shots": MIN_SHOTS}, indent=2), encoding="utf-8")

    append_log(
        phase="PHASE 3 — BUILD SHOT-STYLE SPATIAL TENSORS",
        completed=(
            "Read the restored drafted-player and Hall of Fame shot-chart files, cleaned coordinates on a fixed half-court grid, built three-channel spatial tensors, exported player-season tensor metadata, and also saved real aggregated player-level tensor artifacts. "
            "Also created shot-volume coverage audits and explicit low-volume exclusion flags."
        ),
        learned=(
            "Both drafted-player and HOF shot charts can now be processed in a common coordinate system. "
            "A minimum-shot rule is necessary because some player-seasons remain too sparse for stable spatial embeddings."
        ),
        assumptions=(
            "The fixed 25x25 half-court grid and Gaussian smoothing parameter are inherited from the project’s prior shot-style work. "
            "Player-level tensors are descriptive rollups; training still happens at the player-season level."
        ),
        files_read=[
            str(PATHS.data_dir / "Shot Chart Details Raw" / "raw_shotchart_S1_to_S4_main.csv"),
            str(PATHS.data_dir / "Shot Chart Details Raw" / "raw_shotchart_S1_to_S4_hof_shotstyle.csv"),
            str(PATHS.data_dir / "cohort_1999_2019.csv"),
            str(PATHS.data_dir / "cohort_HOF.csv"),
        ],
        files_written=[str(season_path), str(player_path), str(shot_cov_path), str(hof_cov_path), str(meta_path), str(draft_player_tensor_npz), str(hof_player_tensor_npz)],
    )


if __name__ == "__main__":
    main()
