import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover
    gaussian_filter = None


X_MIN_DEFAULT = -250.0
X_MAX_DEFAULT = 250.0
Y_MIN_DEFAULT = -60.0
Y_MAX_DEFAULT = 470.0
EPS = 1e-8


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists():
            return candidate
    return start.parent


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "data/Shot Feature"

def coerce_bool_flag(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
    return (out > 0).astype(np.int8)


def load_raw_shots(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    rename_map = {}
    if "draft_year" in df.columns and "DRAFT_YEAR" not in df.columns:
        rename_map["draft_year"] = "DRAFT_YEAR"
    if "season_string" in df.columns and "SEASON_STRING" not in df.columns:
        rename_map["season_string"] = "SEASON_STRING"
    if "PLAYER_NAME" not in df.columns and "COHORT_PLAYER_NAME" in df.columns:
        rename_map["COHORT_PLAYER_NAME"] = "PLAYER_NAME"
    if rename_map:
        df = df.rename(columns=rename_map)

    required = [
        "PLAYER_ID", "PLAYER_NAME", "season_num", "LOC_X", "LOC_Y",
        "SHOT_MADE_FLAG", "SHOT_ATTEMPTED_FLAG",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in raw shot file: {missing}")

    numeric_cols = [
        "PLAYER_ID", "season_num", "LOC_X", "LOC_Y", "SHOT_MADE_FLAG",
        "SHOT_ATTEMPTED_FLAG", "GAME_ID", "GAME_EVENT_ID", "SHOT_DISTANCE",
        "PERIOD", "MINUTES_REMAINING", "SECONDS_REMAINING", "DRAFT_YEAR",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["PLAYER_ID"] = df["PLAYER_ID"].astype("Int64")
    df["season_num"] = df["season_num"].astype("Int64")
    df["SHOT_MADE_FLAG"] = coerce_bool_flag(df["SHOT_MADE_FLAG"])
    df["SHOT_ATTEMPTED_FLAG"] = coerce_bool_flag(df["SHOT_ATTEMPTED_FLAG"])

    return df


def filter_halfcourt(
    df: pd.DataFrame,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> pd.DataFrame:
    out = df.copy()
    out = out[out["SHOT_ATTEMPTED_FLAG"] == 1].copy()
    out = out[out["season_num"].between(1, 4, inclusive="both")].copy()
    out = out[out["PLAYER_ID"].notna()].copy()
    out = out[out["LOC_X"].between(x_min, x_max, inclusive="both")].copy()
    out = out[out["LOC_Y"].between(y_min, y_max, inclusive="both")].copy()

    sort_cols = [c for c in ["PLAYER_ID", "season_num", "GAME_ID", "GAME_EVENT_ID"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols)

    if {"PLAYER_ID", "season_num", "GAME_ID", "GAME_EVENT_ID"}.issubset(out.columns):
        out = out.drop_duplicates(subset=["PLAYER_ID", "season_num", "GAME_ID", "GAME_EVENT_ID"]).copy()

    return out


def safe_mode(series: pd.Series):
    if series.empty:
        return np.nan
    m = series.mode(dropna=True)
    return m.iloc[0] if not m.empty else np.nan


def make_shot_counts(df: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "PLAYER_NAME": ("PLAYER_NAME", "first"),
        "shot_attempts": ("SHOT_ATTEMPTED_FLAG", "sum"),
        "shot_makes": ("SHOT_MADE_FLAG", "sum"),
        "avg_loc_x": ("LOC_X", "mean"),
        "avg_loc_y": ("LOC_Y", "mean"),
    }
    if "SEASON_STRING" in df.columns:
        agg["season_string"] = ("SEASON_STRING", safe_mode)
    if "DRAFT_YEAR" in df.columns:
        agg["DRAFT_YEAR"] = ("DRAFT_YEAR", safe_mode)
    if "GAME_ID" in df.columns:
        agg["games_with_shots"] = ("GAME_ID", "nunique")

    counts = (
        df.groupby(["PLAYER_ID", "season_num"], dropna=False)
          .agg(**agg)
          .reset_index()
          .sort_values(["PLAYER_ID", "season_num"])
          .reset_index(drop=True)
    )
    counts["fg_pct"] = counts["shot_makes"] / counts["shot_attempts"].replace(0, np.nan)
    return counts


def _smoothed_make_rate(attempt_counts: np.ndarray, made_counts: np.ndarray, sigma: float) -> np.ndarray:
    if gaussian_filter is None:
        # Fallback: unsmoothed local rate.
        return np.divide(made_counts, np.maximum(attempt_counts, EPS), out=np.zeros_like(made_counts, dtype=np.float32), where=attempt_counts > 0)

    smooth_attempts = gaussian_filter(attempt_counts.astype(np.float32), sigma=sigma, mode="constant")
    smooth_makes = gaussian_filter(made_counts.astype(np.float32), sigma=sigma, mode="constant")
    return np.divide(
        smooth_makes,
        np.maximum(smooth_attempts, EPS),
        out=np.zeros_like(smooth_makes, dtype=np.float32),
        where=smooth_attempts > 0,
    )


def build_tensor_for_group(
    group: pd.DataFrame,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    sigma: float,
) -> np.ndarray:
    x = group["LOC_X"].to_numpy(dtype=np.float32)
    y = group["LOC_Y"].to_numpy(dtype=np.float32)
    made = group["SHOT_MADE_FLAG"].to_numpy(dtype=np.float32)

    # numpy.histogram2d returns [x_bin, y_bin]; transpose so output is [y, x].
    attempt_counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    made_counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=made)
    attempt_counts = attempt_counts.T.astype(np.float32)
    made_counts = made_counts.T.astype(np.float32)

    n_shots = float(max(len(group), 1))
    attempt_density = attempt_counts / n_shots
    made_density = made_counts / n_shots
    make_rate_surface = _smoothed_make_rate(attempt_counts, made_counts, sigma=sigma)

    tensor = np.stack([attempt_density, made_density, make_rate_surface], axis=0).astype(np.float32)
    return tensor


def iter_groups(df: pd.DataFrame) -> Iterable[tuple[tuple[int, int], pd.DataFrame]]:
    grouped = df.groupby(["PLAYER_ID", "season_num"], sort=True, dropna=False)
    for key, grp in grouped:
        yield key, grp


def save_outputs(
    kept_index: pd.DataFrame,
    all_counts: pd.DataFrame,
    tensors: np.ndarray,
    out_dir: Path,
    grid_size: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    sigma: float,
    min_shots: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    kept_index.to_csv(out_dir / "player_shot_tensor_index.csv", index=False)
    all_counts.to_csv(out_dir / "player_season_shot_counts.csv", index=False)

    np.savez_compressed(
        out_dir / "player_season_shot_tensors.npz",
        tensors=tensors,
        tensor_index=kept_index["tensor_index"].to_numpy(dtype=np.int32),
        player_id=kept_index["PLAYER_ID"].to_numpy(dtype=np.int64),
        season_num=kept_index["season_num"].to_numpy(dtype=np.int16),
    )

    metadata = {
        "n_player_seasons_exported": int(len(kept_index)),
        "tensor_shape": [int(v) for v in tensors.shape],
        "channels": [
            "attempt_density_sum_to_1",
            "made_density_sum_to_fg_pct",
            "smoothed_local_make_rate_surface",
        ],
        "grid_size": int(grid_size),
        "x_range": [float(x_min), float(x_max)],
        "y_range": [float(y_min), float(y_max)],
        "gaussian_sigma": float(sigma),
        "min_shots_for_export": int(min_shots),
    }
    (out_dir / "shot_tensor_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build player-season spatial shot tensors from raw NBA shot chart data.")
    parser.add_argument("--raw-shots", type=Path, default=DATA_DIR / "raw_shotchart_S1_to_S4_main.csv")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR / "shot_tensors")
    parser.add_argument("--grid-size", type=int, default=25)
    parser.add_argument("--x-min", type=float, default=X_MIN_DEFAULT)
    parser.add_argument("--x-max", type=float, default=X_MAX_DEFAULT)
    parser.add_argument("--y-min", type=float, default=Y_MIN_DEFAULT)
    parser.add_argument("--y-max", type=float, default=Y_MAX_DEFAULT)
    parser.add_argument("--smoothing-sigma", type=float, default=1.2)
    parser.add_argument("--min-shots", type=int, default=25, help="Only export tensors for player-seasons meeting this shot threshold.")
    args = parser.parse_args()

    if args.grid_size < 5:
        raise ValueError("grid-size must be at least 5.")
    if args.x_min >= args.x_max or args.y_min >= args.y_max:
        raise ValueError("Coordinate ranges are invalid.")

    raw = load_raw_shots(args.raw_shots)
    filtered = filter_halfcourt(raw, args.x_min, args.x_max, args.y_min, args.y_max)
    counts = make_shot_counts(filtered)
    counts["eligible_for_tensor"] = (counts["shot_attempts"] >= args.min_shots).astype(int)

    x_edges = np.linspace(args.x_min, args.x_max, args.grid_size + 1)
    y_edges = np.linspace(args.y_min, args.y_max, args.grid_size + 1)

    rows = []
    tensors = []
    tensor_index = 0

    count_lookup = counts.set_index(["PLAYER_ID", "season_num"])
    for (player_id, season_num), grp in iter_groups(filtered):
        count_row = count_lookup.loc[(player_id, season_num)]
        if int(count_row["eligible_for_tensor"]) != 1:
            continue

        tensor = build_tensor_for_group(grp, x_edges=x_edges, y_edges=y_edges, sigma=args.smoothing_sigma)
        rows.append({
            "tensor_index": tensor_index,
            "PLAYER_ID": int(player_id),
            "season_num": int(season_num),
            "PLAYER_NAME": count_row.get("PLAYER_NAME", np.nan),
            "season_string": count_row.get("season_string", np.nan),
            "DRAFT_YEAR": count_row.get("DRAFT_YEAR", np.nan),
            "games_with_shots": int(count_row.get("games_with_shots", np.nan)) if pd.notna(count_row.get("games_with_shots", np.nan)) else np.nan,
            "shot_attempts": int(count_row["shot_attempts"]),
            "shot_makes": int(count_row["shot_makes"]),
            "fg_pct": float(count_row["fg_pct"]) if pd.notna(count_row["fg_pct"]) else np.nan,
        })
        tensors.append(tensor)
        tensor_index += 1

    kept_index = pd.DataFrame(rows)
    if not tensors:
        raise RuntimeError("No tensors were created. Lower --min-shots or inspect the raw data filters.")

    tensor_arr = np.stack(tensors, axis=0).astype(np.float32)
    save_outputs(
        kept_index=kept_index,
        all_counts=counts,
        tensors=tensor_arr,
        out_dir=args.output_dir,
        grid_size=args.grid_size,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        sigma=args.smoothing_sigma,
        min_shots=args.min_shots,
    )

    print(f"Raw rows loaded: {len(raw):,}")
    print(f"Rows after half-court and Seasons 1-4 filters: {len(filtered):,}")
    print(f"Player-seasons available after filtering: {counts.shape[0]:,}")
    print(f"Player-seasons exported as tensors: {len(kept_index):,}")
    print(f"Tensor array shape: {tensor_arr.shape}")
    print(f"Saved outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
