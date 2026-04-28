from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
KMEANS_DIR = PROJECT_ROOT / "kmeans_k5_outputs_split"
SHOT_OUTPUT_DIR = PROJECT_ROOT / "Output" / "ShotChartDetail"
ARCHETYPE_OUTPUT_DIR = PROJECT_ROOT / "Output" / "Player Archetype Analysis"
ARCHETYPE_VISUAL_DIR = PROJECT_ROOT / "visual" / "Player Archetype"

DEFAULT_LOG_PATH = ARCHETYPE_OUTPUT_DIR / "player_archetype_analysis_workflow_log.txt"


@dataclass(frozen=True)
class Paths:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    kmeans_dir: Path = KMEANS_DIR
    shot_output_dir: Path = SHOT_OUTPUT_DIR
    archetype_output_dir: Path = ARCHETYPE_OUTPUT_DIR
    archetype_visual_dir: Path = ARCHETYPE_VISUAL_DIR
    log_path: Path = DEFAULT_LOG_PATH


PATHS = Paths()


def ensure_dirs() -> None:
    PATHS.archetype_output_dir.mkdir(parents=True, exist_ok=True)
    PATHS.archetype_visual_dir.mkdir(parents=True, exist_ok=True)


def append_log(phase: str, completed: str, learned: str, assumptions: str, files_read: Sequence[str], files_written: Sequence[str]) -> None:
    ensure_dirs()
    with PATHS.log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 80}\n")
        f.write(f"{phase}\n")
        f.write(f"{'=' * 80}\n")
        f.write("1. What was completed\n")
        f.write(f"{completed.strip()}\n\n")
        f.write("2. What was learned\n")
        f.write(f"{learned.strip()}\n\n")
        f.write("3. What assumptions still matter\n")
        f.write(f"{assumptions.strip()}\n\n")
        f.write("4. What files were read\n")
        for item in files_read:
            f.write(f"- {item}\n")
        f.write("\n5. What files were written\n")
        for item in files_written:
            f.write(f"- {item}\n")
        f.write("\n")


def reset_log(objective_text: str, resolved_paths: dict[str, str]) -> None:
    ensure_dirs()
    lines = [
        "Player Archetype Analysis Workflow Log",
        "",
        "Project objective",
        objective_text.strip(),
        "",
        "Resolved paths",
    ]
    for k, v in resolved_paths.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    PATHS.log_path.write_text("\n".join(lines), encoding="utf-8")


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def safe_player_name_merge(left: pd.DataFrame, right: pd.DataFrame, on: Sequence[str], how: str = "left") -> pd.DataFrame:
    return left.merge(right, on=list(on), how=how)


def coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def normalize_name(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def add_name_key(df: pd.DataFrame, name_col: str) -> pd.DataFrame:
    out = df.copy()
    out["name_key"] = out[name_col].map(normalize_name)
    return out


def embedding_columns(df: pd.DataFrame, prefix: str = "shot_emb_") -> list[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    return sorted(cols, key=lambda x: int(x.split("_")[-1]))


def zscore_with_reference(df: pd.DataFrame, cols: Sequence[str], reference: pd.DataFrame | None = None) -> pd.DataFrame:
    ref = df if reference is None else reference
    means = ref[list(cols)].mean()
    stds = ref[list(cols)].std(ddof=0).replace(0, np.nan)
    out = df.copy()
    out[list(cols)] = (out[list(cols)] - means) / stds
    return out


def weighted_mean_matrix(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    values = np.asarray(values, dtype=float)
    denom = np.sum(weights)
    if denom <= 0:
        return np.nanmean(values, axis=0)
    return np.average(values, axis=0, weights=weights)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
