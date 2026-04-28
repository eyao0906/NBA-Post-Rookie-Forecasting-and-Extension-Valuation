from __future__ import annotations

import numpy as np
import pandas as pd

from archetype_workflow_utils import PATHS, append_log, ensure_dirs, embedding_columns, zscore_with_reference

BOXSPACE_COLS = [
    "pts_per36",
    "reb_per36",
    "ast_per36",
    "stl_per36",
    "blk_per36",
    "tov_per36",
    "fga_per36",
    "fta_per36",
    "usage_proxy_per36",
    "ts_pct",
    "fg3a_rate",
    "ftr",
    "mpg",
]


def path_length(mat: np.ndarray) -> float:
    if len(mat) <= 1:
        return np.nan
    return float(np.sum(np.linalg.norm(np.diff(mat, axis=0), axis=1)))


def classify_drift(displacement: float, path_len: float, subtype_changes: float) -> str:
    if np.isnan(displacement):
        return "insufficient_data"
    if displacement < 1.0 and (np.isnan(subtype_changes) or subtype_changes == 0):
        return "stable"
    if displacement < 2.0 and (np.isnan(subtype_changes) or subtype_changes <= 1):
        return "evolving_gradually"
    return "role_shifting_materially"


def main() -> None:
    ensure_dirs()
    season_box = pd.read_csv(PATHS.data_dir / "player_season_feature_table_1999_2019.csv")
    season_shot = pd.read_csv(PATHS.archetype_output_dir / "shot_embedding_player_season.csv")
    season_subtypes = pd.read_csv(PATHS.archetype_output_dir / "shot_style_player_season_table.csv")
    hybrid = pd.read_csv(PATHS.archetype_output_dir / "player_hybrid_archetype_table.csv")

    box_df = season_box[["Player_ID", "COHORT_PLAYER_NAME", "DRAFT_YEAR", "season_num"] + BOXSPACE_COLS].rename(columns={"Player_ID": "PLAYER_ID", "COHORT_PLAYER_NAME": "PLAYER_NAME"})
    box_df = zscore_with_reference(box_df, BOXSPACE_COLS)

    box_rows, shot_rows, identity_rows = [], [], []
    shot_emb_cols = embedding_columns(season_shot)
    subtype_lookup = season_subtypes[["PLAYER_ID", "season_num", "shot_style_subtype_id", "shot_style_subtype"]].copy()

    for player_id, grp in box_df.groupby("PLAYER_ID"):
        grp = grp.sort_values("season_num")
        y1 = grp[grp["season_num"] == 1]
        y4 = grp[grp["season_num"] == 4]
        mat = grp[BOXSPACE_COLS].to_numpy(dtype=float)
        disp = np.nan
        if len(y1) and len(y4):
            disp = float(np.linalg.norm(y4[BOXSPACE_COLS].to_numpy(dtype=float)[0] - y1[BOXSPACE_COLS].to_numpy(dtype=float)[0]))
        p_len = path_length(mat)
        box_rows.append({
            "PLAYER_ID": player_id,
            "PLAYER_NAME": grp["PLAYER_NAME"].iloc[0],
            "draft_year": grp["DRAFT_YEAR"].iloc[0],
            "box_role_y1_to_y4_displacement": disp,
            "box_role_total_path_length": p_len,
            "observed_boxscore_seasons": int(grp["season_num"].nunique()),
        })

    for player_id, grp in season_shot.groupby("PLAYER_ID"):
        grp = grp.sort_values("season_num")
        y1 = grp[grp["season_num"] == 1]
        y4 = grp[grp["season_num"] == 4]
        mat = grp[shot_emb_cols].to_numpy(dtype=float)
        disp = np.nan
        if len(y1) and len(y4):
            disp = float(np.linalg.norm(y4[shot_emb_cols].to_numpy(dtype=float)[0] - y1[shot_emb_cols].to_numpy(dtype=float)[0]))
        p_len = path_length(mat)
        shot_rows.append({
            "PLAYER_ID": player_id,
            "shotstyle_y1_to_y4_displacement": disp,
            "shotstyle_total_path_length": p_len,
            "observed_shot_seasons": int(grp["season_num"].nunique()),
        })

    box_summary = pd.DataFrame(box_rows)
    shot_summary = pd.DataFrame(shot_rows)
    identity = hybrid.merge(box_summary, on=["PLAYER_ID", "PLAYER_NAME", "draft_year"], how="left") if "draft_year" in hybrid.columns else hybrid.merge(box_summary, on=["PLAYER_ID", "PLAYER_NAME"], how="left")
    identity = identity.merge(shot_summary, on="PLAYER_ID", how="left")

    subtype_change_counts = season_shot.merge(subtype_lookup, on=["PLAYER_ID", "season_num"], how="left", validate="one_to_one")
    subtype_change_counts = (
        subtype_change_counts.sort_values(["PLAYER_ID", "season_num"])
        .groupby("PLAYER_ID")
        .agg(shot_style_subtype_changes=("shot_style_subtype_id", lambda s: int((s.ffill().diff().fillna(0) != 0).sum())))
        .reset_index()
    )
    identity = identity.merge(subtype_change_counts, on="PLAYER_ID", how="left")
    def _row_drift_class(r: pd.Series) -> str:
        disp_vals = np.array([r.get("box_role_y1_to_y4_displacement", np.nan), r.get("shotstyle_y1_to_y4_displacement", np.nan)], dtype=float)
        path_vals = np.array([r.get("box_role_total_path_length", np.nan), r.get("shotstyle_total_path_length", np.nan)], dtype=float)
        disp = np.nan if np.all(np.isnan(disp_vals)) else float(np.nanmax(disp_vals))
        path_len = np.nan if np.all(np.isnan(path_vals)) else float(np.nanmax(path_vals))
        return classify_drift(disp, path_len, r.get("shot_style_subtype_changes", np.nan))

    identity["identity_drift_class"] = identity.apply(_row_drift_class, axis=1)

    box_path = PATHS.archetype_output_dir / "player_role_drift_summary.csv"
    shot_path = PATHS.archetype_output_dir / "player_shotstyle_drift_summary.csv"
    identity_path = PATHS.archetype_output_dir / "player_identity_drift_table.csv"

    box_summary.to_csv(box_path, index=False)
    shot_summary.to_csv(shot_path, index=False)
    identity.to_csv(identity_path, index=False)

    append_log(
        phase="PHASE 6 — ENGINEER DRIFT / ROLE EVOLUTION",
        completed=(
            "Computed early-career movement in both boxscore role space and shot-style embedding space, plus total path length and subtype-change counts, then assembled a player-level identity drift table with stability classifications."
        ),
        learned=(
            "Year-1 to Year-4 displacement alone is not enough because some players follow long, nonlinear developmental paths. "
            "Combining endpoint displacement, path length, and subtype changes gives a better early-career drift summary."
        ),
        assumptions=(
            "Boxscore role space is approximated with standardized season-level features rather than the original PCA season scores, which were not available as a saved season-level artifact. "
            "Subtype changes are counted only where the player has usable season-level shot embeddings."
        ),
        files_read=[
            str(PATHS.data_dir / "player_season_feature_table_1999_2019.csv"),
            str(PATHS.archetype_output_dir / "shot_embedding_player_season.csv"),
            str(PATHS.archetype_output_dir / "shot_style_player_season_table.csv"),
            str(PATHS.archetype_output_dir / "player_hybrid_archetype_table.csv"),
        ],
        files_written=[str(box_path), str(shot_path), str(identity_path)],
    )


if __name__ == "__main__":
    main()
