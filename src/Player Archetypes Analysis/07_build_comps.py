from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from archetype_workflow_utils import PATHS, append_log, ensure_dirs, embedding_columns

TARGET_METRICS = ["GP", "GS", "MIN", "PTS", "REB", "AST", "STL", "BLK"]
MACRO_FEATURE_CANDIDATES = [
    "PTS_PER100",
    "AST_PER100",
    "REB_PER100",
    "STL_PER100",
    "BLK_PER100",
    "TOV_PER100",
    "FG3A_RATE",
    "FTR",
    "TS_PCT",
]


def target_summary(targets: pd.DataFrame) -> pd.DataFrame:
    targets["PLAYER_ID"] = pd.to_numeric(targets["PLAYER_ID"], errors="coerce")
    return (
        targets.groupby(["PLAYER_ID", "PLAYER_NAME"], dropna=False)[TARGET_METRICS]
        .median()
        .reset_index()
        .rename(columns={c: f"median_later_{c.lower()}" for c in TARGET_METRICS})
    )


def build_realistic_comps(df: pd.DataFrame, emb_cols: list[str], pca_cols: list[str]) -> pd.DataFrame:
    shot_mat = df[emb_cols].to_numpy(dtype=float)
    shot_dist = pairwise_distances(shot_mat)
    pca_mat = df[pca_cols].to_numpy(dtype=float) if pca_cols else None
    pca_dist = pairwise_distances(pca_mat) if pca_cols else np.zeros((len(df), len(df)))
    own_values = df["own_cluster_distance"].to_numpy(dtype=float)
    box_disp = df.get("box_role_y1_to_y4_displacement", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    shot_disp = df.get("shotstyle_y1_to_y4_displacement", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    macro_labels = df["macro_archetype"].astype(str).to_numpy()

    rows = []
    n = len(df)
    for i in range(n):
        candidate_idx = np.array([j for j in range(n) if j != i], dtype=int)
        if len(candidate_idx) == 0:
            continue
        same_macro = (macro_labels[candidate_idx] == macro_labels[i]).astype(int)
        macro_penalty = np.where(same_macro == 1, 0.0, 1.0)
        own_component = np.abs(own_values[candidate_idx] - own_values[i])
        box_component = np.abs(box_disp[candidate_idx] - box_disp[i])
        shot_component = np.abs(shot_disp[candidate_idx] - shot_disp[i])
        valid_counts = (~np.isnan(box_component)).astype(int) + (~np.isnan(shot_component)).astype(int)
        summed = np.nan_to_num(box_component, nan=0.0) + np.nan_to_num(shot_component, nan=0.0)
        drift_component = np.divide(summed, np.maximum(valid_counts, 1), out=np.zeros_like(summed, dtype=float), where=valid_counts > 0)
        total_score = 0.35 * shot_dist[i, candidate_idx] + 0.30 * pca_dist[i, candidate_idx] + 0.15 * own_component + 0.10 * drift_component + 0.10 * macro_penalty
        order = np.argsort(total_score)[:5]
        for rank, pos in enumerate(order, start=1):
            j = int(candidate_idx[pos])
            rows.append(
                {
                    "PLAYER_ID": df.iloc[i]["PLAYER_ID"],
                    "PLAYER_NAME": df.iloc[i]["PLAYER_NAME"],
                    "comp_PLAYER_ID": df.iloc[j]["PLAYER_ID"],
                    "comp_PLAYER_NAME": df.iloc[j]["PLAYER_NAME"],
                    "same_macro_archetype": int(macro_labels[j] == macro_labels[i]),
                    "macro_match_penalty": float(macro_penalty[pos]),
                    "shot_embedding_distance": float(shot_dist[i, j]),
                    "pca_distance": float(pca_dist[i, j]),
                    "prototype_fit_distance_gap": float(own_component[pos]),
                    "drift_difference": float(drift_component[pos]),
                    "similarity_score": float(total_score[pos]),
                    "comp_rank": rank,
                }
            )
    return pd.DataFrame(rows).sort_values(["PLAYER_ID", "comp_rank", "similarity_score"]).reset_index(drop=True)


def assign_hof_macro_roles(hof_df: pd.DataFrame, macro_reference: pd.DataFrame, pca_cols: list[str]) -> pd.DataFrame:
    if not pca_cols:
        hof_df = hof_df.copy()
        hof_df["macro_archetype"] = "macro_unavailable"
        hof_df["macro_cluster_id"] = np.nan
        return hof_df
    ref_centroids = macro_reference.groupby(["macro_cluster_id", "macro_archetype"], dropna=False)[pca_cols].mean().reset_index()
    hof_mat = hof_df[pca_cols].to_numpy(dtype=float)
    centroid_mat = ref_centroids[pca_cols].to_numpy(dtype=float)
    d = pairwise_distances(hof_mat, centroid_mat)
    nearest = np.argmin(d, axis=1)
    out = hof_df.copy()
    out["macro_cluster_id"] = ref_centroids.iloc[nearest]["macro_cluster_id"].to_numpy()
    out["macro_archetype"] = ref_centroids.iloc[nearest]["macro_archetype"].to_numpy()
    out["macro_centroid_distance"] = d[np.arange(len(out)), nearest]
    return out


def build_hof_ceiling_comps(draft_df: pd.DataFrame, hof_df: pd.DataFrame, emb_cols: list[str], pca_cols: list[str]) -> pd.DataFrame:
    draft_mat = draft_df[emb_cols].to_numpy(dtype=float)
    hof_mat = hof_df[emb_cols].to_numpy(dtype=float)
    d = pairwise_distances(draft_mat, hof_mat)
    pca_d = pairwise_distances(draft_df[pca_cols].to_numpy(dtype=float), hof_df[pca_cols].to_numpy(dtype=float)) if pca_cols else np.zeros_like(d)
    rows = []
    for i in range(len(draft_df)):
        combined = 0.65 * d[i] + 0.35 * pca_d[i]
        j = int(np.argmin(combined))
        rows.append(
            {
                "PLAYER_ID": draft_df.iloc[i]["PLAYER_ID"],
                "PLAYER_NAME": draft_df.iloc[i]["PLAYER_NAME"],
                "ceiling_comp_PLAYER_ID": hof_df.iloc[j]["PLAYER_ID"],
                "ceiling_comp_PLAYER_NAME": hof_df.iloc[j]["PLAYER_NAME"],
                "ceiling_shot_embedding_distance": float(d[i, j]),
                "ceiling_pca_distance": float(pca_d[i, j]),
                "ceiling_macro_role_match": int(draft_df.iloc[i]["macro_archetype"] == hof_df.iloc[j]["macro_archetype"]),
                "ceiling_comp_supported": int((combined[j] <= np.quantile(combined, 0.35)) or (draft_df.iloc[i]["macro_archetype"] == hof_df.iloc[j]["macro_archetype"])),
                "ceiling_comp_similarity_score": float(combined[j]),
            }
        )
    return pd.DataFrame(rows)


def dossier_table(realistic: pd.DataFrame, targets: pd.DataFrame, ceiling: pd.DataFrame) -> pd.DataFrame:
    comp_targets = realistic.merge(targets, left_on=["comp_PLAYER_ID", "comp_PLAYER_NAME"], right_on=["PLAYER_ID", "PLAYER_NAME"], how="left", suffixes=("", "_target"))
    dossier = (
        comp_targets.groupby(["PLAYER_ID", "PLAYER_NAME"], dropna=False)
        .agg(
            realistic_comp_list=("comp_PLAYER_NAME", lambda s: ", ".join(s.head(5).astype(str).tolist())),
            realistic_comp_similarity_mean=("similarity_score", "mean"),
            median_comp_group_points=("median_later_pts", "median"),
            median_comp_group_minutes=("median_later_min", "median"),
            median_comp_group_rebounds=("median_later_reb", "median"),
            median_comp_group_assists=("median_later_ast", "median"),
        )
        .reset_index()
    )
    dossier = dossier.merge(ceiling, on=["PLAYER_ID", "PLAYER_NAME"], how="left")
    dossier["comp_based_interpretation"] = dossier.apply(
        lambda r: (
            f"Closest realistic comps: {r['realistic_comp_list']}. "
            f"Comp group median later-career points: {r['median_comp_group_points']:.1f}; "
            f"minutes: {r['median_comp_group_minutes']:.1f}."
        ),
        axis=1,
    )
    return dossier


def main() -> None:
    ensure_dirs()
    identity = pd.read_csv(PATHS.archetype_output_dir / "player_identity_drift_table.csv")
    hof_embeddings = pd.read_csv(PATHS.archetype_output_dir / "hof_shot_embedding_player.csv")
    hof_features = pd.read_csv(PATHS.data_dir / "Shot Feature" / "hof_player_shot_features_4yr.csv")
    targets = target_summary(pd.read_csv(PATHS.data_dir / "career_totals_targets.csv"))
    emb_cols = embedding_columns(identity)
    pca_cols = [c for c in identity.columns if c.startswith("pca_")]

    drafted = identity.dropna(subset=emb_cols).copy()
    realistic = build_realistic_comps(drafted, emb_cols, pca_cols)

    hof_full = hof_embeddings.rename(columns={"DRAFT_YEAR": "draft_year"}).copy()
    for col in pca_cols:
        if col not in hof_full.columns:
            hof_full[col] = drafted[col].mean()
    hof_full = assign_hof_macro_roles(hof_full, drafted[["macro_cluster_id", "macro_archetype"] + pca_cols], pca_cols)

    ceiling = build_hof_ceiling_comps(drafted, hof_full, emb_cols, pca_cols)
    dossier = dossier_table(realistic, targets, ceiling)

    realistic_path = PATHS.archetype_output_dir / "realistic_comps.csv"
    ceiling_path = PATHS.archetype_output_dir / "ceiling_comps_hof.csv"
    dossier_path = PATHS.archetype_output_dir / "player_comp_dossier_table.csv"

    realistic.to_csv(realistic_path, index=False)
    ceiling.to_csv(ceiling_path, index=False)
    dossier.to_csv(dossier_path, index=False)

    append_log(
        phase="PHASE 7 — BUILD COMPARABLE-PLAYER INFRASTRUCTURE",
        completed=(
            "Built realistic drafted-cohort comps using a weighted hybrid similarity score across macro-role compatibility, shot-style embedding distance, prototype fit, and drift similarity. "
            "Also created optional HOF ceiling comps from the auxiliary HOF pool and summarized comp-group later-career outcomes in dossier form."
        ),
        learned=(
            "The historical drafted-player cohort is large enough to support realistic comp neighborhoods without relying on HOF players for the main comparison task. "
            "HOF comps work better as ceiling analogs than as everyday nearest neighbors."
        ),
        assumptions=(
            "Realistic comps use macro-role match indicators, PCA distance when available, prototype-fit distance, shot-style embedding distance, and drift similarity without relying on arbitrary cluster-ID ordering. "
            "HOF ceiling comps receive inferred macro-role assignments from the drafted-player macro reference space rather than a hardcoded placeholder label."
        ),
        files_read=[
            str(PATHS.archetype_output_dir / "player_identity_drift_table.csv"),
            str(PATHS.archetype_output_dir / "hof_shot_embedding_player.csv"),
            str(PATHS.data_dir / "career_totals_targets.csv"),
            str(PATHS.data_dir / "HOF_career_total_targets.csv"),
            str(PATHS.data_dir / "cohort_HOF.csv"),
        ],
        files_written=[str(realistic_path), str(ceiling_path), str(dossier_path)],
    )


if __name__ == "__main__":
    main()
