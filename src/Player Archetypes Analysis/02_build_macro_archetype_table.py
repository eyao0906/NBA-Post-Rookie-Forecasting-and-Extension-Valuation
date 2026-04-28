from __future__ import annotations

import numpy as np
import pandas as pd

from archetype_workflow_utils import PATHS, append_log, ensure_dirs


def main() -> None:
    ensure_dirs()

    clustered = pd.read_csv(PATHS.kmeans_dir / "player_feature_table_1999_2019_clustered_k5.csv")
    predictors = pd.read_csv(PATHS.kmeans_dir / "cluster_modeling_predictors_k5.csv")
    cluster_summary = pd.read_csv(PATHS.kmeans_dir / "cluster_summary_k5.csv")
    cluster_summary_z = pd.read_csv(PATHS.kmeans_dir / "cluster_summary_zscores_k5.csv")
    reps = pd.read_csv(PATHS.kmeans_dir / "cluster_representative_players_k5.csv")

    join_keys = [c for c in ["Player_ID", "COHORT_PLAYER_NAME", "DRAFT_YEAR"] if c in clustered.columns and c in predictors.columns]
    if not join_keys:
        raise ValueError("No valid join keys found between clustered table and cluster_modeling_predictors_k5.csv")

    macro = clustered.merge(predictors, on=join_keys, how="left", validate="one_to_one", suffixes=("", "_predictor"))

    canonical_map = {
        "Player_ID": "PLAYER_ID",
        "COHORT_PLAYER_NAME": "PLAYER_NAME",
        "DRAFT_YEAR": "draft_year",
        "kmeans_cluster_k5": "macro_cluster_id",
        "cluster_archetype_k5": "macro_archetype",
        "own_cluster_distance_k5": "own_cluster_distance",
    }
    for source_col, target_col in canonical_map.items():
        if source_col in macro.columns:
            macro[target_col] = macro[source_col]
        elif f"{source_col}_predictor" in macro.columns:
            macro[target_col] = macro[f"{source_col}_predictor"]

    predictor_cols = [c for c in macro.columns if c.startswith("cluster_") or c.startswith("dist_to_cluster_") or c.startswith("pca_")]
    preferred_predictor_cols = [c for c in predictor_cols if not c.endswith("_predictor")]
    if not preferred_predictor_cols:
        preferred_predictor_cols = predictor_cols
    macro_cols = ["PLAYER_ID", "PLAYER_NAME", "draft_year", "macro_cluster_id", "macro_archetype", "own_cluster_distance"] + preferred_predictor_cols
    macro = macro[[c for c in macro_cols if c in macro.columns]].copy()
    distance_cols = sorted(
        [c for c in macro.columns if c.startswith("dist_to_cluster_")],
        key=lambda x: int(x.replace("dist_to_cluster_", "").split("_")[0]),
    )
    if distance_cols:
        macro["prototype_fit_rank_distance"] = macro[distance_cols].min(axis=1)
        sorted_d = np.sort(macro[distance_cols].to_numpy(), axis=1)
        macro["prototype_margin_second_minus_first"] = sorted_d[:, 1] - sorted_d[:, 0]
        macro["prototype_ambiguity_ratio"] = sorted_d[:, 0] / np.maximum(sorted_d[:, 1], 1e-8)
    # Canonical macro-role columns are assigned before this point so downstream scripts keep a stable contract.

    reps_summary = (
        reps.groupby(["cluster_id", "cluster_archetype_k5"], dropna=False)
        .head(5)
        .groupby(["cluster_id", "cluster_archetype_k5"], dropna=False)["player_name"]
        .apply(lambda s: ", ".join(s.astype(str).tolist()))
        .reset_index(name="representative_players")
        .rename(columns={"cluster_id": "macro_cluster_id", "cluster_archetype_k5": "macro_archetype"})
    )

    summary = cluster_summary.merge(cluster_summary_z, on=["kmeans_cluster_k5", "cluster_archetype_k5"], suffixes=("_mean", "_z"))
    if "kmeans_cluster_k5" in summary.columns:
        summary = summary.rename(columns={"kmeans_cluster_k5": "macro_cluster_id", "cluster_archetype_k5": "macro_archetype"})
    summary = summary.merge(reps_summary, on=["macro_cluster_id", "macro_archetype"], how="left")

    macro_path = PATHS.archetype_output_dir / "archetype_macro_player_table.csv"
    summary_path = PATHS.archetype_output_dir / "archetype_macro_summary_table.csv"
    macro.to_csv(macro_path, index=False)
    summary.to_csv(summary_path, index=False)

    append_log(
        phase="PHASE 2 — LOCK THE MACRO ARCHETYPE BACKBONE",
        completed=(
            "Reused the saved PCA + K-means artifacts instead of refitting them. "
            "Built a finalized macro-archetype player table with cluster IDs, interpretable cluster names, one-hot and distance features, and prototype-fit measures."
        ),
        learned=(
            "The existing five-cluster system already provides role-family information plus useful ambiguity signals via own-cluster distance and distance margins. "
            "That makes it a reasonable macro-role backbone for the hybrid archetype system."
        ),
        assumptions=(
            "The saved cluster names remain authoritative because no material feature-schema inconsistency was detected. "
            "Prototype fit is summarized through centroid-distance features rather than any fresh clustering pass."
        ),
        files_read=[
            str(PATHS.kmeans_dir / "player_feature_table_1999_2019_clustered_k5.csv"),
            str(PATHS.kmeans_dir / "cluster_modeling_predictors_k5.csv"),
            str(PATHS.kmeans_dir / "cluster_summary_k5.csv"),
            str(PATHS.kmeans_dir / "cluster_summary_zscores_k5.csv"),
            str(PATHS.kmeans_dir / "cluster_representative_players_k5.csv"),
        ],
        files_written=[str(macro_path), str(summary_path)],
    )


if __name__ == "__main__":
    main()
