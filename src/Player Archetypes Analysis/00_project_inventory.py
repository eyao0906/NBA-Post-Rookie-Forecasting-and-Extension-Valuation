from __future__ import annotations

from pathlib import Path

import pandas as pd

from archetype_workflow_utils import PATHS, append_log, ensure_dirs, reset_log


def resolve_path(expected_path: Path, file_name: str, search_root: Path | None = None) -> tuple[Path, bool, bool]:
    if expected_path.exists():
        return expected_path, True, False
    root = search_root or expected_path.parent
    matches = sorted(root.rglob(file_name))
    if not matches and root != PATHS.project_root:
        matches = sorted(PATHS.project_root.rglob(file_name))
    if len(matches) == 1:
        return matches[0], True, True
    if len(matches) > 1:
        matches = sorted(matches, key=lambda p: (len(p.parts), str(p)))
        return matches[0], True, True
    return expected_path, False, False


def build_inventory() -> pd.DataFrame:
    project_root = PATHS.project_root
    target_files = [
        ("raw_game_logs_S1_to_S4.csv", project_root / "data" / "raw_game_logs_S1_to_S4.csv", "Drafted-player raw Seasons 1-4 game logs", "raw"),
        ("career_totals_targets.csv", project_root / "data" / "career_totals_targets.csv", "Drafted-player later-career target totals", "raw"),
        ("cohort_1999_2019.csv", project_root / "data" / "cohort_1999_2019.csv", "Drafted-player cohort definition", "raw"),
        ("HOF_raw_s1_s4_game_logs.csv", project_root / "data" / "HOF_raw_s1_s4_game_logs.csv", "Hall of Fame raw Seasons 1-4 game logs", "raw"),
        ("HOF_career_total_targets.csv", project_root / "data" / "HOF_career_total_targets.csv", "Hall of Fame later-career target totals", "raw"),
        ("player_feature_table_1999_2019.csv", project_root / "data" / "player_feature_table_1999_2019.csv", "Player-level boxscore feature table", "feature"),
        ("player_season_feature_table_1999_2019.csv", project_root / "data" / "player_season_feature_table_1999_2019.csv", "Season-level boxscore feature table", "feature"),
        ("cluster_modeling_predictors_k5.csv", project_root / "kmeans_k5_outputs_split" / "cluster_modeling_predictors_k5.csv", "Saved cluster modeling predictors", "clustering"),
        ("player_feature_table_1999_2019_clustered_k5.csv", project_root / "kmeans_k5_outputs_split" / "player_feature_table_1999_2019_clustered_k5.csv", "Saved player-level cluster assignments", "clustering"),
        ("cluster_summary_k5.csv", project_root / "kmeans_k5_outputs_split" / "cluster_summary_k5.csv", "Cluster summary means", "clustering"),
        ("cluster_summary_zscores_k5.csv", project_root / "kmeans_k5_outputs_split" / "cluster_summary_zscores_k5.csv", "Cluster summary z-scores", "clustering"),
        ("cluster_representative_players_k5.csv", project_root / "kmeans_k5_outputs_split" / "cluster_representative_players_k5.csv", "Representative cluster players", "clustering"),
        ("player_feature_table_pca_scores.csv", project_root / "data" / "pca_player_features" / "player_feature_table_pca_scores.csv", "Saved PCA scores", "clustering"),
        ("pca_artifacts.joblib", project_root / "data" / "pca_player_features" / "pca_artifacts.joblib", "Saved PCA artifacts", "clustering"),
        ("raw_shotchart_S1_to_S4_main.csv", project_root / "data" / "Shot Chart Details Raw" / "raw_shotchart_S1_to_S4_main.csv", "Drafted-player raw shot chart data", "shot"),
        ("raw_shotchart_S1_to_S4_hof_shotstyle.csv", project_root / "data" / "Shot Chart Details Raw" / "raw_shotchart_S1_to_S4_hof_shotstyle.csv", "Hall of Fame raw shot chart data", "shot"),
        ("player_season_shot_tensors.npz", project_root / "data" / "Shot Feature" / "shot_tensors" / "player_season_shot_tensors.npz", "Existing drafted-player shot tensors", "shot"),
        ("player_shot_tensor_index.csv", project_root / "data" / "Shot Feature" / "shot_tensors" / "player_shot_tensor_index.csv", "Existing drafted-player tensor index", "shot"),
        ("player_season_shot_embedding.csv", project_root / "Output" / "ShotChartDetail" / "player_season_shot_embedding.csv", "Existing drafted-player season embeddings", "output"),
        ("player_shot_embedding.csv", project_root / "Output" / "ShotChartDetail" / "player_shot_embedding.csv", "Existing drafted-player player embeddings", "output"),
        ("shot_autoencoder_model.pt", project_root / "Output" / "ShotChartDetail" / "shot_autoencoder_model.pt", "Existing shot autoencoder weights", "output"),
        ("shot_autoencoder_metadata.json", project_root / "Output" / "ShotChartDetail" / "shot_autoencoder_metadata.json", "Existing shot autoencoder metadata", "output"),
        ("training_history.csv", project_root / "Output" / "ShotChartDetail" / "training_history.csv", "Existing shot autoencoder training history", "output"),
        ("build_shotchart_tensors.py", project_root / "src" / "Feature Engineering" / "build_shotchart_tensors.py", "Existing drafted tensor builder", "code"),
        ("train_shotchart_autoencoder.py", project_root / "src" / "Feature Engineering" / "train_shotchart_autoencoder.py", "Existing shot autoencoder trainer", "code"),
        ("kmeans_from_pca_player_features.py", project_root / "src" / "Feature Engineering" / "kmeans_from_pca_player_features.py", "Existing macro archetype clustering script", "code"),
    ]

    rows = []
    corrected = []
    for file_name, expected_path, purpose, category in target_files:
        search_root = expected_path.parent if expected_path.parent.exists() else PATHS.project_root
        resolved_path, exists, corrected_flag = resolve_path(expected_path, file_name, search_root=search_root)
        rows.append(
            {
                "file_name": file_name,
                "expected_path": str(expected_path),
                "resolved_path": str(resolved_path),
                "exists": exists,
                "path_corrected_flag": corrected_flag,
                "purpose": purpose,
                "category": category,
            }
        )
        if corrected_flag:
            corrected.append(f"{file_name}: {expected_path} -> {resolved_path}")
    return pd.DataFrame(rows), corrected


def main() -> None:
    ensure_dirs()
    objective = (
        "Implement Deliverable 2 as a hybrid player-archetype workflow: macro role from existing PCA + K-means, "
        "shot-style subtype from ShotChartDetail spatial data, and early-career drift across Seasons 1-4, then build comps and stakeholder-ready outputs."
    )
    resolved_paths = {
        "project_root": str(PATHS.project_root),
        "src_output_root": str(PATHS.project_root / "src" / "Player Archetypes Analysis"),
        "analysis_output_root": str(PATHS.archetype_output_dir),
        "visual_output_root": str(PATHS.archetype_visual_dir),
    }
    reset_log(objective, resolved_paths)

    inventory, corrected_paths = build_inventory()
    out_path = PATHS.archetype_output_dir / "project_file_inventory.csv"
    inventory.to_csv(out_path, index=False)

    append_log(
        phase="PHASE 0 — PATH AUDIT AND PROJECT MAP",
        completed=(
            "Resolved the required drafted-player, Hall of Fame, feature, clustering, shot-chart, and reusable code artifacts under the existing project structure. "
            "Saved a project inventory table with expected paths, resolved paths, existence flags, and path-correction indicators."
        ),
        learned=(
            "The project already contains reusable PCA/K-means artifacts, drafted-player shot tensors, a trained shot autoencoder, and drafted-player embeddings. "
            f"Resolved path corrections: {('; '.join(corrected_paths) if corrected_paths else 'none required')}."
        ),
        assumptions=(
            "The saved clustering artifacts remain the authoritative macro-role backbone unless a concrete schema mismatch is found later. "
            "Existing shot-model artifacts may be reused, but the new workflow can refit modules inside the new analysis folder when needed for reproducibility."
        ),
        files_read=[str(PATHS.project_root)],
        files_written=[str(out_path), str(PATHS.log_path)],
    )


if __name__ == "__main__":
    main()
