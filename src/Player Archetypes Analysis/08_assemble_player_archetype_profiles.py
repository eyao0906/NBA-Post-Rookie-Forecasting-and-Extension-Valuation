from __future__ import annotations

import pandas as pd

from archetype_workflow_utils import PATHS, append_log, ensure_dirs


def main() -> None:
    ensure_dirs()
    identity = pd.read_csv(PATHS.archetype_output_dir / "player_identity_drift_table.csv")
    dossiers = pd.read_csv(PATHS.archetype_output_dir / "player_comp_dossier_table.csv")
    subtype_summary = pd.read_csv(PATHS.archetype_output_dir / "shot_style_cluster_summary.csv")

    final_df = identity.merge(dossiers, on=["PLAYER_ID", "PLAYER_NAME"], how="left")
    final_df = final_df.merge(subtype_summary[["shot_style_subtype", "descriptive_style_notes"]], on="shot_style_subtype", how="left")
    final_df["prototype_fit_ambiguity"] = final_df["prototype_ambiguity_ratio"]
    final_df["supporting_shot_style_explanation"] = final_df["descriptive_style_notes"].fillna("Shot-style subtype unavailable.")

    keep_cols = [
        "PLAYER_ID",
        "PLAYER_NAME",
        "draft_year",
        "macro_archetype",
        "shot_style_subtype",
        "hybrid_archetype_label",
        "prototype_fit_ambiguity",
        "own_cluster_distance",
        "shot_style_subtype_probability",
        "identity_drift_class",
        "box_role_y1_to_y4_displacement",
        "shotstyle_y1_to_y4_displacement",
        "shot_style_subtype_changes",
        "realistic_comp_list",
        "realistic_comp_similarity_mean",
        "ceiling_comp_PLAYER_NAME",
        "ceiling_comp_supported",
        "median_comp_group_points",
        "median_comp_group_minutes",
        "median_comp_group_rebounds",
        "median_comp_group_assists",
        "supporting_shot_style_explanation",
        "comp_based_interpretation",
    ]
    final_keep = [c for c in keep_cols if c in final_df.columns]
    final_profile = final_df[final_keep].copy()
    case_study = final_profile.sort_values(["shot_style_subtype_probability", "realistic_comp_similarity_mean"], ascending=[False, True]).head(40)

    profile_path = PATHS.archetype_output_dir / "final_player_archetype_profile_table.csv"
    case_path = PATHS.archetype_output_dir / "final_player_archetype_case_study_table.csv"
    final_profile.to_csv(profile_path, index=False)
    case_study.to_csv(case_path, index=False)

    append_log(
        phase="PHASE 8 — ASSEMBLE FINAL DELIVERABLE 2 TABLES",
        completed=(
            "Assembled the final player-archetype profile table by combining macro role, shot-style subtype, hybrid label, ambiguity metrics, drift summaries, realistic comps, optional ceiling comps, comp-group outcomes, and a plain-language shot-style explanation."
        ),
        learned=(
            "The deliverable becomes much more interpretable when the three layers are presented together rather than as separate technical artifacts."
        ),
        assumptions=(
            "Case-study rows are selected as a compact stakeholder-ready subset rather than an exhaustive manual curation."
        ),
        files_read=[
            str(PATHS.archetype_output_dir / "player_identity_drift_table.csv"),
            str(PATHS.archetype_output_dir / "player_comp_dossier_table.csv"),
            str(PATHS.archetype_output_dir / "shot_style_cluster_summary.csv"),
        ],
        files_written=[str(profile_path), str(case_path)],
    )


if __name__ == "__main__":
    main()
