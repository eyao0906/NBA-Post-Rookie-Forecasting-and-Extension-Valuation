from __future__ import annotations

from pathlib import Path

from archetype_workflow_utils import PATHS


SCRIPT_DESCRIPTIONS = {
    "00_project_inventory.py": "Resolves and inventories required project files and initializes the workflow log.",
    "01_cohort_coverage_audit.py": "Audits drafted-player and HOF cohort coverage across logs, features, clusters, shots, and targets.",
    "02_build_macro_archetype_table.py": "Locks the saved macro-role layer from existing PCA + K-means artifacts.",
    "03_build_shotstyle_tensors.py": "Builds drafted-player and HOF player-season spatial shot tensors on a common half-court grid.",
    "04_train_shot_autoencoder.py": "Trains a shot-style CNN autoencoder and exports player-season/player embeddings plus diagnostics.",
    "05_cluster_shot_embeddings.py": "Creates shot-style subtypes from player-level shot embeddings and merges them into the hybrid archetype table.",
    "06_build_identity_drift.py": "Quantifies early-career drift in boxscore role space and shot-style embedding space.",
    "07_build_comps.py": "Constructs realistic drafted-player comps and optional HOF ceiling comps.",
    "08_assemble_player_archetype_profiles.py": "Assembles the final Deliverable 2 profile and case-study tables.",
    "09_make_player_archetype_visuals.py": "Creates stakeholder-facing archetype, embedding, drift, and comp visuals.",
    "10_write_workflow_log.py": "Appends script-by-script workflow documentation, assumptions, warnings, and next-review notes.",
}


def main() -> None:
    script_dir = PATHS.project_root / "src" / "Player Archetypes Analysis"
    lines = []
    lines.append("Workflow structure")
    lines.append("Scripts created in src\\Player Archetypes Analysis:")
    for name, desc in SCRIPT_DESCRIPTIONS.items():
        lines.append(f"- {name}: {desc}")
    lines.append("")
    lines.append("Key assumptions")
    lines.append("- Existing PCA + K-means artifacts remain the macro-role backbone and were reused rather than refit.")
    lines.append("- Shot-style is modeled as a learned spatial embedding layer, not just descriptive shot plots.")
    lines.append("- Early-career drift combines boxscore role movement, shot-style movement, and subtype changes.")
    lines.append("- Realistic comps come only from the historical drafted-player cohort; HOF comps are optional upside analogs.")
    lines.append("")
    lines.append("Warnings / limitations")
    lines.append("- HOF ceiling comps are auxiliary and should not be interpreted as direct realistic comparisons.")
    lines.append("- Boxscore drift uses standardized season-level features rather than saved season-level PCA coordinates because those were not stored as a reusable artifact.")
    lines.append("- GMM subtype labels are descriptive and should be reviewed for presentation language before final report submission.")
    lines.append("- Visuals emphasize readability over exhaustive technical depth; average subtype shot maps can be added in a later refinement pass.")
    lines.append("")
    lines.append("Quality checks performed")
    lines.append("- Confirmed required paths exist and saved an inventory table.")
    lines.append("- Audited sample coverage and missingness before modeling.")
    lines.append("- Reused existing clustering artifacts to avoid unnecessary recomputation.")
    lines.append("- Used common tensor preprocessing for drafted-player and HOF shot charts.")
    lines.append("- Exported nearest-neighbor and reconstruction diagnostics for shot embeddings.")
    lines.append("- Preserved explicit ambiguity/probability columns for macro and shot-style layers.")
    lines.append("")
    lines.append("What should be reviewed or revised later")
    lines.append("- Revisit the number of shot-style subtypes using model selection metrics and basketball face-validity review.")
    lines.append("- Add average subtype shot maps directly from tensors for the final report deck.")
    lines.append("- Stress-test comp weights with sensitivity analysis.")
    lines.append("- Consider calibrating HOF macro-role alignment more explicitly if a reusable HOF boxscore-cluster mapping is later saved.")
    lines.append("")

    with PATHS.log_path.open("a", encoding="utf-8") as f:
        f.write("\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
