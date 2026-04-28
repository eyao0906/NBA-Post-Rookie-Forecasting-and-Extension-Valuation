from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from archetype_workflow_utils import PATHS, append_log, ensure_dirs, embedding_columns

SUBTYPE_LABELS = {
    0: "Rim Pressure Drivers",
    1: "Balanced Three-Level Scorers",
    2: "Arc-Heavy Spacers",
    3: "Interior Finishers",
    4: "Midrange Shot Creators",
}


def representative_players(df: pd.DataFrame, emb_cols: list[str], prob_col: str) -> pd.DataFrame:
    rows = []
    for subtype, grp in df.groupby("shot_style_subtype_id"):
        top = grp.sort_values(prob_col, ascending=False).head(8)
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            rows.append({
                "shot_style_subtype_id": int(subtype),
                "shot_style_subtype": row["shot_style_subtype"],
                "representative_rank": rank,
                "PLAYER_ID": row["PLAYER_ID"],
                "PLAYER_NAME": row["PLAYER_NAME"],
                "membership_probability": row[prob_col],
            })
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    player_emb = pd.read_csv(PATHS.archetype_output_dir / "shot_embedding_player.csv")
    macro = pd.read_csv(PATHS.archetype_output_dir / "archetype_macro_player_table.csv")
    season_emb = pd.read_csv(PATHS.archetype_output_dir / "shot_embedding_player_season.csv")
    emb_cols = embedding_columns(player_emb)

    X = player_emb[emb_cols].to_numpy(dtype=float)
    gmm = GaussianMixture(n_components=5, covariance_type="full", random_state=42)
    labels = gmm.fit_predict(X)
    probs = gmm.predict_proba(X)

    season_X = season_emb[emb_cols].to_numpy(dtype=float)
    season_labels = gmm.predict(season_X)
    season_probs = gmm.predict_proba(season_X)

    player_emb["shot_style_subtype_id"] = labels
    player_emb["shot_style_subtype"] = player_emb["shot_style_subtype_id"].map(SUBTYPE_LABELS)
    player_emb["shot_style_subtype_probability"] = probs.max(axis=1)
    player_emb["shot_style_entropy"] = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1)
    for i in range(probs.shape[1]):
        player_emb[f"shot_style_prob_{i}"] = probs[:, i]

    rep = representative_players(player_emb, emb_cols, "shot_style_subtype_probability")

    season_emb = season_emb.copy()
    season_emb["shot_style_subtype_id"] = season_labels
    season_emb["shot_style_subtype"] = season_emb["shot_style_subtype_id"].map(SUBTYPE_LABELS)
    season_emb["shot_style_subtype_probability"] = season_probs.max(axis=1)
    season_emb["shot_style_entropy"] = -(season_probs * np.log(np.clip(season_probs, 1e-12, None))).sum(axis=1)
    for i in range(season_probs.shape[1]):
        season_emb[f"shot_style_prob_{i}"] = season_probs[:, i]

    subtype_summary = (
        player_emb.groupby(["shot_style_subtype_id", "shot_style_subtype"], dropna=False)
        .agg(
            subtype_size=("PLAYER_ID", "size"),
            avg_membership_probability=("shot_style_subtype_probability", "mean"),
            avg_entropy=("shot_style_entropy", "mean"),
            avg_shot_attempts_covered=("total_shot_attempts_covered", "mean"),
            avg_seasons_with_embeddings=("seasons_with_embeddings", "mean"),
        )
        .reset_index()
    )
    subtype_summary["descriptive_style_notes"] = subtype_summary["shot_style_subtype"].map(
        {
            "Rim Pressure Drivers": "Paint-oriented creators with heavy rim pressure and lower perimeter dependence.",
            "Balanced Three-Level Scorers": "Profiles with meaningful rim, midrange, and three-point balance.",
            "Arc-Heavy Spacers": "High-frequency perimeter shot makers and floor-spacing profiles.",
            "Interior Finishers": "Big-oriented finishing maps concentrated near the basket and paint.",
            "Midrange Shot Creators": "Self-created midrange-heavy profiles with intermediate-area usage.",
        }
    )

    hybrid = macro.merge(player_emb, on=["PLAYER_ID", "PLAYER_NAME"], how="left")
    hybrid["hybrid_archetype_label"] = hybrid["macro_archetype"] + " | " + hybrid["shot_style_subtype"].fillna("No Shot Subtype")

    summary_path = PATHS.archetype_output_dir / "shot_style_cluster_summary.csv"
    rep_path = PATHS.archetype_output_dir / "shot_style_representative_players.csv"
    player_path = PATHS.archetype_output_dir / "shot_style_player_table.csv"
    season_path = PATHS.archetype_output_dir / "shot_style_player_season_table.csv"
    hybrid_path = PATHS.archetype_output_dir / "player_hybrid_archetype_table.csv"

    subtype_summary.to_csv(summary_path, index=False)
    rep.to_csv(rep_path, index=False)
    player_emb.to_csv(player_path, index=False)
    season_emb.to_csv(season_path, index=False)
    hybrid.to_csv(hybrid_path, index=False)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c=labels, s=22, alpha=0.8)
    plt.title("Shot-Style Embedding Neighborhoods")
    plt.xlabel("PC1 of shot embeddings")
    plt.ylabel("PC2 of shot embeddings")
    plt.tight_layout()
    plt.savefig(PATHS.archetype_visual_dir / "shot_style_embedding_scatter.png", dpi=220, bbox_inches="tight")
    plt.close()

    append_log(
        phase="PHASE 5 — CREATE SHOT-STYLE SUBTYPES",
        completed=(
            "Clustered player-level shot embeddings with a Gaussian mixture model, exported both player-level and season-level subtype assignments and soft probabilities, summarized subtype sizes and style notes, and merged the subtype layer into the hybrid archetype table."
        ),
        learned=(
            "A soft-clustering view is useful because some player shot maps sit between neighboring styles rather than mapping cleanly to one rigid subtype. "
            "The subtype probability and entropy columns will help express ambiguity in the final deliverable."
        ),
        assumptions=(
            "Five shot-style subtypes are used to keep the second layer interpretable and comparable in complexity to the five macro-role clusters. "
            "Subtype labels are descriptive post-hoc names rather than externally validated taxonomy labels."
        ),
        files_read=[
            str(PATHS.archetype_output_dir / "shot_embedding_player.csv"),
            str(PATHS.archetype_output_dir / "shot_embedding_player_season.csv"),
            str(PATHS.archetype_output_dir / "archetype_macro_player_table.csv"),
        ],
        files_written=[str(summary_path), str(rep_path), str(player_path), str(season_path), str(hybrid_path), str(PATHS.archetype_visual_dir / "shot_style_embedding_scatter.png")],
    )


if __name__ == "__main__":
    main()
