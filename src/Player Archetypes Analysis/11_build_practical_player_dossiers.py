from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Rectangle
import numpy as np
import pandas as pd

from archetype_workflow_utils import PATHS, append_log, ensure_dirs

MIN_TOTAL_SHOTS_SHOWCASE = 150
MIN_SEASONS_WITH_EMBEDDINGS = 2
TOP_K_PER_SUBTYPE = 1
MANUAL_PLAYER_IDS: list[int] = []


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(text).strip().lower())
    return text.strip("_") or "player"


def pick_existing_col(columns: list[str] | pd.Index, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None


def load_main_shots() -> pd.DataFrame:
    shots = pd.read_csv(PATHS.data_dir / "Shot Chart Details Raw" / "raw_shotchart_S1_to_S4_main.csv")
    numeric_cols = [
        "PLAYER_ID",
        "season_num",
        "LOC_X",
        "LOC_Y",
        "SHOT_MADE_FLAG",
        "SHOT_ATTEMPTED_FLAG",
    ]
    for col in numeric_cols:
        if col in shots.columns:
            shots[col] = pd.to_numeric(shots[col], errors="coerce")
    shots = shots[shots["SHOT_ATTEMPTED_FLAG"] == 1].copy()
    shots = shots[shots["LOC_X"].between(-250, 250) & shots["LOC_Y"].between(-47.5, 422.5)].copy()
    return shots


def compute_shot_volume_tables(raw_shots: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    season_counts = (
        raw_shots.groupby(["PLAYER_ID", "season_num"], dropna=False)
        .agg(
            season_shot_attempts=("SHOT_ATTEMPTED_FLAG", "sum"),
            season_shot_makes=("SHOT_MADE_FLAG", "sum"),
        )
        .reset_index()
    )
    player_counts = (
        raw_shots.groupby("PLAYER_ID", dropna=False)
        .agg(
            total_raw_shot_attempts=("SHOT_ATTEMPTED_FLAG", "sum"),
            total_raw_shot_makes=("SHOT_MADE_FLAG", "sum"),
            seasons_with_raw_shots=("season_num", "nunique"),
        )
        .reset_index()
    )
    return season_counts, player_counts


def minmax_scaled(s: pd.Series, invert: bool = False, fill: float = 0.0) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().sum() == 0:
        return pd.Series(fill, index=s.index, dtype=float)
    mn, mx = float(x.min()), float(x.max())
    if math.isclose(mn, mx):
        out = pd.Series(1.0, index=s.index, dtype=float)
    else:
        out = (x - mn) / (mx - mn)
    out = out.fillna(fill)
    return 1.0 - out if invert else out


def draw_court(ax, color: str = "black", lw: float = 1.2, outer_lines: bool = False):
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color, fill=False)
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color, fill=False)
    top_ft = Arc((0, 142.5), 120, 120, theta1=0, theta2=180, linewidth=lw, color=color, fill=False)
    bottom_ft = Arc((0, 142.5), 120, 120, theta1=180, theta2=0, linewidth=lw, color=color, linestyle="dashed")
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw, color=color)
    corner_left = Rectangle((-220, -47.5), 0, 140, linewidth=lw, color=color)
    corner_right = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw, color=color)
    center_outer = Arc((0, 422.5), 120, 120, theta1=180, theta2=0, linewidth=lw, color=color)
    center_inner = Arc((0, 422.5), 40, 40, theta1=180, theta2=0, linewidth=lw, color=color)

    court_elems = [
        hoop,
        backboard,
        outer_box,
        inner_box,
        top_ft,
        bottom_ft,
        restricted,
        corner_left,
        corner_right,
        three_arc,
        center_outer,
        center_inner,
    ]
    if outer_lines:
        court_elems.append(Rectangle((-250, -47.5), 500, 470, linewidth=lw, color=color, fill=False))

    for elem in court_elems:
        ax.add_patch(elem)

    ax.set_xlim(-250, 250)
    ax.set_ylim(-47.5, 422.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def prepare_candidates(final_profile: pd.DataFrame, player_shot: pd.DataFrame, raw_player_counts: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "PLAYER_ID",
        "PLAYER_NAME",
        "shot_style_subtype",
        "shot_style_subtype_id",
        "shot_style_subtype_probability",
        "seasons_with_embeddings",
        "total_shot_attempts_covered",
    ]
    shot_small = player_shot[[c for c in keep_cols if c in player_shot.columns]].drop_duplicates()
    merged = final_profile.merge(shot_small, on=["PLAYER_ID", "PLAYER_NAME", "shot_style_subtype"], how="left")
    merged = merged.merge(raw_player_counts, on="PLAYER_ID", how="left")

    if "total_shot_attempts_covered" not in merged.columns:
        merged["total_shot_attempts_covered"] = np.nan
    if "seasons_with_embeddings" not in merged.columns:
        merged["seasons_with_embeddings"] = np.nan

    merged["total_shot_attempts_covered"] = merged["total_shot_attempts_covered"].fillna(merged["total_raw_shot_attempts"])
    merged["seasons_with_embeddings"] = merged["seasons_with_embeddings"].fillna(merged["seasons_with_raw_shots"])

    comp_col = pick_existing_col(list(merged.columns), ["realistic_comp_list"])
    sim_col = pick_existing_col(list(merged.columns), ["realistic_comp_similarity_mean"])
    amb_col = pick_existing_col(list(merged.columns), ["prototype_fit_ambiguity", "prototype_ambiguity_ratio"])
    prob_col = pick_existing_col(list(merged.columns), ["shot_style_subtype_probability"])

    merged["has_valid_comp_list"] = merged[comp_col].fillna("").astype(str).str.strip().ne("") if comp_col else False
    merged["comp_support_scaled"] = minmax_scaled(merged[sim_col], invert=True, fill=0.0) if sim_col else 0.0
    merged["clarity_scaled"] = minmax_scaled(merged[amb_col], invert=True, fill=0.5) if amb_col else 0.5
    merged["shot_volume_scaled"] = minmax_scaled(np.log1p(merged["total_shot_attempts_covered"].fillna(0)), invert=False, fill=0.0)
    merged["subtype_confidence_scaled"] = minmax_scaled(merged[prob_col], invert=False, fill=0.0) if prob_col else 0.0
    merged["dossier_showcase_eligible"] = (
        merged["has_valid_comp_list"]
        & (merged["total_shot_attempts_covered"].fillna(0) >= MIN_TOTAL_SHOTS_SHOWCASE)
        & (merged["seasons_with_embeddings"].fillna(0) >= MIN_SEASONS_WITH_EMBEDDINGS)
    )
    merged["dossier_selection_score"] = (
        0.35 * merged["subtype_confidence_scaled"]
        + 0.25 * merged["comp_support_scaled"]
        + 0.20 * merged["clarity_scaled"]
        + 0.20 * merged["shot_volume_scaled"]
    )
    return merged


def select_showcase_players(candidates: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"PLAYER_ID", "PLAYER_NAME", "shot_style_subtype", "shot_style_subtype_id"}
    missing = sorted(required_cols.difference(candidates.columns))
    if missing:
        raise ValueError(
            "select_showcase_players is missing required columns: " + ", ".join(missing)
        )

    rows: list[pd.Series] = []
    subtypes = candidates[["shot_style_subtype_id", "shot_style_subtype"]].drop_duplicates().sort_values(
        ["shot_style_subtype_id", "shot_style_subtype"]
    )
    used_ids: set[int] = set()

    # Keep the manual-override table schema aligned with candidates even when no overrides are provided.
    manual = candidates[candidates["PLAYER_ID"].isin(MANUAL_PLAYER_IDS)].copy() if MANUAL_PLAYER_IDS else candidates.iloc[0:0].copy()
    for _, subtype_row in subtypes.iterrows():
        subtype_id = subtype_row["shot_style_subtype_id"]
        grp = candidates[candidates["shot_style_subtype_id"] == subtype_id].copy()
        if grp.empty:
            continue

        manual_grp = manual[manual["shot_style_subtype_id"] == subtype_id]
        if not manual_grp.empty:
            chosen = manual_grp.sort_values("dossier_selection_score", ascending=False).iloc[0]
            rows.append(chosen)
            used_ids.add(int(chosen["PLAYER_ID"]))
            continue

        eligible = grp[(grp["dossier_showcase_eligible"]) & (~grp["PLAYER_ID"].isin(used_ids))].copy()
        pool = eligible if not eligible.empty else grp[(grp["has_valid_comp_list"]) & (~grp["PLAYER_ID"].isin(used_ids))].copy()
        if pool.empty:
            pool = grp[~grp["PLAYER_ID"].isin(used_ids)].copy()
        if pool.empty:
            continue
        sort_cols = [
            "dossier_selection_score",
            "shot_style_subtype_probability",
            "total_shot_attempts_covered",
            "realistic_comp_similarity_mean",
        ]
        sort_cols = [c for c in sort_cols if c in pool.columns]
        ascending = [False, False, False, True][: len(sort_cols)]
        chosen = pool.sort_values(sort_cols, ascending=ascending, na_position="last").iloc[0]
        rows.append(chosen)
        used_ids.add(int(chosen["PLAYER_ID"]))

    out = pd.DataFrame(rows).reset_index(drop=True)
    return out.sort_values(["shot_style_subtype_id", "PLAYER_NAME"]).reset_index(drop=True)


def ambiguity_text(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "prototype fit unavailable"
    value = float(value)
    if value <= 0.65:
        return "clean prototype fit"
    if value <= 0.85:
        return "mostly aligned prototype fit"
    return "mixed prototype fit"


def drift_text(label: str) -> str:
    mapping = {
        "stable": "an early-career identity that stayed relatively stable across Seasons 1–4",
        "evolving_gradually": "a profile that developed gradually rather than changing abruptly",
        "role_shifting_materially": "a materially shifting early-career role and shot identity",
        "insufficient_data": "insufficient drift evidence",
    }
    return mapping.get(str(label), str(label))


def build_summary_table(showcase: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "PLAYER_ID",
        "PLAYER_NAME",
        "draft_year",
        "macro_archetype",
        "shot_style_subtype",
        "hybrid_archetype_label",
        "identity_drift_class",
        "shot_style_subtype_probability",
        "prototype_fit_ambiguity",
        "total_shot_attempts_covered",
        "seasons_with_embeddings",
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
        "dossier_selection_score",
    ]
    keep = [c for c in cols if c in showcase.columns]
    return showcase[keep].copy()


def build_markdown(showcase: pd.DataFrame, output_dir: Path, visual_dir: Path) -> str:
    lines: list[str] = []
    lines.append("# Deliverable 2 Practical Player Dossier Demonstration")
    lines.append("")
    lines.append(
        "This add-on package converts the hybrid-archetype outputs into a front-office-style demonstration: one dossier-ready player is selected for each shot-style subtype, using subtype confidence, comp support, prototype clarity, and shot-volume coverage."
    )
    lines.append("")
    lines.append("## How to use these dossiers")
    lines.append("")
    lines.append(
        "Each dossier is intended to be read alongside the cohort-level subtype dictionary. The player card explains who the player is now, how stable that identity appears to be, which historical players look most similar, and what later-career path that comp neighborhood suggests."
    )
    lines.append("")

    for _, row in showcase.iterrows():
        player_slug = slugify(row["PLAYER_NAME"])
        fig_rel = f"{visual_dir.name}/{player_slug}_dossier_demo.png"
        lines.append(f"## {row['PLAYER_NAME']}")
        lines.append("")
        lines.append(f"- **Draft year:** {int(row['draft_year']) if 'draft_year' in row and pd.notna(row['draft_year']) else 'n/a'}")
        lines.append(f"- **Macro archetype:** {row.get('macro_archetype', 'n/a')}")
        lines.append(f"- **Shot-style subtype:** {row.get('shot_style_subtype', 'n/a')}")
        lines.append(f"- **Hybrid label:** {row.get('hybrid_archetype_label', 'n/a')}")
        lines.append(f"- **Drift class:** {row.get('identity_drift_class', 'n/a')}")
        prob = row.get("shot_style_subtype_probability")
        lines.append(f"- **Subtype probability:** {float(prob):.2f}" if pd.notna(prob) else "- **Subtype probability:** n/a")
        shots = row.get("total_shot_attempts_covered")
        seasons = row.get("seasons_with_embeddings")
        lines.append(
            f"- **Embedding coverage:** {int(shots)} shots across {int(seasons)} seasons"
            if pd.notna(shots) and pd.notna(seasons)
            else "- **Embedding coverage:** n/a"
        )
        lines.append("")

        comp_text = str(row.get("realistic_comp_list", "n/a"))
        ceiling = row.get("ceiling_comp_PLAYER_NAME")
        ceiling_supported = int(row.get("ceiling_comp_supported", 0)) if pd.notna(row.get("ceiling_comp_supported")) else 0
        med_pts = row.get("median_comp_group_points")
        med_min = row.get("median_comp_group_minutes")
        med_reb = row.get("median_comp_group_rebounds")
        med_ast = row.get("median_comp_group_assists")
        med_text = []
        if pd.notna(med_pts):
            med_text.append(f"{float(med_pts):.1f} points")
        if pd.notna(med_min):
            med_text.append(f"{float(med_min):.1f} minutes")
        if pd.notna(med_reb):
            med_text.append(f"{float(med_reb):.1f} rebounds")
        if pd.notna(med_ast):
            med_text.append(f"{float(med_ast):.1f} assists")
        med_clause = ", ".join(med_text) if med_text else "later-career summary unavailable"

        paragraph = (
            f"{row['PLAYER_NAME']} projects as a **{row.get('macro_archetype', 'n/a')}** with a **{row.get('shot_style_subtype', 'n/a')}** shot identity. "
            f"The profile shows **{ambiguity_text(row.get('prototype_fit_ambiguity'))}** and is classified as **{row.get('identity_drift_class', 'n/a')}**, indicating {drift_text(row.get('identity_drift_class'))}. "
            f"The shot-style explanation is: {row.get('supporting_shot_style_explanation', 'n/a')} "
            f"Closest realistic comps are **{comp_text}**. "
            f"The realistic comp group has median later-career values of **{med_clause}**. "
        )
        if pd.notna(ceiling) and ceiling_supported == 1:
            paragraph += f"A supported upside analog is **{ceiling}**, which should be read as a ceiling comp rather than a direct forecast. "
        paragraph += "In practice, this dossier would be used to contextualize the player’s extension discussion alongside the separate forecast and salary decision layers."
        lines.append(paragraph)
        lines.append("")
        lines.append(f"- **Suggested player-specific figure:** `{fig_rel}`")
        lines.append("")

    return "\n".join(lines)


def plot_player_dossier_demo(row: pd.Series, raw_shots: pd.DataFrame, season_counts: pd.DataFrame, out_path: Path) -> None:
    player_id = row["PLAYER_ID"]
    player_name = row["PLAYER_NAME"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    season_specs = [(None, "Seasons 1–4"), (1, "Year 1"), (4, "Year 4")]
    for ax, (season_num, label) in zip(axes, season_specs):
        grp = raw_shots[raw_shots["PLAYER_ID"] == player_id].copy()
        if season_num is not None:
            grp = grp[grp["season_num"] == season_num].copy()
        made = grp[grp["SHOT_MADE_FLAG"] == 1]
        miss = grp[grp["SHOT_MADE_FLAG"] == 0]
        ax.scatter(miss["LOC_X"], miss["LOC_Y"], s=8, alpha=0.18, color="tab:gray")
        ax.scatter(made["LOC_X"], made["LOC_Y"], s=10, alpha=0.33, color="tab:blue")
        draw_court(ax, outer_lines=True)
        if season_num is None:
            shots = len(grp)
        else:
            shots = int(
                season_counts[
                    (season_counts["PLAYER_ID"] == player_id) & (season_counts["season_num"] == season_num)
                ]["season_shot_attempts"].sum()
            )
        ax.set_title(f"{label}\nShots={shots}")

    comp_text = str(row.get("realistic_comp_list", "n/a"))
    top_comp = comp_text.split(",")[0].strip() if comp_text and comp_text != "nan" else "n/a"
    fig.suptitle(
        f"{player_name} | {row.get('macro_archetype', 'n/a')} | {row.get('shot_style_subtype', 'n/a')}\n"
        f"Drift: {row.get('identity_drift_class', 'n/a')} | Top comp: {top_comp}",
        y=0.98,
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    ensure_dirs()
    raw_shots = load_main_shots()
    season_counts, raw_player_counts = compute_shot_volume_tables(raw_shots)

    final_profile = pd.read_csv(PATHS.archetype_output_dir / "final_player_archetype_profile_table.csv")
    player_shot = pd.read_csv(PATHS.archetype_output_dir / "shot_style_player_table.csv")

    dossier_output_dir = PATHS.archetype_output_dir / "player_dossier_demo"
    dossier_visual_dir = PATHS.archetype_visual_dir / "player_dossier_demo"
    dossier_output_dir.mkdir(parents=True, exist_ok=True)
    dossier_visual_dir.mkdir(parents=True, exist_ok=True)

    candidates = prepare_candidates(final_profile, player_shot, raw_player_counts)
    showcase = select_showcase_players(candidates)
    summary = build_summary_table(showcase)

    manifest_rows = []
    for _, row in showcase.iterrows():
        player_slug = slugify(row["PLAYER_NAME"])
        fig_path = dossier_visual_dir / f"{player_slug}_dossier_demo.png"
        plot_player_dossier_demo(row, raw_shots, season_counts, fig_path)
        manifest_rows.append(
            {
                "PLAYER_ID": row["PLAYER_ID"],
                "PLAYER_NAME": row["PLAYER_NAME"],
                "shot_style_subtype": row.get("shot_style_subtype"),
                "hybrid_archetype_label": row.get("hybrid_archetype_label"),
                "dossier_figure_path": str(fig_path),
            }
        )

    summary_path = dossier_output_dir / "report_ready_player_dossier_table.csv"
    selection_path = dossier_output_dir / "report_ready_player_dossier_selection.csv"
    manifest_path = dossier_output_dir / "report_ready_player_dossier_manifest.csv"
    md_path = dossier_output_dir / "report_ready_player_dossiers.md"

    summary.to_csv(summary_path, index=False)
    showcase.to_csv(selection_path, index=False)
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    md_path.write_text(build_markdown(showcase, dossier_output_dir, dossier_visual_dir), encoding="utf-8")

    append_log(
        phase="PHASE 11 — BUILD PRACTICAL PLAYER DOSSIER DEMO PACKAGE",
        completed=(
            "Selected one dossier-ready player per shot-style subtype from the finalized Deliverable 2 table, exported a report-ready dossier summary table, wrote plain-language markdown cards, and created player-specific three-panel shot-chart figures for practical front-office demonstration use."
        ),
        learned=(
            "Deliverable 2 becomes more report-ready when the hybrid archetype outputs are translated into a small number of explicit player cards instead of leaving the final profile table as the only stakeholder-facing artifact."
        ),
        assumptions=(
            "Selection prioritizes subtype confidence, realistic comp support, prototype clarity, and shot-volume coverage. These dossiers are meant to complement, not replace, the separate forecast and salary decision deliverables."
        ),
        files_read=[
            str(PATHS.archetype_output_dir / "final_player_archetype_profile_table.csv"),
            str(PATHS.archetype_output_dir / "shot_style_player_table.csv"),
            str(PATHS.data_dir / "Shot Chart Details Raw" / "raw_shotchart_S1_to_S4_main.csv"),
        ],
        files_written=[
            str(summary_path),
            str(selection_path),
            str(manifest_path),
            str(md_path),
            str(dossier_visual_dir),
        ],
    )


if __name__ == "__main__":
    main()
