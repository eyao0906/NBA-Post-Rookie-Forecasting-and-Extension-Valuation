from __future__ import annotations

import math
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Arc, Circle, Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from archetype_workflow_utils import PATHS, ensure_dirs, embedding_columns, append_log

MIN_TOTAL_SHOTS_SHOWCASE = 150
MIN_SEASONS_WITH_EMBEDDINGS = 2
MIN_YEAR1_SHOTS_DRIFT = 40
MIN_YEAR4_SHOTS_DRIFT = 40
HEXBIN_GRIDSIZE = 26
HEXBIN_MINCNT_FLOOR = 20
HEXBIN_MINCNT_SHARE = 0.003
def pick_existing_col(columns, candidates):
    """Return the first candidate column name that exists."""
    for c in candidates:
        if c in columns:
            return c
    return None


def row_value(row, candidates, default=np.nan):
    """Return the first available row value from candidate column names."""
    for c in candidates:
        if c in row.index:
            return row[c]
    return default

def draw_court(ax, color: str = "black", lw: float = 1.2, outer_lines: bool = False):
    """Draw an NBA half-court using common shot-chart coordinates."""
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


def load_main_shots() -> pd.DataFrame:
    shots = pd.read_csv(PATHS.data_dir / "Shot Chart Details Raw" / "raw_shotchart_S1_to_S4_main.csv")
    for col in ["PLAYER_ID", "season_num", "LOC_X", "LOC_Y", "SHOT_MADE_FLAG", "SHOT_ATTEMPTED_FLAG"]:
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


def prepare_player_subtype_table(player_shot: pd.DataFrame, raw_player_counts: pd.DataFrame) -> pd.DataFrame:
    out = player_shot.copy()
    out = out.merge(raw_player_counts, on="PLAYER_ID", how="left")
    out["total_shot_attempts_covered"] = out["total_shot_attempts_covered"].fillna(out["total_raw_shot_attempts"])
    out["seasons_with_embeddings"] = out["seasons_with_embeddings"].fillna(out["seasons_with_raw_shots"])
    out["representative_score"] = out["shot_style_subtype_probability"] * np.log1p(
        out["total_shot_attempts_covered"].fillna(0.0)
    )
    out["showcase_eligible"] = (
        (out["total_shot_attempts_covered"].fillna(0.0) >= MIN_TOTAL_SHOTS_SHOWCASE)
        & (out["seasons_with_embeddings"].fillna(0.0) >= MIN_SEASONS_WITH_EMBEDDINGS)
    )
    return out


def subtype_panel_candidates(player_subtypes: pd.DataFrame, raw_shots: pd.DataFrame) -> pd.DataFrame:
    """Use player-level dominant subtype for the main subtype plot."""
    cols = [
        "PLAYER_ID",
        "PLAYER_NAME",
        "shot_style_subtype_id",
        "shot_style_subtype",
        "total_shot_attempts_covered",
        "seasons_with_embeddings",
        "showcase_eligible",
    ]
    merged = raw_shots.merge(player_subtypes[cols], on="PLAYER_ID", how="inner", validate="many_to_one")
    return merged


def select_representative_players(player_subtypes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subtype_id, grp in player_subtypes.groupby("shot_style_subtype_id", dropna=False):
        eligible = grp[grp["showcase_eligible"]].copy()
        pool = eligible if not eligible.empty else grp.copy()
        if pool.empty:
            continue
        chosen = pool.sort_values(
            ["representative_score", "shot_style_subtype_probability", "total_shot_attempts_covered"],
            ascending=[False, False, False],
        ).iloc[0]
        rows.append(chosen)
    out = pd.DataFrame(rows).sort_values("shot_style_subtype_id").reset_index(drop=True)
    return out


def select_dossier_players(final_df: pd.DataFrame, player_subtypes: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "PLAYER_ID",
        "PLAYER_NAME",
        "shot_style_subtype_id",
        "shot_style_subtype",
        "shot_style_subtype_probability",
        "total_shot_attempts_covered",
        "seasons_with_embeddings",
        "representative_score",
        "showcase_eligible",
    ]
    merged = final_df.merge(
        player_subtypes[keep_cols],
        on=["PLAYER_ID", "PLAYER_NAME", "shot_style_subtype"],
        how="left",
        suffixes=("_final", "_subtype"),
    )

    comp_col = pick_existing_col(merged.columns, ["realistic_comp_list"])
    valid_comp = merged[comp_col].fillna("").astype(str).str.strip().ne("") if comp_col else pd.Series(False, index=merged.index)

    show_col = pick_existing_col(merged.columns, ["showcase_eligible", "showcase_eligible_subtype", "showcase_eligible_final"])
    if show_col is None:
        show_flag = pd.Series(False, index=merged.index)
    else:
        show_flag = merged[show_col].astype("boolean").fillna(False).astype(bool)

    eligible = merged[valid_comp & show_flag].copy()

    rows = []
    for subtype_id, grp in eligible.groupby("shot_style_subtype_id", dropna=False):
        prob_col = pick_existing_col(
            grp.columns,
            ["shot_style_subtype_probability", "shot_style_subtype_probability_subtype", "shot_style_subtype_probability_final"],
        )
        rep_col = pick_existing_col(grp.columns, ["representative_score"])
        sim_col = pick_existing_col(grp.columns, ["realistic_comp_similarity_mean"])

        sort_cols = []
        ascending = []

        if rep_col is not None:
            sort_cols.append(rep_col)
            ascending.append(False)
        if prob_col is not None:
            sort_cols.append(prob_col)
            ascending.append(False)
        if sim_col is not None:
            sort_cols.append(sim_col)
            ascending.append(True)

        if sort_cols:
            chosen = grp.sort_values(sort_cols, ascending=ascending, na_position="last").iloc[0]
        else:
            chosen = grp.iloc[0]
        rows.append(chosen)

    if not rows:
        fallback = merged[valid_comp].copy()
        if fallback.empty:
            return pd.DataFrame()

        for subtype_id, grp in fallback.groupby("shot_style_subtype_id", dropna=False):
            prob_col = pick_existing_col(
                grp.columns,
                ["shot_style_subtype_probability", "shot_style_subtype_probability_subtype", "shot_style_subtype_probability_final"],
            )
            sim_col = pick_existing_col(grp.columns, ["realistic_comp_similarity_mean"])

            sort_cols = []
            ascending = []
            if prob_col is not None:
                sort_cols.append(prob_col)
                ascending.append(False)
            if sim_col is not None:
                sort_cols.append(sim_col)
                ascending.append(True)

            if sort_cols:
                chosen = grp.sort_values(sort_cols, ascending=ascending, na_position="last").iloc[0]
            else:
                chosen = grp.iloc[0]
            rows.append(chosen)

    out = pd.DataFrame(rows).sort_values("shot_style_subtype_id").reset_index(drop=True)
    return out


def select_drift_cases(
    drift: pd.DataFrame,
    player_subtypes: pd.DataFrame,
    season_counts: pd.DataFrame,
) -> pd.DataFrame:
    y1 = season_counts[season_counts["season_num"] == 1][["PLAYER_ID", "season_shot_attempts"]].rename(
        columns={"season_shot_attempts": "year1_shots"}
    )
    y4 = season_counts[season_counts["season_num"] == 4][["PLAYER_ID", "season_shot_attempts"]].rename(
        columns={"season_shot_attempts": "year4_shots"}
    )

    subtype_cols = [
        "PLAYER_ID",
        "PLAYER_NAME",
        "total_shot_attempts_covered",
        "seasons_with_embeddings",
        "showcase_eligible",
    ]
    subtype_small = player_subtypes[subtype_cols].copy()

    candidates = drift.merge(
        subtype_small,
        on=["PLAYER_ID", "PLAYER_NAME"],
        how="left",
        suffixes=("", "_subtype"),
    )
    candidates = candidates.merge(y1, on="PLAYER_ID", how="left")
    candidates = candidates.merge(y4, on="PLAYER_ID", how="left")

    if "total_shot_attempts_covered" not in candidates.columns and "total_shot_attempts_covered_subtype" in candidates.columns:
        candidates["total_shot_attempts_covered"] = candidates["total_shot_attempts_covered_subtype"]

    if "seasons_with_embeddings" not in candidates.columns and "seasons_with_embeddings_subtype" in candidates.columns:
        candidates["seasons_with_embeddings"] = candidates["seasons_with_embeddings_subtype"]

    if "showcase_eligible" not in candidates.columns and "showcase_eligible_subtype" in candidates.columns:
        candidates["showcase_eligible"] = candidates["showcase_eligible_subtype"]

    candidates["year1_shots"] = pd.to_numeric(candidates["year1_shots"], errors="coerce").fillna(0)
    candidates["year4_shots"] = pd.to_numeric(candidates["year4_shots"], errors="coerce").fillna(0)
    candidates["total_shot_attempts_covered"] = pd.to_numeric(
        candidates.get("total_shot_attempts_covered"), errors="coerce"
    )

    show_flag = candidates.get("showcase_eligible")
    if show_flag is None:
        show_flag = pd.Series(False, index=candidates.index)
    else:
        show_flag = show_flag.astype("boolean").fillna(False).astype(bool)

    # Strict showcase rule
    candidates["drift_showcase_eligible"] = (
        show_flag
        & (candidates["year1_shots"] >= MIN_YEAR1_SHOTS_DRIFT)
        & (candidates["year4_shots"] >= MIN_YEAR4_SHOTS_DRIFT)
    )

    # Relaxed visibility rule: must at least have nonzero shots in both Y1 and Y4
    candidates["has_visible_y1_y4_shots"] = (
        (candidates["year1_shots"] > 0) & (candidates["year4_shots"] > 0)
    )

    box_vals = pd.to_numeric(candidates["box_role_y1_to_y4_displacement"], errors="coerce").to_numpy(dtype=float)
    shot_vals = pd.to_numeric(candidates["shotstyle_y1_to_y4_displacement"], errors="coerce").to_numpy(dtype=float)
    stacked = np.column_stack([box_vals, shot_vals])
    all_nan = np.isnan(stacked).all(axis=1)
    combined = np.full(len(candidates), np.nan, dtype=float)
    valid = ~all_nan
    if valid.any():
        combined[valid] = np.nanmax(stacked[valid], axis=1)
    candidates["combined_drift"] = combined

    # Do not keep rows with no usable drift magnitude at all
    candidates = candidates[~candidates["combined_drift"].isna()].copy()

    rows = []
    target_classes = ["stable", "evolving_gradually", "role_shifting_materially"]

    for drift_class in target_classes:
        grp = candidates[candidates["identity_drift_class"] == drift_class].copy()
        if grp.empty:
            continue

        # First try strict showcase candidates, then relaxed visible-shot candidates.
        strict_pool = grp[grp["drift_showcase_eligible"]].copy()
        relaxed_pool = grp[grp["has_visible_y1_y4_shots"]].copy()

        if not strict_pool.empty:
            pool = strict_pool
        elif not relaxed_pool.empty:
            pool = relaxed_pool
        else:
            # Skip this drift class rather than plotting 0 -> 0 blank panels
            continue

        if drift_class == "stable":
            pool = pool.sort_values(
                ["combined_drift", "total_shot_attempts_covered"],
                ascending=[True, False],
                na_position="last",
            )
            chosen = pool.iloc[0]

        elif drift_class == "role_shifting_materially":
            pool = pool.sort_values(
                ["combined_drift", "total_shot_attempts_covered"],
                ascending=[False, False],
                na_position="last",
            )
            chosen = pool.iloc[0]

        else:  # evolving_gradually
            pool = pool.sort_values(
                ["combined_drift", "total_shot_attempts_covered"],
                ascending=[True, False],
                na_position="last",
            ).reset_index(drop=True)
            chosen = pool.iloc[len(pool) // 2]

        rows.append(chosen)

    return pd.DataFrame(rows)


def plot_subtype_court_panels(
    subtype_shots: pd.DataFrame,
    player_subtypes: pd.DataFrame,
    out_path,
):
    subtype_meta = (
        player_subtypes.groupby(["shot_style_subtype_id", "shot_style_subtype"], dropna=False)
        .agg(subtype_players=("PLAYER_ID", "nunique"))
        .reset_index()
        .sort_values("shot_style_subtype_id")
    )

    subtype_groups = []
    global_positive_min = None
    global_positive_max = None

    for _, meta_row in subtype_meta.iterrows():
        subtype_id = meta_row["shot_style_subtype_id"]
        subtype_name = meta_row["shot_style_subtype"]
        grp = subtype_shots[subtype_shots["shot_style_subtype_id"] == subtype_id].copy()
        if grp.empty:
            continue

        weights = np.full(len(grp), 1.0 / len(grp), dtype=float)
        mincnt = max(HEXBIN_MINCNT_FLOOR, int(HEXBIN_MINCNT_SHARE * len(grp)))

        # Precompute a reference share range for a stable shared colorbar.
        x_bins = np.linspace(-250, 250, HEXBIN_GRIDSIZE + 1)
        y_bins = np.linspace(-47.5, 422.5, HEXBIN_GRIDSIZE + 1)
        hist, _, _ = np.histogram2d(grp["LOC_X"], grp["LOC_Y"], bins=[x_bins, y_bins], weights=weights)
        raw_count_hist, _, _ = np.histogram2d(grp["LOC_X"], grp["LOC_Y"], bins=[x_bins, y_bins])
        hist = hist.T
        raw_count_hist = raw_count_hist.T
        hist[raw_count_hist < mincnt] = 0.0
        positive = hist[hist > 0]
        if positive.size:
            local_min = float(np.min(positive))
            local_max = float(np.max(positive))
            global_positive_min = local_min if global_positive_min is None else min(global_positive_min, local_min)
            global_positive_max = local_max if global_positive_max is None else max(global_positive_max, local_max)

        subtype_groups.append((subtype_id, subtype_name, grp, int(meta_row["subtype_players"]), len(grp), mincnt))

    if not subtype_groups:
        return

    if global_positive_min is None or global_positive_max is None or global_positive_min <= 0:
        global_positive_min, global_positive_max = 1e-4, 1.0

    n_panels = len(subtype_groups)
    ncols = 2
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5.4 * nrows))
    axes = np.array(axes).reshape(-1)

    mappable = ScalarMappable(norm=LogNorm(vmin=global_positive_min, vmax=global_positive_max), cmap="YlOrRd")
    mappable.set_array([])

    for ax, (subtype_id, subtype_name, grp, n_players, n_shots, mincnt) in zip(axes, subtype_groups):
        weights = np.full(len(grp), 1.0 / len(grp), dtype=float)
        hb = ax.hexbin(
            grp["LOC_X"],
            grp["LOC_Y"],
            C=weights,
            reduce_C_function=np.sum,
            gridsize=HEXBIN_GRIDSIZE,
            extent=(-250, 250, -47.5, 422.5),
            cmap="YlOrRd",
            mincnt=mincnt,
            norm=LogNorm(vmin=global_positive_min, vmax=global_positive_max),
            linewidths=0,
        )
        # draw court after the density so the court lines stay visible
        draw_court(ax, outer_lines=True)
        ax.set_title(f"{subtype_name}\nPlayers: {n_players} | Shots: {n_shots:,} | Mask: < {mincnt} shots/bin")

    for ax in axes[len(subtype_groups):]:
        ax.axis("off")

    cbar = fig.colorbar(
        mappable,
        ax=axes[:len(subtype_groups)].tolist(),
        shrink=0.62,
        fraction=0.028,
        pad=0.015,
        aspect=35,
    )
    cbar.set_label("Within-subtype shot density\n(share of subtype shots per bin; log scale)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Subtype Shot Charts on Court\nPlayer-level dominant subtype; low-density bins masked; shading is normalized within subtype.",
        y=0.975,
        fontsize=11,
    )
    fig.subplots_adjust(left=0.04, right=0.885, bottom=0.04, top=0.90, wspace=0.10, hspace=0.18)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_tensor_diagnostic(
    player_subtypes: pd.DataFrame,
    season_tensor_index_path,
    tensor_npz_path,
    out_path,
):
    season_tensor_index = pd.read_csv(season_tensor_index_path)
    tensors = np.load(tensor_npz_path)["tensors"]
    dominant = player_subtypes[["PLAYER_ID", "shot_style_subtype_id", "shot_style_subtype"]].drop_duplicates()
    subtype_tensors = season_tensor_index.merge(dominant, on="PLAYER_ID", how="left")
    tensor_groups = list(subtype_tensors.groupby(["shot_style_subtype_id", "shot_style_subtype"], dropna=False))

    if not tensor_groups:
        return

    fig, axes = plt.subplots(max(len(tensor_groups), 1), 3, figsize=(10, 3 * max(len(tensor_groups), 1)))
    if len(tensor_groups) == 1:
        axes = np.array([axes])

    for row_idx, ((_, subtype_name), grp) in enumerate(tensor_groups):
        idx = grp["tensor_index"].astype(int).to_numpy()
        avg_tensor = tensors[idx].mean(axis=0)
        for ch in range(3):
            axes[row_idx, ch].imshow(avg_tensor[ch], origin="lower", aspect="auto")
            axes[row_idx, ch].set_title(f"{subtype_name} | tensor ch{ch+1}")
            axes[row_idx, ch].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_representative_players(
    rep_players: pd.DataFrame,
    raw_shots: pd.DataFrame,
    out_path,
):
    if rep_players.empty:
        return

    n_panels = len(rep_players)
    ncols = 2
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5.2 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, (_, row) in zip(axes, rep_players.iterrows()):
        grp = raw_shots[raw_shots["PLAYER_ID"] == row["PLAYER_ID"]]
        made = grp[grp["SHOT_MADE_FLAG"] == 1]
        miss = grp[grp["SHOT_MADE_FLAG"] == 0]

        ax.scatter(miss["LOC_X"], miss["LOC_Y"], s=8, alpha=0.22, color="tab:gray", label="Miss")
        ax.scatter(made["LOC_X"], made["LOC_Y"], s=10, alpha=0.40, color="tab:blue", label="Make")
        draw_court(ax, outer_lines=True)

        ax.set_title(
            f"{row['PLAYER_NAME']}\n"
            f"{row['shot_style_subtype']} | p={row['shot_style_subtype_probability']:.2f}\n"
            f"Shots={int(row['total_shot_attempts_covered'])} | Seasons={int(row['seasons_with_embeddings'])}"
        )

    for ax in axes[len(rep_players):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_drift_case_studies(
    drift_cases: pd.DataFrame,
    raw_shots: pd.DataFrame,
    season_counts: pd.DataFrame,
    out_path,
):
    if drift_cases.empty:
        return

    valid_rows = []
    for _, case in drift_cases.iterrows():
        y1_total = int(
            season_counts[
                (season_counts["PLAYER_ID"] == case["PLAYER_ID"]) & (season_counts["season_num"] == 1)
            ]["season_shot_attempts"].sum()
        )
        y4_total = int(
            season_counts[
                (season_counts["PLAYER_ID"] == case["PLAYER_ID"]) & (season_counts["season_num"] == 4)
            ]["season_shot_attempts"].sum()
        )
        if y1_total > 0 and y4_total > 0:
            valid_rows.append(case)

    if not valid_rows:
        return

    drift_cases = pd.DataFrame(valid_rows).reset_index(drop=True)

    fig, axes = plt.subplots(len(drift_cases), 2, figsize=(12, 5.2 * max(len(drift_cases), 1)))
    if len(drift_cases) == 1:
        axes = np.array([axes])

    for row_idx, (_, case) in enumerate(drift_cases.iterrows()):
        for col_idx, season_num in enumerate([1, 4]):
            ax = axes[row_idx, col_idx]
            grp = raw_shots[
                (raw_shots["PLAYER_ID"] == case["PLAYER_ID"]) & (raw_shots["season_num"] == season_num)
            ]
            made = grp[grp["SHOT_MADE_FLAG"] == 1]
            miss = grp[grp["SHOT_MADE_FLAG"] == 0]
            season_total = int(
                season_counts[
                    (season_counts["PLAYER_ID"] == case["PLAYER_ID"]) & (season_counts["season_num"] == season_num)
                ]["season_shot_attempts"].sum()
            )

            ax.scatter(miss["LOC_X"], miss["LOC_Y"], s=8, alpha=0.18, color="tab:gray")
            ax.scatter(made["LOC_X"], made["LOC_Y"], s=10, alpha=0.35, color="tab:red")
            draw_court(ax, outer_lines=True)

            ax.set_title(
                f"{case['PLAYER_NAME']} | Y{season_num} | {case['identity_drift_class']}\n"
                f"Shots={season_total}"
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_comp_histogram(comps: pd.DataFrame, out_path):
    if comps.empty:
        return
    best_comp = comps.sort_values(["PLAYER_ID", "similarity_score"]).groupby("PLAYER_ID", as_index=False).first()
    plt.figure(figsize=(9, 5))
    plt.hist(best_comp["similarity_score"], bins=30, color="tab:purple", alpha=0.8)
    plt.xlabel("Best realistic comp similarity score (lower is stronger)")
    plt.ylabel("Players")
    plt.title("Distribution of Best Realistic Comp Support")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_dossier_view(
    dossier_sample: pd.DataFrame,
    raw_shots: pd.DataFrame,
    out_path,
):
    if dossier_sample.empty:
        return

    n_panels = len(dossier_sample)
    ncols = 2
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5.4 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, (_, row) in zip(axes, dossier_sample.iterrows()):
        grp = raw_shots[(raw_shots["PLAYER_ID"] == row["PLAYER_ID"]) & (raw_shots["season_num"].between(1, 4))]
        made = grp[grp["SHOT_MADE_FLAG"] == 1]
        miss = grp[grp["SHOT_MADE_FLAG"] == 0]

        ax.scatter(miss["LOC_X"], miss["LOC_Y"], s=8, alpha=0.18, color="tab:gray")
        ax.scatter(made["LOC_X"], made["LOC_Y"], s=10, alpha=0.30, color="tab:green")
        draw_court(ax, outer_lines=True)

        comp_text = row_value(row, ["realistic_comp_list"], default="")
        top_comp = comp_text.split(",")[0].strip() if isinstance(comp_text, str) and comp_text else "n/a"

        shots_val = row_value(
            row,
            ["total_shot_attempts_covered", "total_shot_attempts_covered_subtype", "total_shot_attempts_covered_final"],
            default=np.nan,
        )
        prob_val = row_value(
            row,
            ["shot_style_subtype_probability", "shot_style_subtype_probability_subtype", "shot_style_subtype_probability_final"],
            default=np.nan,
        )

        shots_text = "n/a" if pd.isna(shots_val) else str(int(shots_val))
        prob_text = "n/a" if pd.isna(prob_val) else f"{float(prob_val):.2f}"

        ax.set_title(
            f"{row['PLAYER_NAME']}\n"
            f"{row['macro_archetype']} | {row['shot_style_subtype']}\n"
            f"Drift: {row['identity_drift_class']} | Top comp: {top_comp}\n"
            f"Shots={shots_text} | p={prob_text}"
        )

    for ax in axes[len(dossier_sample):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_drift_scatter(drift: pd.DataFrame, out_path):
    plt.figure(figsize=(8, 5))
    plt.scatter(
        drift["box_role_y1_to_y4_displacement"],
        drift["shotstyle_y1_to_y4_displacement"],
        alpha=0.7,
    )
    plt.xlabel("Boxscore role displacement (Y1→Y4)")
    plt.ylabel("Shot-style displacement (Y1→Y4)")
    plt.title("Identity Drift Movement Plot (diagnostic)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    ensure_dirs()

    final_df = pd.read_csv(PATHS.archetype_output_dir / "final_player_archetype_profile_table.csv")
    subtype_summary = pd.read_csv(PATHS.archetype_output_dir / "shot_style_cluster_summary.csv")
    season_subtypes = pd.read_csv(PATHS.archetype_output_dir / "shot_style_player_season_table.csv")
    player_shot = pd.read_csv(PATHS.archetype_output_dir / "shot_style_player_table.csv")
    drift = pd.read_csv(PATHS.archetype_output_dir / "player_identity_drift_table.csv")
    comps = pd.read_csv(PATHS.archetype_output_dir / "realistic_comps.csv")
    raw_shots = load_main_shots()

    season_counts, raw_player_counts = compute_shot_volume_tables(raw_shots)
    player_subtypes = prepare_player_subtype_table(player_shot, raw_player_counts)

    # Macro summary
    macro_counts = final_df["macro_archetype"].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(9, 5))
    macro_counts.plot(kind="bar")
    plt.ylabel("Players")
    plt.title("Macro Archetype Summary")
    plt.tight_layout()
    macro_path = PATHS.archetype_visual_dir / "macro_archetype_summary_chart.png"
    plt.savefig(macro_path, dpi=220, bbox_inches="tight")
    plt.close()

    # Shot style subtype summary
    plt.figure(figsize=(8, 5))
    plt.bar(subtype_summary["shot_style_subtype"], subtype_summary["subtype_size"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Players")
    plt.title("Shot-Style Subtype Summary")
    plt.tight_layout()
    subtype_path = PATHS.archetype_visual_dir / "shot_style_subtype_summary_chart.png"
    plt.savefig(subtype_path, dpi=220, bbox_inches="tight")
    plt.close()

    # Technical diagnostic: embedding scatter
    emb_cols = embedding_columns(player_shot)
    coords = PCA(n_components=2, random_state=42).fit_transform(player_shot[emb_cols])
    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c=player_shot["shot_style_subtype_id"], s=22, alpha=0.8)
    plt.title("Shot Embedding Neighborhood Plot (diagnostic)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    emb_scatter_path = PATHS.archetype_visual_dir / "embedding_neighborhood_plot.png"
    plt.savefig(emb_scatter_path, dpi=220, bbox_inches="tight")
    plt.close()

    # Main subtype court plot: player-level dominant subtype
    subtype_shots = subtype_panel_candidates(player_subtypes, raw_shots)
    subtype_court_path = PATHS.archetype_visual_dir / "subtype_shot_charts_on_court.png"
    plot_subtype_court_panels(subtype_shots, player_subtypes, subtype_court_path)

    # Tensor diagnostic retained
    subtype_tensor_path = PATHS.archetype_visual_dir / "average_shot_maps_by_subtype.png"
    plot_tensor_diagnostic(
        player_subtypes,
        PATHS.archetype_output_dir / "draft_player_shot_tensor_index.csv",
        PATHS.archetype_output_dir / "draft_player_season_shot_tensors.npz",
        subtype_tensor_path,
    )

    # Representative players: probability * log1p(total shots), with thresholds
    rep_players = select_representative_players(player_subtypes)
    rep_path = PATHS.archetype_visual_dir / "representative_player_shot_scatter.png"
    plot_representative_players(rep_players, raw_shots, rep_path)

    # Better drift case selection
    drift_cases = select_drift_cases(drift, player_subtypes, season_counts)
    drift_case_path = PATHS.archetype_visual_dir / "shot_style_drift_case_studies.png"
    plot_drift_case_studies(drift_cases, raw_shots, season_counts, drift_case_path)

    # Better comp diagnostic
    comp_path = PATHS.archetype_visual_dir / "comp_comparison_visual.png"
    plot_comp_histogram(comps, comp_path)

    # Dossier sampling: one player per subtype, valid comp list, adequate volume
    dossier_sample = select_dossier_players(final_df, player_subtypes)
    dossier_path = PATHS.archetype_visual_dir / "sample_player_dossier_visual.png"
    plot_dossier_view(dossier_sample, raw_shots, dossier_path)

    # Technical diagnostic retained
    drift_path = PATHS.archetype_visual_dir / "drift_movement_plot.png"
    plot_drift_scatter(drift, drift_path)

    append_log(
        phase="PHASE 9 — VISUALS",
        completed=(
            "Revised the stakeholder-facing visuals so subtype court panels now use player-level dominant subtype, "
            "representative-player selection uses subtype probability times log shot volume with minimum thresholds, "
            "dossier sampling takes one eligible player per subtype with valid comps, "
            "subtype court panels now use stronger masking and normalized log-scaled density, "
            "and drift case studies now require usable Year 1 and Year 4 shot volume."
        ),
        learned=(
            "Player-level subtype storytelling and showcase-quality selection rules materially improve readability. "
            "Low-volume high-confidence players can distort representative panels if shot-volume thresholds are not enforced."
        ),
        assumptions=(
            "The main subtype story should be told at the player level, while season-level subtype assignments remain most useful for drift analysis. "
            "Court-based visuals prioritize readability over exhaustive coverage, so sparse players may be excluded from showcase panels."
        ),
        files_read=[
            str(PATHS.archetype_output_dir / "final_player_archetype_profile_table.csv"),
            str(PATHS.archetype_output_dir / "shot_style_cluster_summary.csv"),
            str(PATHS.archetype_output_dir / "shot_style_player_season_table.csv"),
            str(PATHS.archetype_output_dir / "shot_style_player_table.csv"),
            str(PATHS.archetype_output_dir / "player_identity_drift_table.csv"),
            str(PATHS.archetype_output_dir / "realistic_comps.csv"),
            str(PATHS.data_dir / "Shot Chart Details Raw" / "raw_shotchart_S1_to_S4_main.csv"),
            str(PATHS.archetype_output_dir / "draft_player_season_shot_tensors.npz"),
            str(PATHS.archetype_output_dir / "draft_player_shot_tensor_index.csv"),
        ],
        files_written=[
            str(macro_path),
            str(subtype_path),
            str(emb_scatter_path),
            str(subtype_court_path),
            str(subtype_tensor_path),
            str(rep_path),
            str(drift_case_path),
            str(comp_path),
            str(dossier_path),
            str(drift_path),
        ],
    )


if __name__ == "__main__":
    main()