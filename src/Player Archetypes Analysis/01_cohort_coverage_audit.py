from __future__ import annotations

import pandas as pd

from archetype_workflow_utils import PATHS, add_name_key, append_log, ensure_dirs


DRAFT_FILES = {
    "cohort": PATHS.data_dir / "cohort_1999_2019.csv",
    "logs": PATHS.data_dir / "raw_game_logs_S1_to_S4.csv",
    "targets": PATHS.data_dir / "career_totals_targets.csv",
    "player_features": PATHS.data_dir / "player_feature_table_1999_2019.csv",
    "season_features": PATHS.data_dir / "player_season_feature_table_1999_2019.csv",
    "clustered": PATHS.kmeans_dir / "player_feature_table_1999_2019_clustered_k5.csv",
    "shot": PATHS.data_dir / "Shot Chart Details Raw" / "raw_shotchart_S1_to_S4_main.csv",
}

HOF_FILES = {
    "cohort": PATHS.data_dir / "cohort_HOF.csv",
    "logs": PATHS.data_dir / "HOF_raw_s1_s4_game_logs.csv",
    "targets": PATHS.data_dir / "HOF_career_total_targets.csv",
    "shot": PATHS.data_dir / "Shot Chart Details Raw" / "raw_shotchart_S1_to_S4_hof_shotstyle.csv",
}


def unique_players(df: pd.DataFrame, id_cols: list[str], name_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in id_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in name_cols:
        if col in out.columns:
            out[col] = out[col].astype(str)
    use_cols = [c for c in id_cols + name_cols if c in out.columns]
    out = out[use_cols].drop_duplicates().copy()
    if "PLAYER_ID" not in out.columns and "PERSON_ID" in out.columns:
        out = out.rename(columns={"PERSON_ID": "PLAYER_ID"})
    if "PLAYER_NAME" not in out.columns and "COHORT_PLAYER_NAME" in out.columns:
        out = out.rename(columns={"COHORT_PLAYER_NAME": "PLAYER_NAME"})
    if "PLAYER_NAME" not in out.columns and "PLAYER_NAME" not in out.columns:
        raise ValueError("A player-name column is required for coverage audit.")
    return add_name_key(out, "PLAYER_NAME")


def drafted_cohort() -> pd.DataFrame:
    cohort = pd.read_csv(DRAFT_FILES["cohort"]).rename(columns={"PERSON_ID": "PLAYER_ID", "SEASON": "DRAFT_YEAR"})
    cohort = cohort[["PLAYER_ID", "PLAYER_NAME", "DRAFT_YEAR"]].drop_duplicates()
    cohort["PLAYER_ID"] = pd.to_numeric(cohort["PLAYER_ID"], errors="coerce")
    return add_name_key(cohort, "PLAYER_NAME")


def hof_cohort() -> pd.DataFrame:
    cohort = pd.read_csv(HOF_FILES["cohort"]).rename(columns={"PERSON_ID": "PLAYER_ID", "SEASON": "DRAFT_YEAR"})
    cohort = cohort[["PLAYER_ID", "PLAYER_NAME", "DRAFT_YEAR"]].drop_duplicates()
    cohort["PLAYER_ID"] = pd.to_numeric(cohort["PLAYER_ID"], errors="coerce")
    return add_name_key(cohort, "PLAYER_NAME")


def build_audit(cohort: pd.DataFrame, source_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = cohort[["PLAYER_ID", "PLAYER_NAME", "DRAFT_YEAR", "name_key"]].drop_duplicates().copy()
    for label, source_df in source_map.items():
        source_small = source_df[[c for c in ["PLAYER_ID", "name_key"] if c in source_df.columns]].drop_duplicates()
        if "PLAYER_ID" in source_small.columns:
            out = out.merge(source_small.assign(**{f"has_{label}": 1}), on=[c for c in ["PLAYER_ID", "name_key"] if c in source_small.columns], how="left")
        else:
            out = out.merge(source_small.assign(**{f"has_{label}": 1}), on="name_key", how="left")
        out[f"has_{label}"] = out[f"has_{label}"].fillna(0).astype(int)
    return out


def summary_table(audit_df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    total_players = len(audit_df)
    rows = [{"metric": "total_players", "count": total_players}]
    for label in labels:
        rows.append({"metric": f"players_with_{label}", "count": int(audit_df[f"has_{label}"].sum())})
        rows.append({"metric": f"players_missing_{label}", "count": int(total_players - audit_df[f"has_{label}"].sum())})
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()

    draft_sources = {
        "s1_s4_logs": unique_players(pd.read_csv(DRAFT_FILES["logs"]), ["Player_ID"], ["PLAYER_NAME"]).rename(columns={"Player_ID": "PLAYER_ID"}),
        "later_career_targets": unique_players(pd.read_csv(DRAFT_FILES["targets"]), ["PLAYER_ID"], ["PLAYER_NAME"]),
        "player_features": unique_players(pd.read_csv(DRAFT_FILES["player_features"]), ["Player_ID"], ["COHORT_PLAYER_NAME"]).rename(columns={"Player_ID": "PLAYER_ID", "COHORT_PLAYER_NAME": "PLAYER_NAME"}),
        "season_features": unique_players(pd.read_csv(DRAFT_FILES["season_features"]), ["Player_ID"], ["COHORT_PLAYER_NAME"]).rename(columns={"Player_ID": "PLAYER_ID", "COHORT_PLAYER_NAME": "PLAYER_NAME"}),
        "cluster_assignments": unique_players(pd.read_csv(DRAFT_FILES["clustered"]), ["Player_ID"], ["COHORT_PLAYER_NAME"]).rename(columns={"Player_ID": "PLAYER_ID", "COHORT_PLAYER_NAME": "PLAYER_NAME"}),
        "shotchart_rows": unique_players(pd.read_csv(DRAFT_FILES["shot"]), ["PLAYER_ID"], ["PLAYER_NAME"]),
    }
    hof_sources = {
        "s1_s4_logs": unique_players(pd.read_csv(HOF_FILES["logs"]), ["Player_ID"], ["PLAYER_NAME"]).rename(columns={"Player_ID": "PLAYER_ID"}),
        "later_career_targets": unique_players(pd.read_csv(HOF_FILES["targets"]), ["PLAYER_ID"], ["PLAYER_NAME"]),
        "shotchart_rows": unique_players(pd.read_csv(HOF_FILES["shot"]), ["PLAYER_ID"], ["PLAYER_NAME"]),
    }

    draft_audit = build_audit(drafted_cohort(), draft_sources)
    hof_audit = build_audit(hof_cohort(), hof_sources)

    drafted_summary = summary_table(draft_audit, list(draft_sources))
    hof_summary = summary_table(hof_audit, list(hof_sources))

    cohort_join_audit = draft_audit.copy()
    coverage_cols = [c for c in draft_audit.columns if c.startswith("has_")]
    cohort_join_audit["missing_stage_count"] = (1 - cohort_join_audit[coverage_cols]).sum(axis=1)
    cohort_join_audit["fully_covered_for_hybrid_pipeline"] = (
        cohort_join_audit[["has_s1_s4_logs", "has_player_features", "has_season_features", "has_cluster_assignments", "has_shotchart_rows", "has_later_career_targets"]].min(axis=1)
    )

    drafted_path = PATHS.archetype_output_dir / "drafted_player_coverage_audit.csv"
    hof_path = PATHS.archetype_output_dir / "hof_player_coverage_audit.csv"
    join_path = PATHS.archetype_output_dir / "cohort_join_audit.csv"

    drafted_summary.to_csv(drafted_path, index=False)
    hof_summary.to_csv(hof_path, index=False)
    cohort_join_audit.to_csv(join_path, index=False)

    append_log(
        phase="PHASE 1 — COHORT AND COVERAGE AUDIT",
        completed=(
            "Constructed drafted-player and Hall of Fame coverage audits across cohort membership, raw logs, target tables, feature tables, cluster assignments, and restored shot-chart coverage. "
            "Saved clean audit tables and a player-level join audit that tracks shrinkage and missing stages."
        ),
        learned=(
            "The drafted cohort has broad coverage across the existing boxscore and clustering pipeline, but not every player has complete shot-chart and target support. "
            "The HOF library is much smaller and should be treated as an auxiliary ceiling-comp pool rather than a realistic-comp source."
        ),
        assumptions=(
            "Coverage is keyed primarily by PLAYER_ID; normalized player names are used only as a support key. "
            "Players missing shot-chart rows or later-career targets should remain documented rather than silently dropped."
        ),
        files_read=[str(v) for v in DRAFT_FILES.values()] + [str(v) for v in HOF_FILES.values()],
        files_written=[str(drafted_path), str(hof_path), str(join_path)],
    )


if __name__ == "__main__":
    main()
