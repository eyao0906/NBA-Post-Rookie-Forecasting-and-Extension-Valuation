import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Script location expected: PROJECT_ROOT / "src" / "FeaturePrep" / "build_shot_feature_table.py"
# Therefore PROJECT_ROOT is parents[2].
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "data/Shot Feature"

SEASON_BLOCK_METRICS = [
    "games_with_shots",
    "shot_attempts",
    "fg_pct",
    "three_share",
    "avg_shot_distance",
    "avg_loc_x",
    "avg_abs_loc_x",
    "avg_loc_y",
    "avg_minutes_remaining",
    "avg_seconds_remaining",
    "avg_total_seconds_remaining",
    "q1_share",
    "q2_share",
    "q3_share",
    "q4_share",
    "ot_share",
    "late_period_share",
    "last_minute_share",
    "clutch_share",
    "zone_restricted_area_share",
    "zone_paint_non_ra_share",
    "zone_midrange_share",
    "zone_corner3_share",
    "zone_above_break3_share",
    "side_left_share",
    "side_center_share",
    "side_right_share",
    "action_jump_share",
    "action_layup_share",
    "action_dunk_share",
    "action_pullup_share",
    "action_stepback_share",
]

SLOPE_METRICS = [
    "shot_attempts",
    "fg_pct",
    "three_share",
    "avg_shot_distance",
    "avg_abs_loc_x",
    "avg_loc_y",
    "avg_total_seconds_remaining",
    "q4_share",
    "late_period_share",
    "last_minute_share",
    "clutch_share",
    "zone_restricted_area_share",
    "zone_paint_non_ra_share",
    "zone_midrange_share",
    "zone_corner3_share",
    "zone_above_break3_share",
    "side_left_share",
    "side_center_share",
    "side_right_share",
    "action_jump_share",
    "action_layup_share",
    "action_dunk_share",
    "action_pullup_share",
    "action_stepback_share",
]

ZONE_BASIC_MAP = {
    "zone_restricted_area": "Restricted Area",
    "zone_paint_non_ra": "In The Paint (Non-RA)",
    "zone_midrange": "Mid-Range",
    "zone_left_corner3": "Left Corner 3",
    "zone_right_corner3": "Right Corner 3",
    "zone_above_break3": "Above the Break 3",
    "zone_backcourt": "Backcourt",
}

ZONE_RANGE_MAP = {
    "range_lt8": "Less Than 8 ft.",
    "range_8_16": "8-16 ft.",
    "range_16_24": "16-24 ft.",
    "range_24_plus": "24+ ft.",
    "range_backcourt": "Back Court Shot",
}

ACTION_FAMILY_PATTERNS = {
    "action_jump": ["jump"],
    "action_layup": ["layup"],
    "action_dunk": ["dunk"],
    "action_hook": ["hook"],
    "action_tip": ["tip"],
    "action_fadeaway": ["fadeaway"],
    "action_pullup": ["pullup", "pull-up", "pull up"],
    "action_stepback": ["step back", "stepback"],
    "action_turnaround": ["turnaround", "turn-around"],
    "action_floating": ["floating", "floater"],
    "action_bank": ["bank"],
    "action_alley_oop": ["alley oop", "alley-oop"],
    "action_driving": ["driving"],
    "action_running": ["running"],
    "action_reverse": ["reverse"],
    "action_putback": ["putback", "put back"],
}

ZONE_COUNT_FLAGS = [
    "is_2pt",
    "is_3pt",
    "zone_restricted_area",
    "zone_paint_non_ra",
    "zone_all_paint",
    "zone_midrange",
    "zone_left_corner3",
    "zone_right_corner3",
    "zone_corner3",
    "zone_above_break3",
    "zone_backcourt",
    "range_lt8",
    "range_8_16",
    "range_16_24",
    "range_24_plus",
    "range_backcourt",
    "side_left",
    "side_right",
    "side_center",
    "side_backcourt",
    "q1_flag",
    "q2_flag",
    "q3_flag",
    "q4_flag",
    "ot_flag",
    "early_period_flag",
    "late_period_flag",
    "last_minute_flag",
    "clutch_flag",
]


def safe_div(numerator, denominator):
    if isinstance(numerator, pd.Series) or isinstance(denominator, pd.Series):
        out = numerator / denominator
        return out.replace([np.inf, -np.inf], np.nan)
    return np.nan if denominator == 0 else numerator / denominator


def weighted_avg(series: pd.Series, weights: pd.Series) -> float:
    mask = series.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return np.nan
    return float(np.average(series[mask].astype(float), weights=weights[mask].astype(float)))


def compute_trend(group: pd.DataFrame, metric: str) -> pd.Series:
    tmp = group[["season_num", metric]].dropna()
    y1 = group.loc[group["season_num"] == 1, metric].iloc[0] if (group["season_num"] == 1).any() else np.nan
    y4 = group.loc[group["season_num"] == 4, metric].iloc[0] if (group["season_num"] == 4).any() else np.nan
    if len(tmp) < 2:
        return pd.Series(
            {
                f"{metric}_slope": np.nan,
                f"{metric}_delta_4_1": (y4 - y1) if (pd.notna(y4) and pd.notna(y1)) else np.nan,
                f"{metric}_year1": y1,
                f"{metric}_year4": y4,
            }
        )
    x = tmp["season_num"].astype(float).to_numpy()
    y = tmp[metric].astype(float).to_numpy()
    slope = np.polyfit(x, y, 1)[0]
    return pd.Series(
        {
            f"{metric}_slope": slope,
            f"{metric}_delta_4_1": (y4 - y1) if (pd.notna(y4) and pd.notna(y1)) else np.nan,
            f"{metric}_year1": y1,
            f"{metric}_year4": y4,
        }
    )


def normalize_game_date(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    dt = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    fallback = pd.to_datetime(s, errors="coerce")
    return dt.fillna(fallback)


def load_and_prepare(raw_path: Path, cohort_path: Path | None = None) -> pd.DataFrame:
    raw = pd.read_csv(raw_path)

    if cohort_path is not None and cohort_path.exists():
        cohort = pd.read_csv(cohort_path)
        cohort["PERSON_ID"] = pd.to_numeric(cohort["PERSON_ID"], errors="coerce").astype("Int64")
        cohort["SEASON"] = pd.to_numeric(cohort["SEASON"], errors="coerce").astype("Int64")
        raw["PLAYER_ID"] = pd.to_numeric(raw["PLAYER_ID"], errors="coerce").astype("Int64")
        raw = raw.merge(
            cohort[["PERSON_ID", "PLAYER_NAME", "SEASON"]].rename(
                columns={
                    "PERSON_ID": "PLAYER_ID",
                    "PLAYER_NAME": "COHORT_PLAYER_NAME",
                    "SEASON": "DRAFT_YEAR",
                }
            ),
            on="PLAYER_ID",
            how="left",
            validate="many_to_one",
        )
    else:
        raw["COHORT_PLAYER_NAME"] = raw["PLAYER_NAME"]
        raw["DRAFT_YEAR"] = pd.to_numeric(raw.get("draft_year"), errors="coerce").astype("Int64")

    raw["PLAYER_ID"] = pd.to_numeric(raw["PLAYER_ID"], errors="coerce").astype("Int64")
    raw["GAME_ID"] = pd.to_numeric(raw["GAME_ID"], errors="coerce").astype("Int64")
    raw["GAME_EVENT_ID"] = pd.to_numeric(raw["GAME_EVENT_ID"], errors="coerce").astype("Int64")
    raw["season_num"] = pd.to_numeric(raw["season_num"], errors="coerce").astype("Int64")
    raw["SHOT_DISTANCE"] = pd.to_numeric(raw["SHOT_DISTANCE"], errors="coerce")
    raw["LOC_X"] = pd.to_numeric(raw["LOC_X"], errors="coerce")
    raw["LOC_Y"] = pd.to_numeric(raw["LOC_Y"], errors="coerce")
    raw["PERIOD"] = pd.to_numeric(raw["PERIOD"], errors="coerce")
    raw["MINUTES_REMAINING"] = pd.to_numeric(raw["MINUTES_REMAINING"], errors="coerce")
    raw["SECONDS_REMAINING"] = pd.to_numeric(raw["SECONDS_REMAINING"], errors="coerce")
    raw["SHOT_ATTEMPTED_FLAG"] = pd.to_numeric(raw["SHOT_ATTEMPTED_FLAG"], errors="coerce").fillna(0)
    raw["SHOT_MADE_FLAG"] = pd.to_numeric(raw["SHOT_MADE_FLAG"], errors="coerce").fillna(0)
    raw["GAME_DATE"] = normalize_game_date(raw["GAME_DATE"])

    if "source_pool" in raw.columns:
        raw = raw[raw["source_pool"].fillna("cohort_main") == "cohort_main"].copy()
    if "is_hof" in raw.columns:
        raw = raw[~raw["is_hof"].fillna(False)].copy()

    raw = raw[raw["season_num"].between(1, 4, inclusive="both")].copy()
    raw = raw[raw["SHOT_ATTEMPTED_FLAG"] == 1].copy()
    raw = raw.sort_values(["PLAYER_ID", "GAME_ID", "GAME_EVENT_ID"]).drop_duplicates(
        subset=["PLAYER_ID", "GAME_ID", "GAME_EVENT_ID"]
    )

    raw["COHORT_PLAYER_NAME"] = raw["COHORT_PLAYER_NAME"].fillna(raw["PLAYER_NAME"])
    if "draft_year" in raw.columns:
        raw["draft_year"] = pd.to_numeric(raw["draft_year"], errors="coerce").astype("Int64")
        raw["DRAFT_YEAR"] = raw["DRAFT_YEAR"].fillna(raw["draft_year"])

    add_indicator_columns(raw)
    return raw


def add_indicator_columns(df: pd.DataFrame) -> None:
    action = df["ACTION_TYPE"].fillna("").astype(str).str.lower()
    zone_basic = df["SHOT_ZONE_BASIC"].fillna("")
    zone_area = df["SHOT_ZONE_AREA"].fillna("")
    zone_range = df["SHOT_ZONE_RANGE"].fillna("")
    shot_type = df["SHOT_TYPE"].fillna("")

    df["is_2pt"] = (shot_type == "2PT Field Goal").astype(int)
    df["is_3pt"] = (shot_type == "3PT Field Goal").astype(int)

    for col, label in ZONE_BASIC_MAP.items():
        df[col] = (zone_basic == label).astype(int)

    df["zone_corner3"] = ((zone_basic == "Left Corner 3") | (zone_basic == "Right Corner 3")).astype(int)
    df["zone_all_paint"] = ((zone_basic == "Restricted Area") | (zone_basic == "In The Paint (Non-RA)")).astype(int)

    for col, label in ZONE_RANGE_MAP.items():
        df[col] = (zone_range == label).astype(int)

    df["side_left"] = zone_area.str.contains("Left", na=False).astype(int)
    df["side_right"] = zone_area.str.contains("Right", na=False).astype(int)
    df["side_center"] = zone_area.str.contains("Center", na=False).astype(int)
    df["side_backcourt"] = zone_area.str.contains("Back Court", na=False).astype(int)

    for col, patterns in ACTION_FAMILY_PATTERNS.items():
        mask = pd.Series(False, index=df.index)
        for pat in patterns:
            mask = mask | action.str.contains(pat, regex=False, na=False)
        df[col] = mask.astype(int)

    df["abs_loc_x"] = df["LOC_X"].abs()
    df["total_seconds_remaining"] = 60.0 * df["MINUTES_REMAINING"].fillna(0) + df["SECONDS_REMAINING"].fillna(0)

    df["q1_flag"] = (df["PERIOD"] == 1).astype(int)
    df["q2_flag"] = (df["PERIOD"] == 2).astype(int)
    df["q3_flag"] = (df["PERIOD"] == 3).astype(int)
    df["q4_flag"] = (df["PERIOD"] == 4).astype(int)
    df["ot_flag"] = (df["PERIOD"] >= 5).astype(int)
    df["early_period_flag"] = (df["total_seconds_remaining"] >= 540).astype(int)
    df["late_period_flag"] = (df["total_seconds_remaining"] <= 120).astype(int)
    df["last_minute_flag"] = (df["total_seconds_remaining"] <= 60).astype(int)
    df["clutch_flag"] = ((df["PERIOD"] >= 4) & (df["total_seconds_remaining"] <= 300)).astype(int)


def flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            parts = [str(x) for x in c if x not in (None, "")]
            cols.append("_".join(parts))
        else:
            cols.append(str(c))
    df.columns = cols
    return df


def add_share_and_pct(df: pd.DataFrame, prefix: str, total_col: str = "shot_attempts") -> pd.DataFrame:
    df[f"{prefix}_share"] = safe_div(df[f"{prefix}_attempts"], df[total_col])
    df[f"{prefix}_fg_pct"] = safe_div(df[f"{prefix}_makes"], df[f"{prefix}_attempts"])
    return df


def aggregate_shot_groups(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    base = (
        df.groupby(group_cols, dropna=False)
        .agg(
            games_with_shots=("GAME_ID", "nunique"),
            shot_attempts=("SHOT_ATTEMPTED_FLAG", "sum"),
            shot_makes=("SHOT_MADE_FLAG", "sum"),
            avg_shot_distance=("SHOT_DISTANCE", "mean"),
            median_shot_distance=("SHOT_DISTANCE", "median"),
            std_shot_distance=("SHOT_DISTANCE", "std"),
            avg_loc_x=("LOC_X", "mean"),
            median_loc_x=("LOC_X", "median"),
            avg_abs_loc_x=("abs_loc_x", "mean"),
            std_loc_x=("LOC_X", "std"),
            avg_loc_y=("LOC_Y", "mean"),
            median_loc_y=("LOC_Y", "median"),
            std_loc_y=("LOC_Y", "std"),
            avg_minutes_remaining=("MINUTES_REMAINING", "mean"),
            median_minutes_remaining=("MINUTES_REMAINING", "median"),
            std_minutes_remaining=("MINUTES_REMAINING", "std"),
            avg_seconds_remaining=("SECONDS_REMAINING", "mean"),
            median_seconds_remaining=("SECONDS_REMAINING", "median"),
            std_seconds_remaining=("SECONDS_REMAINING", "std"),
            avg_total_seconds_remaining=("total_seconds_remaining", "mean"),
            median_total_seconds_remaining=("total_seconds_remaining", "median"),
            std_total_seconds_remaining=("total_seconds_remaining", "std"),
            avg_period=("PERIOD", "mean"),
            unique_action_types=("ACTION_TYPE", "nunique"),
        )
        .reset_index()
    )

    base["fg_pct"] = safe_div(base["shot_makes"], base["shot_attempts"])
    base["attempts_per_game"] = safe_div(base["shot_attempts"], base["games_with_shots"])
    base["makes_per_game"] = safe_div(base["shot_makes"], base["games_with_shots"])

    counts = df.groupby(group_cols, dropna=False)[ZONE_COUNT_FLAGS].sum().reset_index()
    counts = counts.rename(columns={c: f"{c}_attempts" for c in ZONE_COUNT_FLAGS})

    made_df = df[ZONE_COUNT_FLAGS].mul(df["SHOT_MADE_FLAG"].to_numpy(), axis=0)
    made_df.columns = [f"{c}_makes" for c in ZONE_COUNT_FLAGS]
    made_df = pd.concat([df[group_cols].reset_index(drop=True), made_df.reset_index(drop=True)], axis=1)
    made_counts = made_df.groupby(group_cols, dropna=False).sum().reset_index()

    action_counts = df.groupby(group_cols, dropna=False)[list(ACTION_FAMILY_PATTERNS)].sum().reset_index()
    action_counts = action_counts.rename(columns={c: f"{c}_attempts" for c in ACTION_FAMILY_PATTERNS})

    base = base.merge(counts, on=group_cols, how="left")
    base = base.merge(made_counts, on=group_cols, how="left")
    base = base.merge(action_counts, on=group_cols, how="left")

    share_pct_cols = {}
    for flag in ZONE_COUNT_FLAGS:
        att_col = f"{flag}_attempts"
        make_col = f"{flag}_makes"
        base[att_col] = base[att_col].fillna(0)
        base[make_col] = base[make_col].fillna(0)
        share_pct_cols[f"{flag}_share"] = safe_div(base[att_col], base["shot_attempts"])
        share_pct_cols[f"{flag}_fg_pct"] = safe_div(base[make_col], base[att_col])
    base = pd.concat([base, pd.DataFrame(share_pct_cols, index=base.index)], axis=1)

    action_share_cols = {}
    for c in ACTION_FAMILY_PATTERNS:
        att_col = f"{c}_attempts"
        base[att_col] = base[att_col].fillna(0)
        action_share_cols[f"{c}_share"] = safe_div(base[att_col], base["shot_attempts"])
    base = pd.concat([base, pd.DataFrame(action_share_cols, index=base.index)], axis=1)

    alias_cols = {
        "two_pt_attempts": base["is_2pt_attempts"],
        "two_pt_makes": base["is_2pt_makes"],
        "two_pt_pct": base["is_2pt_fg_pct"],
        "three_pt_attempts": base["is_3pt_attempts"],
        "three_pt_makes": base["is_3pt_makes"],
        "three_pt_pct": base["is_3pt_fg_pct"],
        "three_share": base["is_3pt_share"],
        "q1_share": base["q1_flag_share"],
        "q2_share": base["q2_flag_share"],
        "q3_share": base["q3_flag_share"],
        "q4_share": base["q4_flag_share"],
        "ot_share": base["ot_flag_share"],
        "early_period_share": base["early_period_flag_share"],
        "late_period_share": base["late_period_flag_share"],
        "last_minute_share": base["last_minute_flag_share"],
        "clutch_share": base["clutch_flag_share"],
        "low_volume_lt25_flag": (base["shot_attempts"] < 25).astype(int),
        "low_volume_lt50_flag": (base["shot_attempts"] < 50).astype(int),
        "low_volume_lt100_flag": (base["shot_attempts"] < 100).astype(int),
    }
    base = pd.concat([base, pd.DataFrame(alias_cols, index=base.index)], axis=1)
    return base


def build_shot_audit_summary(df: pd.DataFrame) -> pd.DataFrame:
    duplicate_events = int(df.duplicated(["PLAYER_ID", "GAME_ID", "GAME_EVENT_ID"]).sum())
    per_player = df.groupby("PLAYER_ID", dropna=False)["SHOT_ATTEMPTED_FLAG"].sum()
    per_player_season = df.groupby(["PLAYER_ID", "season_num"], dropna=False)["SHOT_ATTEMPTED_FLAG"].sum()
    season_counts = df.groupby("season_num", dropna=False)["SHOT_ATTEMPTED_FLAG"].sum().to_dict()

    rows = [
        {"metric": "rows_after_cleaning", "value": int(len(df))},
        {"metric": "unique_players", "value": int(df["PLAYER_ID"].nunique())},
        {"metric": "unique_player_seasons", "value": int(df[["PLAYER_ID", "season_num"]].drop_duplicates().shape[0])},
        {"metric": "unique_games", "value": int(df["GAME_ID"].nunique())},
        {"metric": "duplicate_player_game_event_rows_after_cleaning", "value": duplicate_events},
        {"metric": "missing_loc_x_rows", "value": int(df["LOC_X"].isna().sum())},
        {"metric": "missing_loc_y_rows", "value": int(df["LOC_Y"].isna().sum())},
        {"metric": "missing_shot_distance_rows", "value": int(df["SHOT_DISTANCE"].isna().sum())},
        {"metric": "missing_minutes_remaining_rows", "value": int(df["MINUTES_REMAINING"].isna().sum())},
        {"metric": "missing_seconds_remaining_rows", "value": int(df["SECONDS_REMAINING"].isna().sum())},
        {"metric": "players_lt_50_total_attempts", "value": int((per_player < 50).sum())},
        {"metric": "players_lt_100_total_attempts", "value": int((per_player < 100).sum())},
        {"metric": "player_seasons_lt_25_attempts", "value": int((per_player_season < 25).sum())},
        {"metric": "player_seasons_lt_50_attempts", "value": int((per_player_season < 50).sum())},
        {"metric": "player_seasons_lt_100_attempts", "value": int((per_player_season < 100).sum())},
    ]
    for k, v in sorted(season_counts.items()):
        rows.append({"metric": f"season_{int(k)}_attempts", "value": int(v)})
    return pd.DataFrame(rows)


def build_player_level_table(season_df: pd.DataFrame) -> pd.DataFrame:
    player_group_cols = ["PLAYER_ID", "COHORT_PLAYER_NAME", "DRAFT_YEAR"]
    weighted_cols = [
        "avg_shot_distance",
        "median_shot_distance",
        "std_shot_distance",
        "avg_loc_x",
        "median_loc_x",
        "avg_abs_loc_x",
        "std_loc_x",
        "avg_loc_y",
        "median_loc_y",
        "std_loc_y",
        "avg_minutes_remaining",
        "median_minutes_remaining",
        "std_minutes_remaining",
        "avg_seconds_remaining",
        "median_seconds_remaining",
        "std_seconds_remaining",
        "avg_total_seconds_remaining",
        "median_total_seconds_remaining",
        "std_total_seconds_remaining",
        "avg_period",
    ]

    player_df = (
        season_df.groupby(player_group_cols, dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "seasons_with_shots": g["season_num"].nunique(),
                    "total_shot_games": g["games_with_shots"].sum(),
                    "total_shot_attempts": g["shot_attempts"].sum(),
                    "total_shot_makes": g["shot_makes"].sum(),
                    "mean_shot_attempts_per_season": g["shot_attempts"].mean(),
                    "min_shot_attempts_in_any_season": g["shot_attempts"].min(),
                    "max_shot_attempts_in_any_season": g["shot_attempts"].max(),
                    "std_shot_attempts_across_seasons": g["shot_attempts"].std(),
                    "unique_action_types_mean": g["unique_action_types"].mean(),
                    "low_volume_seasons_lt25": g["low_volume_lt25_flag"].sum(),
                    "low_volume_seasons_lt50": g["low_volume_lt50_flag"].sum(),
                    "low_volume_seasons_lt100": g["low_volume_lt100_flag"].sum(),
                    **{col + "_4yr": weighted_avg(g[col], g["shot_attempts"]) for col in weighted_cols},
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    aggregate_cols = [
        "games_with_shots",
        "shot_attempts",
        "shot_makes",
        "two_pt_attempts",
        "two_pt_makes",
        "three_pt_attempts",
        "three_pt_makes",
    ]
    aggregate_cols += [f"{c}_attempts" for c in ZONE_COUNT_FLAGS]
    aggregate_cols += [f"{c}_makes" for c in ZONE_COUNT_FLAGS]
    aggregate_cols += [f"{c}_attempts" for c in ACTION_FAMILY_PATTERNS]

    summed = season_df[player_group_cols + aggregate_cols].groupby(player_group_cols, dropna=False).sum().reset_index()
    player_df = player_df.merge(summed, on=player_group_cols, how="left")

    player_df["fg_pct_4yr"] = safe_div(player_df["total_shot_makes"], player_df["total_shot_attempts"])
    player_df["two_pt_pct_4yr"] = safe_div(player_df["two_pt_makes"], player_df["two_pt_attempts"])
    player_df["three_pt_pct_4yr"] = safe_div(player_df["three_pt_makes"], player_df["three_pt_attempts"])
    player_df["three_share_4yr"] = safe_div(player_df["three_pt_attempts"], player_df["total_shot_attempts"])
    player_df["attempts_per_game_4yr"] = safe_div(player_df["total_shot_attempts"], player_df["total_shot_games"])
    player_df["makes_per_game_4yr"] = safe_div(player_df["total_shot_makes"], player_df["total_shot_games"])

    for flag in ZONE_COUNT_FLAGS:
        player_df[f"{flag}_share_4yr"] = safe_div(player_df[f"{flag}_attempts"], player_df["total_shot_attempts"])
        player_df[f"{flag}_fg_pct_4yr"] = safe_div(player_df[f"{flag}_makes"], player_df[f"{flag}_attempts"])

    for action_flag in ACTION_FAMILY_PATTERNS:
        player_df[f"{action_flag}_share_4yr"] = safe_div(player_df[f"{action_flag}_attempts"], player_df["total_shot_attempts"])

    player_df["q1_share_4yr"] = player_df["q1_flag_share_4yr"]
    player_df["q2_share_4yr"] = player_df["q2_flag_share_4yr"]
    player_df["q3_share_4yr"] = player_df["q3_flag_share_4yr"]
    player_df["q4_share_4yr"] = player_df["q4_flag_share_4yr"]
    player_df["ot_share_4yr"] = player_df["ot_flag_share_4yr"]
    player_df["early_period_share_4yr"] = player_df["early_period_flag_share_4yr"]
    player_df["late_period_share_4yr"] = player_df["late_period_flag_share_4yr"]
    player_df["last_minute_share_4yr"] = player_df["last_minute_flag_share_4yr"]
    player_df["clutch_share_4yr"] = player_df["clutch_flag_share_4yr"]

    season_wide = season_df[["PLAYER_ID", "season_num"] + [c for c in SEASON_BLOCK_METRICS if c in season_df.columns]].copy()
    season_wide["season_label"] = "s" + season_wide["season_num"].astype(int).astype(str)
    wide = season_wide.pivot(index="PLAYER_ID", columns="season_label", values=[c for c in SEASON_BLOCK_METRICS if c in season_df.columns])
    wide = flatten_multiindex_columns(wide.reset_index())

    trend_df = pd.DataFrame({"PLAYER_ID": season_df["PLAYER_ID"].drop_duplicates().sort_values().to_list()})
    x = np.array([1.0, 2.0, 3.0, 4.0])
    for metric in SLOPE_METRICS:
        metric_pivot = season_df.pivot(index="PLAYER_ID", columns="season_num", values=metric).reindex(columns=[1, 2, 3, 4])
        metric_pivot = metric_pivot.reindex(trend_df["PLAYER_ID"])
        y = metric_pivot.to_numpy(dtype=float)
        mask = ~np.isnan(y)
        n = mask.sum(axis=1).astype(float)
        sum_x = (mask * x).sum(axis=1)
        sum_y = np.nansum(y, axis=1)
        sum_xy = np.nansum(y * x, axis=1)
        sum_x2 = (mask * (x ** 2)).sum(axis=1)
        denom = n * sum_x2 - sum_x ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            slope = np.where((n >= 2) & (denom != 0), (n * sum_xy - sum_x * sum_y) / denom, np.nan)
        year1 = metric_pivot[1].to_numpy(dtype=float)
        year4 = metric_pivot[4].to_numpy(dtype=float)
        delta = np.where(~np.isnan(year1) & ~np.isnan(year4), year4 - year1, np.nan)
        trend_df[f"{metric}_slope"] = slope
        trend_df[f"{metric}_delta_4_1"] = delta
        trend_df[f"{metric}_year1"] = year1
        trend_df[f"{metric}_year4"] = year4

    recent_share = (
        season_df.assign(
            recent_attempts=lambda x: np.where(x["season_num"].isin([3, 4]), x["shot_attempts"], 0),
            recent_corner3_attempts=lambda x: np.where(x["season_num"].isin([3, 4]), x["zone_corner3_attempts"], 0),
            recent_clutch_attempts=lambda x: np.where(x["season_num"].isin([3, 4]), x["clutch_flag_attempts"], 0),
            early_attempts=lambda x: np.where(x["season_num"].isin([1, 2]), x["shot_attempts"], 0),
            early_corner3_attempts=lambda x: np.where(x["season_num"].isin([1, 2]), x["zone_corner3_attempts"], 0),
            early_clutch_attempts=lambda x: np.where(x["season_num"].isin([1, 2]), x["clutch_flag_attempts"], 0),
        )
        .groupby("PLAYER_ID", dropna=False)
        .agg(
            recent_attempts=("recent_attempts", "sum"),
            recent_corner3_attempts=("recent_corner3_attempts", "sum"),
            recent_clutch_attempts=("recent_clutch_attempts", "sum"),
            early_attempts=("early_attempts", "sum"),
            early_corner3_attempts=("early_corner3_attempts", "sum"),
            early_clutch_attempts=("early_clutch_attempts", "sum"),
        )
        .reset_index()
    )
    recent_share["recent_attempt_share"] = safe_div(recent_share["recent_attempts"], recent_share["recent_attempts"] + recent_share["early_attempts"])
    recent_share["late_vs_early_attempt_ratio"] = safe_div(recent_share["recent_attempts"], recent_share["early_attempts"])
    recent_share["recent_corner3_share"] = safe_div(recent_share["recent_corner3_attempts"], recent_share["recent_attempts"])
    recent_share["early_corner3_share"] = safe_div(recent_share["early_corner3_attempts"], recent_share["early_attempts"])
    recent_share["corner3_share_delta_late_early"] = recent_share["recent_corner3_share"] - recent_share["early_corner3_share"]
    recent_share["recent_clutch_share"] = safe_div(recent_share["recent_clutch_attempts"], recent_share["recent_attempts"])
    recent_share["early_clutch_share"] = safe_div(recent_share["early_clutch_attempts"], recent_share["early_attempts"])
    recent_share["clutch_share_delta_late_early"] = recent_share["recent_clutch_share"] - recent_share["early_clutch_share"]
    recent_share = recent_share.drop(
        columns=[
            "recent_attempts",
            "recent_corner3_attempts",
            "recent_clutch_attempts",
            "early_attempts",
            "early_corner3_attempts",
            "early_clutch_attempts",
        ]
    )

    player_feature_table = (
        player_df.merge(wide, on="PLAYER_ID", how="left")
        .merge(trend_df, on="PLAYER_ID", how="left")
        .merge(recent_share, on="PLAYER_ID", how="left")
        .sort_values(["DRAFT_YEAR", "PLAYER_ID"])
        .reset_index(drop=True)
    )

    return player_feature_table


def build_counts_df(season_df: pd.DataFrame) -> pd.DataFrame:
    return season_df[
        [
            "PLAYER_ID",
            "COHORT_PLAYER_NAME",
            "DRAFT_YEAR",
            "season_num",
            "season_string",
            "games_with_shots",
            "shot_attempts",
            "shot_makes",
            "fg_pct",
            "avg_loc_x",
            "avg_loc_y",
            "avg_total_seconds_remaining",
            "low_volume_lt25_flag",
            "low_volume_lt50_flag",
            "low_volume_lt100_flag",
        ]
    ].copy()

def main():
    parser = argparse.ArgumentParser(
        description="Build game-level, player-season, and player-level shot-style feature tables from Seasons 1-4 ShotChartDetail data."
    )
    parser.add_argument("--raw-shotchart", type=Path, default=DATA_DIR / "raw_shotchart_S1_to_S4_main.csv")
    parser.add_argument("--cohort", type=Path, default=DATA_DIR / "cohort_1999_2019.csv")
    parser.add_argument("--game-out", type=Path, default=None, help="Optional smaller-unit output: one row per Player_ID x Game_ID.")
    parser.add_argument("--season-out", type=Path, default=OUT_DIR / "player_season_shot_features.csv")
    parser.add_argument("--player-out", type=Path, default=OUT_DIR / "player_shot_features_4yr.csv")
    parser.add_argument("--audit-out", type=Path, default=OUT_DIR / "shotchart_audit_summary.csv")
    parser.add_argument("--counts-out", type=Path, default=OUT_DIR / "player_season_shot_counts.csv")
    args = parser.parse_args()

    df = load_and_prepare(args.raw_shotchart, args.cohort)

    game_group_cols = [
        "PLAYER_ID",
        "COHORT_PLAYER_NAME",
        "DRAFT_YEAR",
        "season_num",
        "season_string",
        "GAME_ID",
        "GAME_DATE",
    ]
    season_group_cols = [
        "PLAYER_ID",
        "COHORT_PLAYER_NAME",
        "DRAFT_YEAR",
        "season_num",
        "season_string",
    ]

    game_df = aggregate_shot_groups(df, game_group_cols) if args.game_out is not None else None
    season_df = aggregate_shot_groups(df, season_group_cols)
    player_df = build_player_level_table(season_df)
    audit_df = build_shot_audit_summary(df)
    counts_df = build_counts_df(season_df)

    output_paths = [args.season_out, args.player_out, args.audit_out, args.counts_out]
    if args.game_out is not None:
        output_paths.append(args.game_out)
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    if args.game_out is not None and game_df is not None:
        game_df.to_csv(args.game_out, index=False)
    season_df.to_csv(args.season_out, index=False)
    player_df.to_csv(args.player_out, index=False)
    audit_df.to_csv(args.audit_out, index=False)
    counts_df.to_csv(args.counts_out, index=False)

    if args.game_out is not None:
        print(f"Saved player-game shot features to: {args.game_out.resolve()}")
    print(f"Saved player-season shot features to: {args.season_out.resolve()}")
    print(f"Saved player-level 4-year shot features to: {args.player_out.resolve()}")
    print(f"Saved shotchart audit summary to: {args.audit_out.resolve()}")
    print(f"Saved player-season shot counts to: {args.counts_out.resolve()}")
    if game_df is not None:
        print(f"Rows in player-game shot table: {len(game_df):,}")
    print(f"Rows in player-season shot table: {len(season_df):,}")
    print(f"Rows in player-level shot table: {len(player_df):,}")
    print(f"Unique players in cleaned shot table: {df['PLAYER_ID'].nunique():,}")


if __name__ == "__main__":
    main()
