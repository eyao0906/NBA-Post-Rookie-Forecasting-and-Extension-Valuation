
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Script location expected:
# PROJECT_ROOT / "src" / "FeaturePrep" / "build_hof_shot_feature_table.py"
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
    "avg_total_seconds_remaining",
    "q1_share",
    "q2_share",
    "q3_share",
    "q4_share",
    "ot_share",
    "early_period_share",
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
    "avg_loc_x",
    "avg_abs_loc_x",
    "avg_loc_y",
    "avg_total_seconds_remaining",
    "q1_share",
    "q4_share",
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
    "action_pullup": ["pullup"],
    "action_stepback": ["step back"],
    "action_turnaround": ["turnaround"],
    "action_floating": ["floating", "floater"],
    "action_bank": ["bank"],
    "action_alley_oop": ["alley oop"],
    "action_driving": ["driving"],
    "action_running": ["running"],
    "action_reverse": ["reverse"],
    "action_putback": ["putback"],
}

def safe_div(numerator, denominator):
    if isinstance(numerator, pd.Series) or isinstance(denominator, pd.Series):
        out = numerator / denominator
        return out.replace([np.inf, -np.inf], np.nan)
    return np.nan if denominator == 0 else numerator / denominator

def weighted_average(values: pd.Series, weights: pd.Series):
    mask = values.notna() & weights.notna()
    if mask.sum() == 0 or weights.loc[mask].sum() == 0:
        return np.nan
    return float(np.average(values.loc[mask], weights=weights.loc[mask]))

def compute_trend(group: pd.DataFrame, metric: str) -> pd.Series:
    tmp = group[["season_num", metric]].dropna()
    y1 = group.loc[group["season_num"] == 1, metric].iloc[0] if (group["season_num"] == 1).any() else np.nan
    y4 = group.loc[group["season_num"] == 4, metric].iloc[0] if (group["season_num"] == 4).any() else np.nan
    if len(tmp) < 2:
        return pd.Series({
            f"{metric}_slope": np.nan,
            f"{metric}_delta_4_1": (y4 - y1) if (pd.notna(y4) and pd.notna(y1)) else np.nan,
            f"{metric}_year1": y1,
            f"{metric}_year4": y4,
        })
    x = tmp["season_num"].astype(float).to_numpy()
    y = tmp[metric].astype(float).to_numpy()
    slope = np.polyfit(x, y, 1)[0]
    return pd.Series({
        f"{metric}_slope": float(slope),
        f"{metric}_delta_4_1": (y4 - y1) if (pd.notna(y4) and pd.notna(y1)) else np.nan,
        f"{metric}_year1": y1,
        f"{metric}_year4": y4,
    })

def normalize_game_date(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    dt = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    fallback = pd.to_datetime(s, errors="coerce")
    return dt.fillna(fallback)

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

    df["MINUTES_REMAINING"] = pd.to_numeric(df["MINUTES_REMAINING"], errors="coerce")
    df["SECONDS_REMAINING"] = pd.to_numeric(df["SECONDS_REMAINING"], errors="coerce")
    df["total_seconds_remaining"] = df["MINUTES_REMAINING"] * 60 + df["SECONDS_REMAINING"]

    df["q1_flag"] = (df["PERIOD"] == 1).astype(int)
    df["q2_flag"] = (df["PERIOD"] == 2).astype(int)
    df["q3_flag"] = (df["PERIOD"] == 3).astype(int)
    df["q4_flag"] = (df["PERIOD"] == 4).astype(int)
    df["ot_flag"] = (pd.to_numeric(df["PERIOD"], errors="coerce") >= 5).astype(int)
    df["early_period_flag"] = (df["total_seconds_remaining"] >= 8 * 60).astype(int)
    df["late_period_flag"] = (df["total_seconds_remaining"] <= 2 * 60).astype(int)
    df["last_minute_flag"] = (df["total_seconds_remaining"] <= 60).astype(int)
    df["clutch_flag"] = (((pd.to_numeric(df["PERIOD"], errors="coerce") >= 4) & (df["total_seconds_remaining"] <= 5 * 60))).astype(int)

    df["abs_loc_x"] = df["LOC_X"].abs()

def load_and_prepare(raw_path: Path, cohort_path: Path | None = None) -> pd.DataFrame:
    raw = pd.read_csv(raw_path)

    if cohort_path is not None and cohort_path.exists():
        cohort = pd.read_csv(cohort_path)
        cohort["PERSON_ID"] = pd.to_numeric(cohort["PERSON_ID"], errors="coerce").astype("Int64")
        cohort["SEASON"] = pd.to_numeric(cohort["SEASON"], errors="coerce").astype("Int64")
        raw["PLAYER_ID"] = pd.to_numeric(raw["PLAYER_ID"], errors="coerce").astype("Int64")
        raw = raw.merge(
            cohort[["PERSON_ID", "PLAYER_NAME", "SEASON"]].rename(
                columns={"PERSON_ID": "PLAYER_ID", "PLAYER_NAME": "COHORT_PLAYER_NAME", "SEASON": "DRAFT_YEAR"}
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
    raw["PERIOD"] = pd.to_numeric(raw["PERIOD"], errors="coerce")
    raw["SHOT_DISTANCE"] = pd.to_numeric(raw["SHOT_DISTANCE"], errors="coerce")
    raw["LOC_X"] = pd.to_numeric(raw["LOC_X"], errors="coerce")
    raw["LOC_Y"] = pd.to_numeric(raw["LOC_Y"], errors="coerce")
    raw["SHOT_ATTEMPTED_FLAG"] = pd.to_numeric(raw["SHOT_ATTEMPTED_FLAG"], errors="coerce").fillna(0)
    raw["SHOT_MADE_FLAG"] = pd.to_numeric(raw["SHOT_MADE_FLAG"], errors="coerce").fillna(0)
    raw["GAME_DATE"] = normalize_game_date(raw["GAME_DATE"])

    # Keep only the HOF shot-style auxiliary library.
    if "shotstyle_eligible_hof" in raw.columns:
        raw = raw[raw["shotstyle_eligible_hof"].fillna(False)].copy()
    if "is_hof" in raw.columns:
        raw = raw[raw["is_hof"].fillna(False)].copy()
    if "source_pool" in raw.columns:
        raw = raw[raw["source_pool"].fillna("") == "hof_shotstyle_subset"].copy()

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

def add_share_and_pct(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df[f"{prefix}_share"] = safe_div(df[f"{prefix}_attempts"], df["shot_attempts"])
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
            avg_seconds_remaining=("SECONDS_REMAINING", "mean"),
            avg_total_seconds_remaining=("total_seconds_remaining", "mean"),
            unique_action_types=("ACTION_TYPE", "nunique"),
        )
        .reset_index()
    )

    base["fg_pct"] = safe_div(base["shot_makes"], base["shot_attempts"])
    base["attempts_per_game"] = safe_div(base["shot_attempts"], base["games_with_shots"])
    base["makes_per_game"] = safe_div(base["shot_makes"], base["games_with_shots"])

    binary_flags_with_pct = [
        "is_2pt", "is_3pt",
        "q1_flag", "q2_flag", "q3_flag", "q4_flag", "ot_flag",
        "early_period_flag", "late_period_flag", "last_minute_flag", "clutch_flag",
        "zone_restricted_area", "zone_paint_non_ra", "zone_all_paint", "zone_midrange",
        "zone_left_corner3", "zone_right_corner3", "zone_corner3", "zone_above_break3", "zone_backcourt",
        "range_lt8", "range_8_16", "range_16_24", "range_24_plus", "range_backcourt",
        "side_left", "side_right", "side_center", "side_backcourt",
    ]

    counts = df.groupby(group_cols, dropna=False)[binary_flags_with_pct].sum().reset_index()
    counts = counts.rename(columns={c: f"{c}_attempts" for c in binary_flags_with_pct})
    base = base.merge(counts, on=group_cols, how="left")

    for flag in binary_flags_with_pct:
        tmp = (
            df.loc[df[flag] == 1]
            .groupby(group_cols, dropna=False)["SHOT_MADE_FLAG"]
            .sum()
            .reset_index()
            .rename(columns={"SHOT_MADE_FLAG": f"{flag}_makes"})
        )
        base = base.merge(tmp, on=group_cols, how="left")

    for flag in binary_flags_with_pct:
        base[f"{flag}_attempts"] = base[f"{flag}_attempts"].fillna(0)
        base[f"{flag}_makes"] = base[f"{flag}_makes"].fillna(0)
        base = add_share_and_pct(base, flag)

    base["two_pt_attempts"] = base["is_2pt_attempts"]
    base["two_pt_makes"] = base["is_2pt_makes"]
    base["two_pt_pct"] = base["is_2pt_fg_pct"]
    base["three_pt_attempts"] = base["is_3pt_attempts"]
    base["three_pt_makes"] = base["is_3pt_makes"]
    base["three_pt_pct"] = base["is_3pt_fg_pct"]
    base["three_share"] = base["is_3pt_share"]

    base["q1_share"] = base["q1_flag_share"]
    base["q2_share"] = base["q2_flag_share"]
    base["q3_share"] = base["q3_flag_share"]
    base["q4_share"] = base["q4_flag_share"]
    base["ot_share"] = base["ot_flag_share"]
    base["early_period_share"] = base["early_period_flag_share"]
    base["late_period_share"] = base["late_period_flag_share"]
    base["last_minute_share"] = base["last_minute_flag_share"]
    base["clutch_share"] = base["clutch_flag_share"]

    action_counts = df.groupby(group_cols, dropna=False)[list(ACTION_FAMILY_PATTERNS)].sum().reset_index()
    action_counts = action_counts.rename(columns={c: f"{c}_attempts" for c in ACTION_FAMILY_PATTERNS})
    base = base.merge(action_counts, on=group_cols, how="left")
    for c in ACTION_FAMILY_PATTERNS:
        base[f"{c}_attempts"] = base[f"{c}_attempts"].fillna(0)
        base[f"{c}_share"] = safe_div(base[f"{c}_attempts"], base["shot_attempts"])

    base["low_volume_lt50_flag"] = (base["shot_attempts"] < 50).astype(int)
    base["low_volume_lt100_flag"] = (base["shot_attempts"] < 100).astype(int)
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
        {"metric": "player_seasons_lt_50_attempts", "value": int((per_player_season < 50).sum())},
        {"metric": "player_seasons_lt_100_attempts", "value": int((per_player_season < 100).sum())},
    ]
    for k, v in sorted(season_counts.items()):
        rows.append({"metric": f"season_{int(k)}_attempts", "value": int(v)})
    return pd.DataFrame(rows)

def build_player_level_table(season_df: pd.DataFrame) -> pd.DataFrame:
    player_group_cols = ["PLAYER_ID", "COHORT_PLAYER_NAME", "DRAFT_YEAR"]

    player_df = (
        season_df.groupby(player_group_cols, dropna=False)
        .apply(_aggregate_player_row, include_groups=False)
        .reset_index()
    )

    season_wide = season_df[["PLAYER_ID", "season_num"] + [c for c in SEASON_BLOCK_METRICS if c in season_df.columns]].copy()
    season_wide["season_label"] = "s" + season_wide["season_num"].astype(int).astype(str)
    wide = season_wide.pivot(index="PLAYER_ID", columns="season_label", values=[c for c in SEASON_BLOCK_METRICS if c in season_wide.columns])
    wide.columns = [f"{season}_{metric}" for metric, season in wide.columns]
    wide = wide.reset_index()

    trend_frames = []
    for metric in [c for c in SLOPE_METRICS if c in season_df.columns]:
        metric_trend = (
            season_df.groupby("PLAYER_ID", dropna=False)
            .apply(lambda g: compute_trend(g, metric), include_groups=False)
            .reset_index()
        )
        trend_frames.append(metric_trend)

    if trend_frames:
        trend_df = trend_frames[0]
        for extra in trend_frames[1:]:
            trend_df = trend_df.merge(extra, on="PLAYER_ID", how="outer")
    else:
        trend_df = season_df[["PLAYER_ID"]].drop_duplicates().copy()

    recent_share = (
        season_df.assign(
            recent_attempts=lambda x: np.where(x["season_num"].isin([3, 4]), x["shot_attempts"], 0),
            recent_corner3_attempts=lambda x: np.where(x["season_num"].isin([3, 4]), x["zone_corner3_attempts"], 0),
            early_attempts=lambda x: np.where(x["season_num"].isin([1, 2]), x["shot_attempts"], 0),
            early_corner3_attempts=lambda x: np.where(x["season_num"].isin([1, 2]), x["zone_corner3_attempts"], 0),
        )
        .groupby("PLAYER_ID", dropna=False)
        .agg(
            recent_attempts=("recent_attempts", "sum"),
            recent_corner3_attempts=("recent_corner3_attempts", "sum"),
            early_attempts=("early_attempts", "sum"),
            early_corner3_attempts=("early_corner3_attempts", "sum"),
        )
        .reset_index()
    )
    recent_share["recent_attempt_share"] = safe_div(recent_share["recent_attempts"], recent_share["recent_attempts"] + recent_share["early_attempts"])
    recent_share["late_vs_early_attempt_ratio"] = safe_div(recent_share["recent_attempts"], recent_share["early_attempts"])
    recent_share["recent_corner3_share"] = safe_div(recent_share["recent_corner3_attempts"], recent_share["recent_attempts"])
    recent_share["early_corner3_share"] = safe_div(recent_share["early_corner3_attempts"], recent_share["early_attempts"])
    recent_share["corner3_share_delta_late_early"] = recent_share["recent_corner3_share"] - recent_share["early_corner3_share"]
    recent_share = recent_share.drop(columns=["recent_attempts", "recent_corner3_attempts", "early_attempts", "early_corner3_attempts"])

    player_feature_table = (
        player_df.merge(wide, on="PLAYER_ID", how="left")
        .merge(trend_df, on="PLAYER_ID", how="left")
        .merge(recent_share, on="PLAYER_ID", how="left")
        .sort_values(["DRAFT_YEAR", "PLAYER_ID"])
        .reset_index(drop=True)
    )
    return player_feature_table

def _aggregate_player_row(group: pd.DataFrame) -> pd.Series:
    attempts = group["shot_attempts"].fillna(0)
    out = {
        "seasons_with_shots": int(group["season_num"].nunique()),
        "total_shot_games": int(group["games_with_shots"].sum()),
        "total_shot_attempts": int(group["shot_attempts"].sum()),
        "total_shot_makes": int(group["shot_makes"].sum()),
        "mean_shot_attempts_per_season": float(group["shot_attempts"].mean()),
        "min_shot_attempts_in_any_season": float(group["shot_attempts"].min()),
        "max_shot_attempts_in_any_season": float(group["shot_attempts"].max()),
        "std_shot_attempts_across_seasons": float(group["shot_attempts"].std(ddof=1)) if len(group) > 1 else np.nan,
        "fg_pct_4yr": safe_div(group["shot_makes"].sum(), group["shot_attempts"].sum()),
        "avg_shot_distance_4yr": weighted_average(group["avg_shot_distance"], attempts),
        "median_season_shot_distance": float(group["median_shot_distance"].mean()),
        "std_shot_distance_4yr": weighted_average(group["std_shot_distance"].fillna(0), attempts),
        "avg_loc_x_4yr": weighted_average(group["avg_loc_x"].fillna(0), attempts),
        "median_loc_x_4yr": float(group["median_loc_x"].mean()),
        "avg_abs_loc_x_4yr": weighted_average(group["avg_abs_loc_x"].fillna(0), attempts),
        "std_loc_x_4yr": weighted_average(group["std_loc_x"].fillna(0), attempts),
        "avg_loc_y_4yr": weighted_average(group["avg_loc_y"].fillna(0), attempts),
        "median_loc_y_4yr": float(group["median_loc_y"].mean()),
        "std_loc_y_4yr": weighted_average(group["std_loc_y"].fillna(0), attempts),
        "avg_minutes_remaining_4yr": weighted_average(group["avg_minutes_remaining"], attempts),
        "avg_seconds_remaining_4yr": weighted_average(group["avg_seconds_remaining"], attempts),
        "avg_total_seconds_remaining_4yr": weighted_average(group["avg_total_seconds_remaining"], attempts),
        "three_pt_attempts_4yr": int(group["three_pt_attempts"].sum()),
        "three_pt_makes_4yr": int(group["three_pt_makes"].sum()),
        "three_pt_pct_4yr": safe_div(group["three_pt_makes"].sum(), group["three_pt_attempts"].sum()),
        "three_share_4yr": safe_div(group["three_pt_attempts"].sum(), group["shot_attempts"].sum()),
        "unique_action_types_mean": float(group["unique_action_types"].mean()),
    }

    share_metrics = [
        "q1_share","q2_share","q3_share","q4_share","ot_share",
        "early_period_share","late_period_share","last_minute_share","clutch_share",
        "zone_restricted_area_share","zone_paint_non_ra_share","zone_all_paint_share","zone_midrange_share",
        "zone_left_corner3_share","zone_right_corner3_share","zone_corner3_share","zone_above_break3_share","zone_backcourt_share",
        "range_lt8_share","range_8_16_share","range_16_24_share","range_24_plus_share","range_backcourt_share",
        "side_left_share","side_right_share","side_center_share","side_backcourt_share",
    ]
    pct_metrics = [
        "zone_restricted_area_fg_pct","zone_paint_non_ra_fg_pct","zone_all_paint_fg_pct","zone_midrange_fg_pct",
        "zone_left_corner3_fg_pct","zone_right_corner3_fg_pct","zone_corner3_fg_pct","zone_above_break3_fg_pct",
        "range_lt8_fg_pct","range_8_16_fg_pct","range_16_24_fg_pct","range_24_plus_fg_pct",
        "side_left_fg_pct","side_right_fg_pct","side_center_fg_pct",
        "two_pt_pct","three_pt_pct",
    ]
    action_share_metrics = [f"{c}_share" for c in ACTION_FAMILY_PATTERNS]

    for metric in share_metrics + pct_metrics + action_share_metrics:
        if metric in group.columns:
            out[f"{metric}_4yr"] = weighted_average(group[metric], attempts)

    return pd.Series(out)

def main():
    parser = argparse.ArgumentParser(description="Build player-season and player-level HOF shot-style feature tables from Seasons 1-4 ShotChartDetail data.")
    parser.add_argument("--raw-shotchart", type=Path, default=DATA_DIR / "raw_shotchart_S1_to_S4_hof_shotstyle.csv")
    parser.add_argument("--cohort", type=Path, default=DATA_DIR / "cohort_HOF.csv")
    parser.add_argument("--season-out", type=Path, default=OUT_DIR / "hof_player_season_shot_features.csv")
    parser.add_argument("--player-out", type=Path, default=OUT_DIR / "hof_player_shot_features_4yr.csv")
    parser.add_argument("--audit-out", type=Path, default=OUT_DIR / "hof_shotchart_audit_summary.csv")
    parser.add_argument("--counts-out", type=Path, default=OUT_DIR / "hof_player_season_shot_counts.csv")
    args = parser.parse_args()

    df = load_and_prepare(args.raw_shotchart, args.cohort)
    season_group_cols = ["PLAYER_ID", "COHORT_PLAYER_NAME", "DRAFT_YEAR", "season_num", "season_string"]
    season_df = aggregate_shot_groups(df, season_group_cols)
    player_df = build_player_level_table(season_df)
    audit_df = build_shot_audit_summary(df)

    counts_df = season_df[[
        "PLAYER_ID","COHORT_PLAYER_NAME","DRAFT_YEAR","season_num","season_string",
        "games_with_shots","shot_attempts","shot_makes","fg_pct","low_volume_lt50_flag","low_volume_lt100_flag"
    ]].copy()

    for path in [args.season_out, args.player_out, args.audit_out, args.counts_out]:
        path.parent.mkdir(parents=True, exist_ok=True)

    season_df.to_csv(args.season_out, index=False)
    player_df.to_csv(args.player_out, index=False)
    audit_df.to_csv(args.audit_out, index=False)
    counts_df.to_csv(args.counts_out, index=False)

    print(f"Saved HOF player-season shot features to: {args.season_out.resolve()}")
    print(f"Saved HOF player-level 4-year shot features to: {args.player_out.resolve()}")
    print(f"Saved HOF shotchart audit summary to: {args.audit_out.resolve()}")
    print(f"Saved HOF player-season shot counts to: {args.counts_out.resolve()}")
    print(f"Rows in HOF player-season shot table: {len(season_df):,}")
    print(f"Rows in HOF player-level shot table: {len(player_df):,}")
    print(f"Unique HOF players in cleaned shot table: {df['PLAYER_ID'].nunique():,}")

if __name__ == "__main__":
    main()
