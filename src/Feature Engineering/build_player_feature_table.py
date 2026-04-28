import argparse
from pathlib import Path
import numpy as np
import pandas as pd

SEASON_WIDE_METRICS = [
    "games", "total_minutes", "mpg", "median_min", "min_std",
    "pct_10plus_min", "pct_20plus_min", "pct_25plus_min", "pct_30plus_min",
    "win_pct", "home_share", "avg_rest_days", "b2b_rate", "long_rest_rate",
    "plus_minus_mean", "plus_minus_std",
    "fg_pct", "fg3_pct", "ft_pct", "ts_pct", "fg3a_rate", "ftr", "ast_tov_ratio",
    "pts_pg", "reb_pg", "ast_pg", "stl_pg", "blk_pg", "tov_pg",
    "fga_pg", "fta_pg", "usage_proxy_pg",
    "pts_per36", "reb_per36", "ast_per36", "stl_per36", "blk_per36", "tov_per36",
    "fga_per36", "fta_per36", "usage_proxy_per36",
]

SLOPE_METRICS = [
    "games", "mpg", "pts_per36", "reb_per36", "ast_per36", "stl_per36", "blk_per36",
    "tov_per36", "fga_per36", "usage_proxy_per36", "ts_pct", "fg3a_rate", "ftr",
    "win_pct", "plus_minus_mean", "pct_25plus_min", "pct_30plus_min"
]


def safe_div(numerator, denominator):
    if isinstance(numerator, pd.Series) or isinstance(denominator, pd.Series):
        out = numerator / denominator
        return out.replace([np.inf, -np.inf], np.nan)
    return np.nan if denominator == 0 else numerator / denominator


def season_start_from_string(season_string: str) -> float:
    if pd.isna(season_string):
        return np.nan
    s = str(season_string)
    try:
        return int(s[:4])
    except Exception:
        return np.nan


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
        f"{metric}_slope": slope,
        f"{metric}_delta_4_1": (y4 - y1) if (pd.notna(y4) and pd.notna(y1)) else np.nan,
        f"{metric}_year1": y1,
        f"{metric}_year4": y4,
    })


def load_and_prepare(raw_path: Path, cohort_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(raw_path)
    cohort = pd.read_csv(cohort_path)

    raw["Player_ID"] = pd.to_numeric(raw["Player_ID"], errors="coerce").astype("Int64")
    cohort["PERSON_ID"] = pd.to_numeric(cohort["PERSON_ID"], errors="coerce").astype("Int64")
    cohort["SEASON"] = pd.to_numeric(cohort["SEASON"], errors="coerce").astype("Int64")

    df = raw.merge(
        cohort[["PERSON_ID", "PLAYER_NAME", "SEASON"]].rename(
            columns={"PERSON_ID": "Player_ID", "PLAYER_NAME": "COHORT_PLAYER_NAME", "SEASON": "DRAFT_YEAR"}
        ),
        on="Player_ID",
        how="inner",
        validate="many_to_one"
    ).copy()

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df["season_start_year"] = df["SEASON_STRING"].map(season_start_from_string).astype("Int64")
    df["DRAFT_YEAR"] = df["DRAFT_YEAR"].astype("Int64")
    df["season_num"] = (df["season_start_year"] - df["DRAFT_YEAR"] + 1).astype("Int64")

    df = df[df["season_num"].between(1, 4, inclusive="both")].copy()
    df = df.sort_values(["Player_ID", "GAME_DATE", "Game_ID"]).drop_duplicates(subset=["Player_ID", "Game_ID"])

    df["win_flag"] = (df["WL"] == "W").astype(int)
    df["home_flag"] = df["MATCHUP"].astype(str).str.contains("vs.", regex=False).astype(int)
    df["usage_proxy"] = df["FGA"] + 0.44 * df["FTA"] + df["TOV"]

    df = df.sort_values(["Player_ID", "GAME_DATE", "Game_ID"]).copy()
    df["rest_days"] = df.groupby("Player_ID")["GAME_DATE"].diff().dt.days
    df["b2b_flag"] = (df["rest_days"] == 1).astype(float)
    df["long_rest_flag"] = (df["rest_days"] >= 4).astype(float)

    return df


def aggregate_player_season(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["Player_ID", "COHORT_PLAYER_NAME", "DRAFT_YEAR", "season_num", "SEASON_STRING"]

    season_df = (
        df.groupby(group_cols, dropna=False)
          .agg(
              games=("Game_ID", "nunique"),
              total_minutes=("MIN", "sum"),
              mpg=("MIN", "mean"),
              median_min=("MIN", "median"),
              min_std=("MIN", "std"),
              pct_10plus_min=("MIN", lambda s: np.mean(s >= 10)),
              pct_20plus_min=("MIN", lambda s: np.mean(s >= 20)),
              pct_25plus_min=("MIN", lambda s: np.mean(s >= 25)),
              pct_30plus_min=("MIN", lambda s: np.mean(s >= 30)),
              win_pct=("win_flag", "mean"),
              home_share=("home_flag", "mean"),
              avg_rest_days=("rest_days", "mean"),
              b2b_rate=("b2b_flag", "mean"),
              long_rest_rate=("long_rest_flag", "mean"),
              plus_minus_mean=("PLUS_MINUS", "mean"),
              plus_minus_std=("PLUS_MINUS", "std"),
              FGM=("FGM", "sum"),
              FGA=("FGA", "sum"),
              FG3M=("FG3M", "sum"),
              FG3A=("FG3A", "sum"),
              FTM=("FTM", "sum"),
              FTA=("FTA", "sum"),
              OREB=("OREB", "sum"),
              DREB=("DREB", "sum"),
              REB=("REB", "sum"),
              AST=("AST", "sum"),
              STL=("STL", "sum"),
              BLK=("BLK", "sum"),
              TOV=("TOV", "sum"),
              PF=("PF", "sum"),
              PTS=("PTS", "sum"),
              usage_proxy=("usage_proxy", "sum"),
          )
          .reset_index()
    )

    season_df["fg_pct"] = safe_div(season_df["FGM"], season_df["FGA"])
    season_df["fg3_pct"] = safe_div(season_df["FG3M"], season_df["FG3A"])
    season_df["ft_pct"] = safe_div(season_df["FTM"], season_df["FTA"])
    season_df["ts_pct"] = safe_div(season_df["PTS"], 2 * (season_df["FGA"] + 0.44 * season_df["FTA"]))
    season_df["fg3a_rate"] = safe_div(season_df["FG3A"], season_df["FGA"])
    season_df["ftr"] = safe_div(season_df["FTA"], season_df["FGA"])
    season_df["ast_tov_ratio"] = safe_div(season_df["AST"], season_df["TOV"])

    for new_col, base_col in {
        "pts_pg": "PTS", "reb_pg": "REB", "ast_pg": "AST", "stl_pg": "STL",
        "blk_pg": "BLK", "tov_pg": "TOV", "fga_pg": "FGA", "fta_pg": "FTA", "usage_proxy_pg": "usage_proxy"
    }.items():
        season_df[new_col] = safe_div(season_df[base_col], season_df["games"])

    for new_col, base_col in {
        "pts_per36": "PTS", "reb_per36": "REB", "ast_per36": "AST", "stl_per36": "STL",
        "blk_per36": "BLK", "tov_per36": "TOV", "fga_per36": "FGA", "fta_per36": "FTA", "usage_proxy_per36": "usage_proxy"
    }.items():
        season_df[new_col] = safe_div(season_df[base_col] * 36.0, season_df["total_minutes"])

    return season_df


def build_player_level_table(game_df: pd.DataFrame, season_df: pd.DataFrame) -> pd.DataFrame:
    player_df = (
        game_df.groupby(["Player_ID", "COHORT_PLAYER_NAME", "DRAFT_YEAR"], dropna=False)
               .agg(
                   seasons_played=("season_num", "nunique"),
                   total_games=("Game_ID", "nunique"),
                   total_minutes=("MIN", "sum"),
                   mean_min=("MIN", "mean"),
                   median_min=("MIN", "median"),
                   min_std=("MIN", "std"),
                   pct_10plus_min=("MIN", lambda s: np.mean(s >= 10)),
                   pct_20plus_min=("MIN", lambda s: np.mean(s >= 20)),
                   pct_25plus_min=("MIN", lambda s: np.mean(s >= 25)),
                   pct_30plus_min=("MIN", lambda s: np.mean(s >= 30)),
                   win_pct=("win_flag", "mean"),
                   home_share=("home_flag", "mean"),
                   avg_rest_days=("rest_days", "mean"),
                   b2b_rate=("b2b_flag", "mean"),
                   long_rest_rate=("long_rest_flag", "mean"),
                   plus_minus_mean=("PLUS_MINUS", "mean"),
                   plus_minus_std=("PLUS_MINUS", "std"),
                   points_game_std=("PTS", "std"),
                   minutes_game_std=("MIN", "std"),
                   FGM=("FGM", "sum"),
                   FGA=("FGA", "sum"),
                   FG3M=("FG3M", "sum"),
                   FG3A=("FG3A", "sum"),
                   FTM=("FTM", "sum"),
                   FTA=("FTA", "sum"),
                   OREB=("OREB", "sum"),
                   DREB=("DREB", "sum"),
                   REB=("REB", "sum"),
                   AST=("AST", "sum"),
                   STL=("STL", "sum"),
                   BLK=("BLK", "sum"),
                   TOV=("TOV", "sum"),
                   PF=("PF", "sum"),
                   PTS=("PTS", "sum"),
                   usage_proxy=("usage_proxy", "sum"),
               )
               .reset_index()
    )

    player_df["fg_pct_4yr"] = safe_div(player_df["FGM"], player_df["FGA"])
    player_df["fg3_pct_4yr"] = safe_div(player_df["FG3M"], player_df["FG3A"])
    player_df["ft_pct_4yr"] = safe_div(player_df["FTM"], player_df["FTA"])
    player_df["ts_pct_4yr"] = safe_div(player_df["PTS"], 2 * (player_df["FGA"] + 0.44 * player_df["FTA"]))
    player_df["fg3a_rate_4yr"] = safe_div(player_df["FG3A"], player_df["FGA"])
    player_df["ftr_4yr"] = safe_div(player_df["FTA"], player_df["FGA"])
    player_df["ast_tov_ratio_4yr"] = safe_div(player_df["AST"], player_df["TOV"])

    for new_col, base_col in {
        "pts_pg_4yr": "PTS", "reb_pg_4yr": "REB", "ast_pg_4yr": "AST", "stl_pg_4yr": "STL",
        "blk_pg_4yr": "BLK", "tov_pg_4yr": "TOV", "fga_pg_4yr": "FGA", "fta_pg_4yr": "FTA",
        "usage_proxy_pg_4yr": "usage_proxy"
    }.items():
        player_df[new_col] = safe_div(player_df[base_col], player_df["total_games"])

    for new_col, base_col in {
        "pts_per36_4yr": "PTS", "reb_per36_4yr": "REB", "ast_per36_4yr": "AST", "stl_per36_4yr": "STL",
        "blk_per36_4yr": "BLK", "tov_per36_4yr": "TOV", "fga_per36_4yr": "FGA", "fta_per36_4yr": "FTA",
        "usage_proxy_per36_4yr": "usage_proxy"
    }.items():
        player_df[new_col] = safe_div(player_df[base_col] * 36.0, player_df["total_minutes"])

    season_wide = season_df[["Player_ID", "season_num"] + SEASON_WIDE_METRICS].copy()
    season_wide["season_label"] = "s" + season_wide["season_num"].astype(int).astype(str)

    wide = season_wide.pivot(index="Player_ID", columns="season_label", values=SEASON_WIDE_METRICS)
    wide.columns = [f"{season}_{metric}" for metric, season in wide.columns]
    wide = wide.reset_index()

    trend_frames = []
    for metric in SLOPE_METRICS:
        metric_trend = (
            season_df.groupby("Player_ID", dropna=False)
                     .apply(lambda g: compute_trend(g, metric), include_groups=False)
                     .reset_index()
        )
        trend_frames.append(metric_trend)

    trend_df = trend_frames[0]
    for extra in trend_frames[1:]:
        trend_df = trend_df.merge(extra, on="Player_ID", how="outer")

    season_counts = (
        season_df.groupby("Player_ID", dropna=False)["games"]
                 .agg(
                     mean_games_per_season="mean",
                     min_games_in_any_season="min",
                     max_games_in_any_season="max",
                     std_games_across_seasons="std"
                 )
                 .reset_index()
    )

    recent_share = (
        season_df.assign(
            recent_games=lambda x: np.where(x["season_num"].isin([3, 4]), x["games"], 0),
            recent_minutes=lambda x: np.where(x["season_num"].isin([3, 4]), x["total_minutes"], 0),
            early_minutes=lambda x: np.where(x["season_num"].isin([1, 2]), x["total_minutes"], 0),
        )
        .groupby("Player_ID", dropna=False)
        .agg(
            recent_games=("recent_games", "sum"),
            recent_minutes=("recent_minutes", "sum"),
            early_minutes=("early_minutes", "sum"),
            total_games_check=("games", "sum"),
            total_minutes_check=("total_minutes", "sum"),
        )
        .reset_index()
    )
    recent_share["recent_games_share"] = safe_div(recent_share["recent_games"], recent_share["total_games_check"])
    recent_share["recent_minutes_share"] = safe_div(recent_share["recent_minutes"], recent_share["total_minutes_check"])
    recent_share["late_vs_early_minutes_ratio"] = safe_div(recent_share["recent_minutes"], recent_share["early_minutes"])
    recent_share = recent_share.drop(columns=["recent_games", "recent_minutes", "early_minutes", "total_games_check", "total_minutes_check"])

    player_feature_table = (
        player_df.merge(wide, on="Player_ID", how="left")
                 .merge(trend_df, on="Player_ID", how="left")
                 .merge(season_counts, on="Player_ID", how="left")
                 .merge(recent_share, on="Player_ID", how="left")
                 .sort_values(["DRAFT_YEAR", "Player_ID"])
                 .reset_index(drop=True)
    )

    return player_feature_table

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

def main():
    parser = argparse.ArgumentParser(description="Build a player-level feature table from Seasons 1-4 NBA game logs.")
    parser.add_argument("--raw-logs", type=Path, default=Path(DATA_DIR / "raw_game_logs_S1_to_S4.csv"))
    parser.add_argument("--cohort", type=Path, default=Path(DATA_DIR / "cohort_1999_2019.csv"))
    parser.add_argument("--out", type=Path, default=Path(DATA_DIR / "player_feature_table_1999_2019.csv"))
    parser.add_argument("--season-out", type=Path, default=Path(DATA_DIR / "player_season_feature_table_1999_2019.csv"))
    parser.add_argument("--min-total-games", type=int, default=0, help="Optional filter on total Seasons 1-4 games after features are built.")
    args = parser.parse_args()

    game_df = load_and_prepare(args.raw_logs, args.cohort)
    season_df = aggregate_player_season(game_df)
    player_feature_table = build_player_level_table(game_df, season_df)

    if args.min_total_games > 0:
        player_feature_table = player_feature_table[player_feature_table["total_games"] >= args.min_total_games].copy()

    player_feature_table.to_csv(args.out, index=False)
    season_df.to_csv(args.season_out, index=False)

    print(f"Saved player feature table to: {args.out.resolve()}")
    print(f"Saved player-season feature table to: {args.season_out.resolve()}")
    print(f"Rows in player feature table: {len(player_feature_table):,}")
    print(f"Rows in player-season feature table: {len(season_df):,}")
    print(f"Unique players in raw logs after cohort merge: {game_df['Player_ID'].nunique():,}")


if __name__ == "__main__":
    main()
