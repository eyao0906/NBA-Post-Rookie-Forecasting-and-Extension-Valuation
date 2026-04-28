from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from archetype_workflow_utils import PATHS, append_log, ensure_dirs

# Conservative manual aliases for name variants that clearly refer to the same player.
# Keep this table small and explicit. Expand only after reviewing the unmatched diagnostic.
MANUAL_NAME_ALIASES = {
    "hedo turkoglu": "hidayet turkoglu",
    "wang zhi zhi": "wang zhizhi",
    "t j ford": "tj ford",
    "lou williams": "louis williams",
    "o j mayo": "oj mayo",
    "d j augustin": "dj augustin",
    "dennis schroder": "dennis schroeder",
    "t j warren": "tj warren",
    "juancho hernangomez": "juan hernangomez",
    "d j wilson": "dj wilson",
    "p j washington": "pj washington",
    "wes iwundu": "wesley iwundu",
    "svi mykhailiuk": "sviatoslav mykhailiuk",
    "nic claxton": "nicolas claxton",
}

BAD_CAP_MONTH_MAP = {
    "Jan": 2000,
    "Feb": 2001,
    "Mar": 2002,
    "Apr": 2003,
    "May": 2004,
    "Jun": 2005,
    "Jul": 2006,
    "Aug": 2007,
    "Sep": 2008,
    "Oct": 2009,
    "Nov": 2010,
    "Dec": 2011,
}


def normalize_name(name: str | float | int | None) -> str:
    if pd.isna(name):
        return ""
    text = str(name)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("'", "")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [t for t in text.split() if t not in {"jr", "sr", "ii", "iii", "iv", "v"}]
    key = " ".join(tokens)
    return MANUAL_NAME_ALIASES.get(key, key)


def repair_cap_season_string(value: str | float | int | None) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if re.fullmatch(r"\d{4}-\d{2}", text):
        return text
    m = re.fullmatch(r"([A-Za-z]{3})-(\d{2})", text)
    if m and m.group(1) in BAD_CAP_MONTH_MAP:
        start_year = BAD_CAP_MONTH_MAP[m.group(1)]
        end_suffix = str((start_year + 1) % 100).zfill(2)
        return f"{start_year}-{end_suffix}"
    return text


def season_start_year_from_string(season_string: str | None) -> float:
    if season_string is None or pd.isna(season_string):
        return np.nan
    m = re.match(r"^(\d{4})-\d{2}$", str(season_string))
    return float(m.group(1)) if m else np.nan


def load_cohort() -> pd.DataFrame:
    # Align the salary target table to the same drafted-player universe used by the existing feature/archetype pipeline.
    cohort = pd.read_csv(PATHS.data_dir / "player_feature_table_1999_2019.csv")
    cohort = cohort.rename(
        columns={
            "Player_ID": "PLAYER_ID",
            "COHORT_PLAYER_NAME": "PLAYER_NAME",
            "DRAFT_YEAR": "draft_year",
        }
    )
    cohort = cohort[["PLAYER_ID", "PLAYER_NAME", "draft_year"]].drop_duplicates().copy()
    cohort["PLAYER_ID"] = pd.to_numeric(cohort["PLAYER_ID"], errors="coerce")
    cohort["draft_year"] = pd.to_numeric(cohort["draft_year"], errors="coerce")
    cohort["name_key_raw"] = cohort["PLAYER_NAME"].map(normalize_name)
    cohort["name_key"] = cohort["name_key_raw"]
    cohort["year5_season_start"] = cohort["draft_year"] + 4
    cohort["year5_season_string"] = cohort["year5_season_start"].astype("Int64").astype(str).str.cat(
        ((cohort["year5_season_start"] + 1) % 100).astype("Int64").astype(str).str.zfill(2),
        sep="-",
    )
    return cohort


def build_salary_player_season_agg() -> pd.DataFrame:
    salary = pd.read_csv(PATHS.data_dir / "SalaryData/salaries_players_1990_2025_combined.csv")
    salary["season"] = pd.to_numeric(salary["season"], errors="coerce")
    salary["salary"] = pd.to_numeric(salary["salary"], errors="coerce").fillna(0.0)
    salary["player_name"] = salary["player_name"].astype(str)
    salary["name_key_raw"] = salary["player_name"].map(normalize_name)
    salary["name_key"] = salary["name_key_raw"]

    agg = (
        salary.groupby(["season", "name_key", "player_name"], dropna=False)
        .agg(
            year_salary_total=("salary", "sum"),
            raw_salary_rows=("salary", "size"),
            raw_team_count=("team_id", pd.Series.nunique),
            any_team_option=("team_option", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).gt(0).any())),
            any_player_option=("player_option", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).gt(0).any())),
            any_qualifying_offer=("qualifying_offer", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).gt(0).any())),
            any_two_way_contract=("two_way_contract", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).gt(0).any())),
            any_terminated=("terminated", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).gt(0).any())),
        )
        .reset_index()
        .rename(
            columns={
                "season": "season_start_year",
                "player_name": "salary_player_name_example",
            }
        )
    )

    # If multiple raw spellings collapse to the same normalized key in the same season, keep the highest-paid spelling as the example.
    agg = (
        agg.sort_values(["season_start_year", "name_key", "year_salary_total"], ascending=[True, True, False])
        .groupby(["season_start_year", "name_key"], as_index=False)
        .first()
    )
    return agg


def build_clean_salary_cap() -> pd.DataFrame:
    cap = pd.read_csv(PATHS.data_dir / "SalaryData/Cleaned_Salary_Cap.csv")
    cap["salary_cap"] = pd.to_numeric(cap["SALARY_CAP"], errors="coerce")
    cap["luxury_tax"] = (
        cap["LUXURY_TAX"]
        .astype(str)
        .str.replace(r"[^0-9.]", "", regex=True)
        .replace("", np.nan)
        .astype(float)
    )
    cap["season_string_fixed"] = cap["SEASON_STRING"].map(repair_cap_season_string)
    cap["season_start_year"] = cap["season_string_fixed"].map(season_start_year_from_string)
    cap = (
        cap[["season_start_year", "season_string_fixed", "salary_cap", "luxury_tax"]]
        .dropna(subset=["season_start_year"])
        .drop_duplicates(subset=["season_start_year"], keep="first")
        .sort_values("season_start_year")
        .reset_index(drop=True)
    )
    return cap


def build_year5_salary_target_table(
    cohort: pd.DataFrame,
    salary_agg: pd.DataFrame,
    cap: pd.DataFrame,
) -> pd.DataFrame:
    out = cohort.merge(
        salary_agg,
        left_on=["year5_season_start", "name_key"],
        right_on=["season_start_year", "name_key"],
        how="left",
    )
    out = out.merge(
        cap,
        left_on="year5_season_start",
        right_on="season_start_year",
        how="left",
        suffixes=("", "_cap"),
    )

    out["salary_name_match_flag"] = out["year_salary_total"].notna().astype(int)
    out["cap_match_flag"] = out["salary_cap"].notna().astype(int)
    out["year5_salary_match_flag"] = ((out["salary_name_match_flag"] == 1) & (out["cap_match_flag"] == 1)).astype(int)
    out["year5_salary_cap_pct"] = out["year_salary_total"] / out["salary_cap"]

    out["salary_match_type"] = np.select(
        [
            (out["salary_name_match_flag"] == 1) & (out["name_key"] != out["name_key_raw"]),
            (out["salary_name_match_flag"] == 1) & (out["name_key"] == out["name_key_raw"]),
        ],
        [
            "manual_alias",
            "exact_normalized_name",
        ],
        default="unmatched",
    )

    keep_cols = [
        "PLAYER_ID",
        "PLAYER_NAME",
        "draft_year",
        "year5_season_start",
        "year5_season_string",
        "salary_player_name_example",
        "year_salary_total",
        "salary_cap",
        "luxury_tax",
        "year5_salary_cap_pct",
        "raw_salary_rows",
        "raw_team_count",
        "any_team_option",
        "any_player_option",
        "any_qualifying_offer",
        "any_two_way_contract",
        "any_terminated",
        "salary_name_match_flag",
        "cap_match_flag",
        "year5_salary_match_flag",
        "salary_match_type",
        "name_key_raw",
        "name_key",
    ]
    return out[keep_cols].copy()


def build_summary(year5: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_players = int(len(year5))
    matched = int(year5["year5_salary_match_flag"].sum())
    unmatched = total_players - matched
    rows.append({"metric": "total_players", "value": total_players})
    rows.append({"metric": "players_with_year5_salary_cap_match", "value": matched})
    rows.append({"metric": "players_without_year5_salary_cap_match", "value": unmatched})
    rows.append({"metric": "year5_salary_match_rate", "value": matched / max(total_players, 1)})
    rows.append({"metric": "players_matched_via_manual_alias", "value": int((year5["salary_match_type"] == "manual_alias").sum())})
    rows.append({"metric": "players_with_multirow_salary_season", "value": int(year5["raw_salary_rows"].fillna(0).gt(1).sum())})
    rows.append({"metric": "players_with_multiteam_salary_season", "value": int(year5["raw_team_count"].fillna(0).gt(1).sum())})
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    cohort = load_cohort()
    salary_agg = build_salary_player_season_agg()
    cap = build_clean_salary_cap()
    year5 = build_year5_salary_target_table(cohort, salary_agg, cap)
    summary = build_summary(year5)

    unmatched = (
        year5[year5["year5_salary_match_flag"] == 0][
            ["PLAYER_ID", "PLAYER_NAME", "draft_year", "year5_season_start", "year5_season_string", "name_key_raw"]
        ]
        .sort_values(["draft_year", "PLAYER_NAME"])
        .reset_index(drop=True)
    )

    salary_agg_path = PATHS.archetype_output_dir / "SalaryBlock/salary_player_season_agg.csv"
    cap_path = PATHS.archetype_output_dir / "SalaryBlock/salary_cap_cleaned.csv"
    year5_path = PATHS.archetype_output_dir / "SalaryBlock/year5_salary_target_table.csv"
    unmatched_path = PATHS.archetype_output_dir / "SalaryBlock/year5_salary_unmatched_diagnostic.csv"
    summary_path = PATHS.archetype_output_dir / "SalaryBlock/year5_salary_merge_summary.csv"

    salary_agg.to_csv(salary_agg_path, index=False)
    cap.to_csv(cap_path, index=False)
    year5.to_csv(year5_path, index=False)
    unmatched.to_csv(unmatched_path, index=False)
    summary.to_csv(summary_path, index=False)

    append_log(
        phase="PHASE 12 — CLEAN SALARY DATA AND BUILD YEAR-5 TARGET",
        completed=(
            "Cleaned the historical salary-cap table, repaired malformed season strings, aggregated multi-team salary rows to player-season totals, "
            "matched the drafted-player cohort to Year-5 salary seasons using normalized player-name plus season matching, and built Year5_Salary_Cap_Pct."
        ),
        learned=(
            "The salary file is not one row per player-season because players can appear multiple times in the same year after team changes. "
            "The salary-cap table also contains broken season strings from 2000-01 through 2011-12, which must be repaired before merging."
        ),
        assumptions=(
            "Year 5 is defined as the season starting in draft_year + 4. "
            "Hoopshype IDs are not used because they do not reliably align with the NBA pipeline IDs; the merge uses normalized name plus season, with a small explicit alias table and an unmatched diagnostic for later review."
        ),
        files_read=[
            str(PATHS.data_dir / "player_feature_table_1999_2019.csv"),
            str(PATHS.data_dir / "salaries_players_1990_2025_combined.csv"),
            str(PATHS.data_dir / "Cleaned_Salary_Cap.csv"),
        ],
        files_written=[
            str(salary_agg_path),
            str(cap_path),
            str(year5_path),
            str(unmatched_path),
            str(summary_path),
        ],
    )


if __name__ == "__main__":
    main()
