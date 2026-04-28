"""Stage 1-3: Cohort build + ingestion + Year-5 target snapshot.

This module intentionally emphasizes reliability:
- retries + exponential backoff
- 1-second per-call sleep to avoid rate-limit bans
- clear checkpoints written to disk

Primary Python path: nba_api
R fallback path: src/r_ingest_fallback.R (NBAloveR / hoopR)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import drafthistory, playergamelog

from config import PipelineConfig


@dataclass
class PlayerCohortRecord:
    player_id: int
    player_name: str
    draft_year: int


def _safe_api_call(call_fn: Callable[[], pd.DataFrame], cfg: PipelineConfig) -> pd.DataFrame:
    """Execute API call with retry/backoff + mandatory sleep."""
    last_exc = None
    for attempt in range(1, cfg.api_max_retries + 1):
        try:
            out = call_fn()
            time.sleep(cfg.api_sleep_seconds)
            return out
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == cfg.api_max_retries:
                break
            time.sleep(cfg.api_backoff_seconds * attempt)
    raise RuntimeError(f"API call failed after retries: {last_exc}")


def build_draft_cohort(cfg: PipelineConfig, out_path: Path) -> pd.DataFrame:
    """Get players drafted in [2005, 2019] with Player_ID and Draft_Year."""
    records: List[PlayerCohortRecord] = []

    for year in range(cfg.draft_start_year, cfg.draft_end_year + 1):
        season = f"{year}"

        def _call() -> pd.DataFrame:
            endpoint = drafthistory.DraftHistory(season_year_nullable=season)
            return endpoint.get_data_frames()[0]

        df = _safe_api_call(_call, cfg)
        for _, row in df.iterrows():
            pid = row.get("PERSON_ID")
            name = row.get("PLAYER_NAME")
            if pd.notna(pid) and pd.notna(name):
                records.append(PlayerCohortRecord(int(pid), str(name), year))

    cohort = pd.DataFrame([r.__dict__ for r in records]).drop_duplicates("player_id")
    cohort = cohort.sort_values(["draft_year", "player_id"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(out_path, index=False)
    return cohort


def _season_string_for_year_index(draft_year: int, season_idx: int) -> str:
    """season_idx=1 means first NBA season after draft year.

    Example draft 2015:
      season_idx=1 -> '2015-16'
      season_idx=5 -> '2019-20'
    """
    y1 = draft_year + season_idx - 1
    y2 = str((y1 + 1) % 100).zfill(2)
    return f"{y1}-{y2}"


def fetch_player_first4_gamelogs(player_id: int, draft_year: int, cfg: PipelineConfig) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for season_num in range(1, 5):
        season = _season_string_for_year_index(draft_year, season_num)

        def _call() -> pd.DataFrame:
            endpoint = playergamelog.PlayerGameLog(player_id=player_id, season=season)
            return endpoint.get_data_frames()[0]

        try:
            g = _safe_api_call(_call, cfg)
        except RuntimeError:
            continue

        if g.empty:
            continue

        g["Season_Number"] = season_num
        g["Season_Label"] = season
        g["Player_ID"] = player_id

        # Parse and sort chronologically
        g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE"], errors="coerce")
        g = g.sort_values("GAME_DATE").reset_index(drop=True)
        g["Game_Number"] = np.arange(1, len(g) + 1)
        rows.append(g)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values("GAME_DATE").reset_index(drop=True)
    return out


def fetch_year5_targets_placeholder(player_name: str, draft_year: int) -> Dict[str, Optional[float]]:
    """Target snapshot placeholder for Year-5 totals.

    IMPORTANT:
    - Minutes can often be reconstructed from nba_api season logs.
    - Win Shares are not consistently available from nba_api endpoints.
    - Recommended production path: use R fallback script with NBAloveR/hoopR + BRef season tables.

    This function intentionally returns NaN placeholders to keep the pipeline structurally complete.
    """
    _ = (player_name, draft_year)
    return {"Y5_Minutes": np.nan, "Y5_WinShares": np.nan}


def build_raw_dataset(cohort: pd.DataFrame, cfg: PipelineConfig, out_path: Path) -> pd.DataFrame:
    all_rows: List[pd.DataFrame] = []

    for _, p in cohort.iterrows():
        logs = fetch_player_first4_gamelogs(int(p.player_id), int(p.draft_year), cfg)
        if logs.empty:
            continue

        if len(logs) < cfg.min_games_first4:
            continue

        target = fetch_year5_targets_placeholder(str(p.player_name), int(p.draft_year))
        logs["Player_Name"] = p.player_name
        logs["Draft_Year"] = p.draft_year
        logs["Y5_Minutes"] = target["Y5_Minutes"]
        logs["Y5_WinShares"] = target["Y5_WinShares"]
        all_rows.append(logs)

    if not all_rows:
        df = pd.DataFrame()
    else:
        df = pd.concat(all_rows, ignore_index=True)
        df = df.sort_values(["Player_ID", "GAME_DATE"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df
