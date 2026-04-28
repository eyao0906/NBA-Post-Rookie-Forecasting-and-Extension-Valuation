import os
import time
import datetime
import pandas as pd
from requests.exceptions import ReadTimeout, ConnectionError, HTTPError
from nba_api.stats.endpoints import shotchartdetail
from pathlib import Path

# --- Configuration & Paths ---
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Input Files
COHORT_MAIN_FILE = DATA_DIR / "cohort_1999_2019.csv"
HOF_LOGS_FILE = DATA_DIR / "Shot Chart Details Raw/HOF_raw_s1_s4_game_logs.csv"

# Output Files
OUT_MAIN = DATA_DIR / "Shot Chart Details Raw/raw_shotchart_S1_to_S4_main.csv"
OUT_HOF = DATA_DIR / "Shot Chart Details Raw/raw_shotchart_S1_to_S4_hof_shotstyle.csv"
OUT_AUDIT = DATA_DIR / "Shot Chart Details Raw/shotchart_pull_audit.csv"
OUT_ERRORS = DATA_DIR / "Shot Chart Details Raw/shotchart_pull_errors.csv"

# NBA Shot Chart Era began in the 1996-97 Season
SHOT_CHART_ERA_START_YEAR = 1996

# Expected columns to keep from the API
EXPECTED_COLUMNS = [
    'GAME_ID', 'GAME_EVENT_ID', 'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_NAME',
    'PERIOD', 'MINUTES_REMAINING', 'SECONDS_REMAINING', 'EVENT_TYPE', 'ACTION_TYPE',
    'SHOT_TYPE', 'SHOT_ZONE_BASIC', 'SHOT_ZONE_AREA', 'SHOT_ZONE_RANGE', 'SHOT_DISTANCE',
    'LOC_X', 'LOC_Y', 'SHOT_ATTEMPTED_FLAG', 'SHOT_MADE_FLAG', 'GAME_DATE', 'HTM', 'VTM'
]

def format_season_string(start_year):
    """Converts 1999 into '1999-00' format."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"

def build_request_queue():
    """Builds the master list of Player-Seasons to query."""
    queue = []
    
    # 1. Main Cohort Logic (1999-2019)
    if COHORT_MAIN_FILE.exists():
        main_df = pd.read_csv(COHORT_MAIN_FILE)
        for _, row in main_df.iterrows():
            player_id = row['PERSON_ID']
            player_name = row['PLAYER_NAME']
            draft_year = int(row['SEASON'])
            
            for offset in range(4): # Seasons 1, 2, 3, 4
                season_num = offset + 1
                season_start_year = draft_year + offset
                queue.append({
                    'PLAYER_ID': player_id,
                    'PLAYER_NAME': player_name,
                    'draft_year': draft_year,
                    'season_num': season_num,
                    'season_string': format_season_string(season_start_year),
                    'source_pool': 'cohort_main',
                    'is_hof': False,
                    'shotstyle_eligible_hof': False
                })

    # 2. HOF Cohort Logic
    if HOF_LOGS_FILE.exists():
        hof_df = pd.read_csv(HOF_LOGS_FILE)
        # Infer rookie year from their earliest game log season
        hof_rookie_years = hof_df.groupby(['Player_ID', 'PLAYER_NAME'])['SEASON_STRING'].min().reset_index()
        
        for _, row in hof_rookie_years.iterrows():
            player_id = row['Player_ID']
            player_name = row['PLAYER_NAME']
            rookie_season_str = row['SEASON_STRING'] # e.g. "1996-97"
            rookie_year = int(rookie_season_str[:4])
            
            # Check eligibility: All 4 seasons must fall in the shot-chart era (>= 1996)
            is_eligible = (rookie_year >= SHOT_CHART_ERA_START_YEAR)
            source_pool = 'hof_shotstyle_subset' if is_eligible else 'hof_candidate'
            
            if is_eligible:
                for offset in range(4):
                    season_num = offset + 1
                    season_start_year = rookie_year + offset
                    queue.append({
                        'PLAYER_ID': player_id,
                        'PLAYER_NAME': player_name,
                        'draft_year': rookie_year,
                        'season_num': season_num,
                        'season_string': format_season_string(season_start_year),
                        'source_pool': source_pool,
                        'is_hof': True,
                        'shotstyle_eligible_hof': is_eligible
                    })

    return pd.DataFrame(queue)

def fetch_shotchart_with_retry(player_id, season_string, max_retries=3):
    """Executes the API call safely with retry logic and rate limit respect."""
    for attempt in range(max_retries):
        try:
            time.sleep(1.5) # Strict rate-limit pacing
            response = shotchartdetail.ShotChartDetail(
                player_id=player_id,
                team_id=0, # 0 gets all teams if player was traded mid-season
                season_nullable=season_string,
                season_type_all_star='Regular Season',
                context_measure_simple='FGA' # All Field Goal Attempts
            )
            df = response.get_data_frames()[0]
            return df, None
        except (ReadTimeout, ConnectionError, HTTPError) as e:
            time.sleep(3 * (attempt + 1)) # Backoff
            if attempt == max_retries - 1:
                return None, str(e)
        except Exception as e:
            return None, str(e) # Catch-all for unexpected API changes

def execute_pull():
    """Main execution engine with strict RESUME logic and incremental saving."""
    request_queue = build_request_queue()
    if request_queue.empty:
        print("Error: No cohort files found or queue is empty.")
        return

    # =========================================================
    # RESUME LOGIC: Check the audit file for completed pulls
    # =========================================================
    processed_keys = set()
    if OUT_AUDIT.exists():
        try:
            existing_audit = pd.read_csv(OUT_AUDIT)
            # Create a unique key of "PlayerID_Season" to track completion
            for _, row in existing_audit.iterrows():
                processed_keys.add(f"{row['PLAYER_ID']}_{row['season_string']}")
            
            print(f"\nFound existing audit log! Resuming script...")
            print(f"Skipping {len(processed_keys)} player-seasons already processed.\n")
        except Exception:
            pass # File might be empty or corrupted

    print(f"Starting execution for {len(request_queue)} Player-Seasons...")

    for index, req in request_queue.iterrows():
        # Check if we should skip
        req_key = f"{req['PLAYER_ID']}_{req['season_string']}"
        if req_key in processed_keys:
            print(f"[{index+1}/{len(request_queue)}] Skipping {req['PLAYER_NAME']} | {req['season_string']} (Already done)")
            continue
            
        pull_timestamp = datetime.datetime.now().isoformat()
        print(f"[{index+1}/{len(request_queue)}] Pulling {req['PLAYER_NAME']} | {req['season_string']}")
        
        raw_df, error_msg = fetch_shotchart_with_retry(req['PLAYER_ID'], req['season_string'])
        
        # --- Audit & Error Handling ---
        success_flag = error_msg is None
        is_empty = True if (success_flag and raw_df.empty) else False
        n_rows = len(raw_df) if (success_flag and not raw_df.empty) else 0
        pull_status = 'Success' if success_flag else 'Failed'
        if is_empty: pull_status = 'Empty'

        audit_log = {
            'PLAYER_ID': req['PLAYER_ID'],
            'PLAYER_NAME': req['PLAYER_NAME'],
            'season_string': req['season_string'],
            'season_num': req['season_num'],
            'source_pool': req['source_pool'],
            'requested': True,
            'success_flag': success_flag,
            'n_rows_returned': n_rows,
            'error_message': error_msg,
            'is_empty_result': is_empty,
            'is_hof': req['is_hof'],
            'shotstyle_eligible_hof': req['shotstyle_eligible_hof']
        }
        
        # INCREMENTAL SAVE: Write the audit log immediately
        pd.DataFrame([audit_log]).to_csv(OUT_AUDIT, mode='a', header=not OUT_AUDIT.exists(), index=False)

        if not success_flag:
            error_log = {
                'PLAYER_ID': req['PLAYER_ID'],
                'season_string': req['season_string'],
                'timestamp': pull_timestamp,
                'error': error_msg
            }
            pd.DataFrame([error_log]).to_csv(OUT_ERRORS, mode='a', header=not OUT_ERRORS.exists(), index=False)
            continue

        if not is_empty:
            # --- Data Formatting & Metadata Tagging ---
            valid_cols = [col for col in EXPECTED_COLUMNS if col in raw_df.columns]
            df_clean = raw_df[valid_cols].copy()
            
            df_clean['source_pool'] = req['source_pool']
            df_clean['draft_year'] = req['draft_year']
            df_clean['season_num'] = req['season_num']
            df_clean['season_string'] = req['season_string']
            df_clean['is_hof'] = req['is_hof']
            df_clean['shotstyle_eligible_hof'] = req['shotstyle_eligible_hof']
            df_clean['pull_status'] = pull_status
            df_clean['pull_timestamp'] = pull_timestamp

            # INCREMENTAL SAVE: Write raw data immediately based on the source pool
            if req['source_pool'] == 'cohort_main':
                df_clean.to_csv(OUT_MAIN, mode='a', header=not OUT_MAIN.exists(), index=False)
            elif req['source_pool'] == 'hof_shotstyle_subset':
                df_clean.to_csv(OUT_HOF, mode='a', header=not OUT_HOF.exists(), index=False)

    # --- Final Cleanup (Deduplication) ---
    print("\nExtraction finished. Running final deduplication pass just in case...")
    
    if OUT_MAIN.exists():
        df_main = pd.read_csv(OUT_MAIN)
        df_main.drop_duplicates(subset=['PLAYER_ID', 'GAME_ID', 'GAME_EVENT_ID'], inplace=True)
        df_main.to_csv(OUT_MAIN, index=False)
        
    if OUT_HOF.exists():
        df_hof = pd.read_csv(OUT_HOF)
        df_hof.drop_duplicates(subset=['PLAYER_ID', 'GAME_ID', 'GAME_EVENT_ID'], inplace=True)
        df_hof.to_csv(OUT_HOF, index=False)

    print("Execution Complete! Safe to proceed to the Autoencoder phase.")


if __name__ == "__main__":
    execute_pull()