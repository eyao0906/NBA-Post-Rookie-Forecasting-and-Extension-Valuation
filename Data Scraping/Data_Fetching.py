import os
import time
import pandas as pd
from nba_api.stats.endpoints import drafthistory, playergamelog, playercareerstats
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

def setup_environment(folder_name= DATA_DIR):
    """Creates the data directory if it doesn't already exist."""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Directory '{folder_name}' created successfully.")
    else:
        print(f"Directory '{folder_name}' already exists. Ready to write.")

def build_draft_cohort(start_year=1999, end_year=2019):
    """Fetches all drafted players within the specified year range."""
    print("\nFetching draft history...")
    draft_data = drafthistory.DraftHistory().get_data_frames()[0]
    
    cohort_df = draft_data[(draft_data['SEASON'].astype(int) >= start_year) & 
                           (draft_data['SEASON'].astype(int) <= end_year)]
    
    cohort_list = cohort_df[['PERSON_ID', 'PLAYER_NAME', 'SEASON']].drop_duplicates()
    print(f"Cohort built! Found {len(cohort_list)} drafted players.")
    
    # Save the cohort list for reference
    cohort_list.to_csv("data/draft_cohort_2005_2019.csv", index=False)
    return cohort_list

# def fetch_complete_pipeline(cohort_df):
    """
    Pulls Years 1-4 Game Logs and Career Totals (for Year 5 targets).
    Saves incrementally to avoid data loss during the long runtime.
    """
    logs_file = "data/raw_game_logs_S1_to_S4.csv"
    targets_file = "data/career_totals_targets.csv"
    
    print("\nStarting full data extraction. Grab a coffee, this will take a while! ☕")
    
    for index, row in cohort_df.iterrows():
        player_id = row['PERSON_ID']
        player_name = row['PLAYER_NAME']
        draft_year = int(row['SEASON'])
        
        # ---------------------------------------------------------
        # PART 1: Fetch Career Totals (To find Year-5 Target Data)
        # ---------------------------------------------------------
        try:
            career = playercareerstats.PlayerCareerStats(player_id=player_id)
            career_df = career.get_data_frames()[0]
            
            if not career_df.empty:
                career_df['PLAYER_NAME'] = player_name
                # Save incrementally
                career_df.to_csv(targets_file, mode='a', header=not os.path.exists(targets_file), index=False)
        except Exception as e:
            print(f"  [!] Target error for {player_name}: {e}")
            
        time.sleep(1) # Be polite to the API
        
        # ---------------------------------------------------------
        # PART 2: Fetch Game Logs for Seasons 1 through 4
        # ---------------------------------------------------------
        player_logs = []
        for year_offset in range(4): # 0, 1, 2, 3
            current_year = draft_year + year_offset
            season_string = f"{current_year}-{str(current_year + 1)[-2:]}"
            
            try:
                gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season_string)
                df = gamelog.get_data_frames()[0]
                
                if not df.empty:
                    df['PLAYER_NAME'] = player_name
                    df['SEASON_STRING'] = season_string
                    player_logs.append(df)
            except Exception as e:
                pass # Player might have been injured or out of the league this season
                
            time.sleep(1.5) # Crucial sleep to prevent IP ban
            
        # If we found game logs for this player, append them to our master CSV
        if player_logs:
            combined_player_logs = pd.concat(player_logs, ignore_index=True)
            combined_player_logs.to_csv(logs_file, mode='a', header=not os.path.exists(logs_file), index=False)
            
        # Progress tracker
        print(f"[{index + 1}/{len(cohort_df)}] Successfully processed: {player_name}")

    print("\n COMPLETE! All data has been safely stored in the 'data' folder.")

def fetch_complete_pipeline(cohort_df):
    """
    Pulls Years 1-4 Game Logs and Career Totals (for Year 5 targets).
    Saves incrementally to avoid data loss during the long runtime.
    """
    logs_file = DATA_DIR / "raw_game_logs_S1_to_S4.csv"
    targets_file = DATA_DIR / "career_totals_targets.csv"
    
    # =========================================================
    # RESUME LOGIC: Check who is already saved in the CSVs
    # =========================================================
    processed_players = set()
    
    if os.path.exists(targets_file):
        try:
            existing_targets = pd.read_csv(targets_file, usecols=['PLAYER_NAME'])
            processed_players.update(existing_targets['PLAYER_NAME'].dropna().unique())
        except Exception:
            pass # File might be empty
            
    if os.path.exists(logs_file):
        try:
            existing_logs = pd.read_csv(logs_file, usecols=['PLAYER_NAME'])
            processed_players.update(existing_logs['PLAYER_NAME'].dropna().unique())
        except Exception:
            pass # File might be empty

    if processed_players:
        print(f"\nFound existing data! Resuming script...")
        print(f"Skipping {len(processed_players)} players who are already safe in your CSVs.")
    
    print("\nStarting data extraction. Grab a coffee, this will take a while! ☕")
    
    for index, row in cohort_df.iterrows():
        player_id = row['PERSON_ID']
        player_name = row['PLAYER_NAME']
        draft_year = int(row['SEASON'])
        
        # --- NEW: Skip the player if they are already in our list ---
        if player_name in processed_players:
             print(f"[{index + 1}/{len(cohort_df)}] Skipping {player_name} (Already done)")
             continue
             
        # ---------------------------------------------------------
        # PART 1: Fetch Career Totals (To find Year-5 Target Data)
        # ---------------------------------------------------------
        try:
            career = playercareerstats.PlayerCareerStats(player_id=player_id)
            career_df = career.get_data_frames()[0]
            
            if not career_df.empty:
                career_df['PLAYER_NAME'] = player_name
                # Save incrementally
                career_df.to_csv(targets_file, mode='a', header=not os.path.exists(targets_file), index=False)
        except Exception as e:
            print(f"  [!] Target error for {player_name}: {e}")
            
        time.sleep(1) # Be polite to the API
        
        # ---------------------------------------------------------
        # PART 2: Fetch Game Logs for Seasons 1 through 4
        # ---------------------------------------------------------
        player_logs = []
        for year_offset in range(4): # 0, 1, 2, 3
            current_year = draft_year + year_offset
            season_string = f"{current_year}-{str(current_year + 1)[-2:]}"
            
            try:
                gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season_string)
                df = gamelog.get_data_frames()[0]
                
                if not df.empty:
                    df['PLAYER_NAME'] = player_name
                    df['SEASON_STRING'] = season_string
                    player_logs.append(df)
            except Exception as e:
                pass # Player might have been injured or out of the league this season
                
            time.sleep(1.5) # Crucial sleep to prevent IP ban
            
        # If we found game logs for this player, append them to our master CSV
        if player_logs:
            combined_player_logs = pd.concat(player_logs, ignore_index=True)
            combined_player_logs.to_csv(logs_file, mode='a', header=not os.path.exists(logs_file), index=False)
            
        # Progress tracker
        print(f"[{index + 1}/{len(cohort_df)}] Successfully processed: {player_name}")

    print("\n COMPLETE! All data has been safely stored in the 'data' folder.")

# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    setup_environment("data")
    my_cohort = build_draft_cohort(1999, 2019)
    fetch_complete_pipeline(my_cohort)