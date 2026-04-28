import os
import time
import pandas as pd
from nba_api.stats.endpoints import drafthistory, playergamelog, playercareerstats
from pathlib import Path

# Robust path handling (works in both scripts and Jupyter Notebooks)
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd()
    
DATA_DIR = PROJECT_ROOT / "data"

def setup_environment(folder_name=DATA_DIR):
    """Creates the data directory if it doesn't already exist."""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Directory '{folder_name}' created successfully.")
    else:
        print(f"Directory '{folder_name}' already exists. Ready to write.")

def build_hof_cohort(start_year=1990):
    """Fetches draft history and filters ONLY for post-1990 Hall of Fame players."""
    print("\nFetching draft history...")
    draft_data = drafthistory.DraftHistory().get_data_frames()[0]
    
    # Filter for 1990 onwards
    cohort_df = draft_data[draft_data['SEASON'].astype(int) >= start_year]
    
    # Hardcoded list of Hall of Famers drafted 1990 or later
    # (Note: Ben Wallace is HOF but went undrafted in '96, so he is excluded to maintain the draft-year format)
    hof_names = [
        "Gary Payton", "Toni Kukoc", "Dikembe Mutombo", 
        "Shaquille O'Neal", "Alonzo Mourning", "Chris Webber", 
        "Jason Kidd", "Grant Hill", "Kevin Garnett", 
        "Allen Iverson", "Kobe Bryant", "Steve Nash", "Ray Allen", 
        "Tim Duncan", "Tracy McGrady", "Chauncey Billups",
        "Dirk Nowitzki", "Paul Pierce", "Vince Carter", 
        "Manu Ginobili", "Tony Parker", "Pau Gasol", 
        "Yao Ming", "Dwyane Wade", "Chris Bosh", "Carmelo Anthony", "Dwight Howard", "Chris Paul",
        "Kevin Durant", "Russell Westbrook", "James Harden",
        "Kawhi Leonard", "Anthony Davis", "Stephen Curry", "LeBron James"
    ]
    
    # Filter the draft cohort to only include our HOF list
    cohort_df = cohort_df[cohort_df['PLAYER_NAME'].isin(hof_names)]
    cohort_list = cohort_df[['PERSON_ID', 'PLAYER_NAME', 'SEASON']].drop_duplicates()
    
    print(f"HOF Cohort built! Found {len(cohort_list)} Hall of Fame players.")
    
    # Save the cohort list using the new naming convention
    cohort_list.to_csv(DATA_DIR / "cohort_HOF.csv", index=False)
    return cohort_list

def fetch_complete_hof_pipeline(cohort_df):
    """
    Pulls Years 1-4 Game Logs and Career Totals for the HOF cohort.
    Includes resume logic to pick up where it left off.
    """
    # Updated file names
    logs_file = DATA_DIR / "HOF_raw_s1_s4_game_logs.csv"
    targets_file = DATA_DIR / "HOF_career_total_targets.csv"
    
    # =========================================================
    # RESUME LOGIC: Check who is already saved in the CSVs
    # =========================================================
    processed_players = set()
    
    if os.path.exists(targets_file):
        try:
            existing_targets = pd.read_csv(targets_file, usecols=['PLAYER_NAME'])
            processed_players.update(existing_targets['PLAYER_NAME'].dropna().unique())
        except Exception:
            pass 
            
    if os.path.exists(logs_file):
        try:
            existing_logs = pd.read_csv(logs_file, usecols=['PLAYER_NAME'])
            processed_players.update(existing_logs['PLAYER_NAME'].dropna().unique())
        except Exception:
            pass 

    if processed_players:
        print(f"\nFound existing HOF data! Resuming script...")
        print(f"Skipping {len(processed_players)} players who are already safe in your CSVs.")
    
    print("\nStarting HOF data extraction...")
    
    for index, row in cohort_df.iterrows():
        player_id = row['PERSON_ID']
        player_name = row['PLAYER_NAME']
        draft_year = int(row['SEASON'])
        
        # --- Skip the player if they are already in our list ---
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
                career_df.to_csv(targets_file, mode='a', header=not os.path.exists(targets_file), index=False)
        except Exception as e:
            print(f"  [!] Target error for {player_name}: {e}")
            
        time.sleep(1) 
        
        # ---------------------------------------------------------
        # PART 2: Fetch Game Logs for Seasons 1 through 4
        # ---------------------------------------------------------
        player_logs = []
        for year_offset in range(4): 
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
                pass 
                
            time.sleep(1.5) 
            
        if player_logs:
            combined_player_logs = pd.concat(player_logs, ignore_index=True)
            combined_player_logs.to_csv(logs_file, mode='a', header=not os.path.exists(logs_file), index=False)
            
        print(f"[{index + 1}/{len(cohort_df)}] Successfully processed: {player_name}")

    print("\n✅ COMPLETE! All HOF data has been safely stored in the 'data' folder.")

# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    setup_environment(DATA_DIR)
    
    # 1. Build the Hall of Fame cohort (Drafted 1990 or later)
    hof_cohort = build_hof_cohort(start_year=1990)
    
    # 2. Extract their data
    fetch_complete_hof_pipeline(hof_cohort)