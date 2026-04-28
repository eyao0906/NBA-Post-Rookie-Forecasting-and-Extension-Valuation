import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
VISUAL_DIR = PROJECT_ROOT / "visual"

shotdata_df = pd.read_csv(DATA_DIR / "Shot Chart Details Raw" /"raw_shotchart_S1_to_S4_main.csv")
raw_games_df = pd.read_csv(DATA_DIR / "raw_game_logs_S1_to_S4.csv")
career_total_df = pd.read_csv(DATA_DIR / "career_totals_targets.csv")
player_split_df = pd.read_csv(DATA_DIR / "player_train_test_split_with_score.csv")

# Data Cleaning
shotdata_df['SHOT_MADE_NUMERIC'] = (shotdata_df['SHOT_MADE_FLAG'] == 'Made Shot').astype(float)
shotdata_df = shotdata_df.dropna(subset=['LOC_X', 'LOC_Y', 'SHOT_DISTANCE'])

# remove partial team rows
initial_len = len(career_total_df)
career_total_df['IS_TOT'] = career_total_df['TEAM_ABBREVIATION'] == 'TOT'
career_total_df = career_total_df.sort_values(
        by=['PLAYER_ID', 'SEASON_ID', 'IS_TOT'], 
        ascending=[True, True, False]
    )
career_total_df = career_total_df.drop_duplicates(subset=['PLAYER_ID', 'SEASON_ID'], keep='first')
career_total_df = career_total_df.drop(columns=['IS_TOT'])

MAX_SEASONS = 4
MAX_GAMES = 82
MAX_SHOTS = 23
shot_features = ['LOC_X', 'LOC_Y', 'SHOT_DISTANCE', 'SHOT_MADE_NUMERIC']
game_features = [
    'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
    'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 
    'STL', 'BLK', 'TOV', 'PF', 'PTS', 'PLUS_MINUS'
]
season_features = ['PLAYER_AGE', 'GP', 'MIN', 'PTS']
TARGET_COLS = [
    'Scoring_Metric_Score',
    'Efficiency_Metric_Score',
    'Playmaking_Metric_Score',
    'Defense_Metric_Score',
    'Control_Metric_Score'
]

def build_season_level_cache(career_df, game_df, shot_df):
    """
    Caches tensors at the (PLAYER_ID, SEASON_ID) level. 
    This prevents OOM errors since sliding windows share seasons.
    """
    season_cache = {}
    
    print("Grouping DataFrames for O(1) lookups (This takes a few seconds)...")
    # career_df needs to be indexed for fast single-row lookups
    career_indexed = career_df.set_index(['PLAYER_ID', 'SEASON_ID'])
    game_grouped = dict(tuple(game_df.groupby(['Player_ID', 'SEASON_ID'])))
    shot_grouped = dict(tuple(shot_df.groupby(['PLAYER_ID', 'GAME_ID'])))

    for (player_id, season_id), season_row in tqdm(career_indexed.iterrows(), total=len(career_indexed)):
        
        season_tensor = np.zeros(len(season_features))
        game_tensor = np.zeros((MAX_GAMES, len(game_features)))
        shot_tensor = np.zeros((MAX_GAMES, MAX_SHOTS, len(shot_features)))
        
        season_tensor[:] = season_row[season_features].values
        
        s_games = game_grouped.get((player_id, season_id))
        if s_games is not None:
            s_games = s_games.head(MAX_GAMES)
            game_tensor[:len(s_games), :] = s_games[game_features].values
            
            for g_idx, g_id in enumerate(s_games['Game_ID']):
                g_shots = shot_grouped.get((player_id, g_id))
                
                if g_shots is not None:
                    g_shots = g_shots.head(MAX_SHOTS)
                    shot_tensor[g_idx, :len(g_shots), :] = g_shots[shot_features].values
                    
        # Store as PyTorch tensors immediately to save conversion time later
        season_cache[(player_id, season_id)] = {
            'season': torch.FloatTensor(season_tensor),
            'games': torch.FloatTensor(game_tensor),
            'shots': torch.FloatTensor(shot_tensor)
        }
        
    return season_cache

# Execute the cache build
data_cache = build_season_level_cache(career_total_df, raw_games_df, shotdata_df)

def parse_input_seasons(input_season_str):
    """
    Converts "2016-2020" into a list of exact season strings: 
    ['2016-17', '2017-18', '2018-19', '2019-20']
    """
    start_yr, end_yr = map(int, input_season_str.split('-'))
    seasons = []
    for y in range(start_yr, end_yr):
        next_yr_suffix = str(y + 1)[-2:] # Gets '17' from 2017
        seasons.append(f"{y}-{next_yr_suffix}")
    return seasons
    

class NBASlidingWindowDataset(Dataset):
    def __init__(self, split_df, cache, target_cols):
        # Convert df to list of dictionaries for much faster row access during training
        self.samples = split_df.to_dict('records')
        self.cache = cache
        self.target_cols = target_cols

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        player_id = row['PLAYER_ID']
        
        # Get the exact 4 season strings for this window
        required_seasons = parse_input_seasons(row['INPUT_SEASON'])
        
        season_tensors = []
        game_tensors = []
        shot_tensors = []
        
        # Retrieve the pre-computed tensors from cache
        for season_id in required_seasons:
            cached_data = self.cache.get((player_id, season_id))
            
            if cached_data is not None:
                season_tensors.append(cached_data['season'])
                game_tensors.append(cached_data['games'])
                shot_tensors.append(cached_data['shots'])
            else:
                # Fallback to zero tensors if a player completely missed a season due to injury, etc.
                season_tensors.append(torch.zeros(len(season_features)))
                game_tensors.append(torch.zeros((MAX_GAMES, len(game_features))))
                shot_tensors.append(torch.zeros((MAX_GAMES, MAX_SHOTS, len(shot_features))))
                
        # Stack the 4 seasons together to create the final input tensors
        # shots shape: [4, MAX_GAMES, MAX_SHOTS, shot_features]
        final_shots = torch.stack(shot_tensors)
        # games shape: [4, MAX_GAMES, game_features]
        final_games = torch.stack(game_tensors)
        # seasons shape: [4, season_features]
        final_seasons = torch.stack(season_tensors)
        
        # Extract the 5 target scores as a 1D tensor directly from the dataframe row
        targets = [row[col] for col in self.target_cols]
        target_tensor = torch.FloatTensor(targets)
        
        return final_shots, final_games, final_seasons, target_tensor

train_test_split_df = pd.read_csv(DATA_DIR / "player_train_test_split_with_score.csv")
test_df = train_test_split_df[train_test_split_df['SPLIT'] == 'Test']
test_dataset = NBASlidingWindowDataset(test_df, data_cache, TARGET_COLS)

class ShotEncoder(nn.Module):
    """
    Phase 1: Compresses a sequence of shots into a single game-level 'shot profile'.
    """
    def __init__(self, shot_feature_dim, hidden_dim):
        super(ShotEncoder, self).__init__()
        self.gru = nn.GRU(
            input_size=shot_feature_dim, 
            hidden_size=hidden_dim, 
            batch_first=True
        )

    def forward(self, shots):
        # Input shape: [Batch * Seasons * Games, Max_Shots, Shot_Features]
        _, hidden = self.gru(shots)
        
        # We only want the final hidden state summarizing the shots
        # Output shape: [Batch * Seasons * Games, hidden_dim]
        return hidden.squeeze(0) 


class GameEncoder(nn.Module):
    """
    Phase 2: Fuses raw game stats with the shot profile, compressing 
    an 82-game season into a single 'season profile'.
    """
    def __init__(self, game_stat_dim, shot_hidden_dim, hidden_dim):
        super(GameEncoder, self).__init__()
        fused_dim = game_stat_dim + shot_hidden_dim
        
        self.gru = nn.GRU(
            input_size=fused_dim, 
            hidden_size=hidden_dim, 
            batch_first=True
        )

    def forward(self, games, shot_profiles):
        # games shape:         [Batch * Seasons, Max_Games, Game_Features]
        # shot_profiles shape: [Batch * Seasons, Max_Games, Shot_Hidden_Dim]
        
        # Concatenate raw game stats with the embedded shot data along the feature dimension
        fused_game_data = torch.cat((games, shot_profiles), dim=-1)
        
        _, hidden = self.gru(fused_game_data)
        
        # Output shape: [Batch * Seasons, hidden_dim]
        return hidden.squeeze(0)


class MultiLevelForecastingNetwork(nn.Module):
    """
    Phase 3 (The Master Module): Tracks the 4-season trajectory and outputs the 5 forecast scores.
    """
    def __init__(self, shot_dim, game_dim, season_dim, shot_hidden=32, game_hidden=64, rnn_hidden=64, output_dim=5):
        super(MultiLevelForecastingNetwork, self).__init__()
        
        self.shot_encoder = ShotEncoder(shot_dim, shot_hidden)
        self.game_encoder = GameEncoder(game_dim, shot_hidden, game_hidden)
        
        # RNN to track career trajectory over the 4 seasons
        self.career_gru = nn.GRU(
            input_size=season_dim + game_hidden, 
            hidden_size=rnn_hidden, 
            batch_first=True
        )
        
        # Distinct prediction head for each output target
        self.regressor = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(rnn_hidden, 32),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(32, output_dim)
        )

    def forward(self, shots, games, seasons):
        """
        Expected Inputs:
        shots:   [B, S, G, Sh, F_shot]
        games:   [B, S, G, F_game]
        seasons: [B, S, F_season]
        """
        B, S, G, Sh, F_shot = shots.shape
        _, _, _, F_game = games.shape
        _, _, F_season = seasons.shape

        # Shot Level 
        shots_flat = shots.view(B * S * G, Sh, F_shot)
        shot_profiles_flat = self.shot_encoder(shots_flat) 
        shot_profiles = shot_profiles_flat.view(B * S, G, -1)

        # Game Level (Parallelized across batches and seasons)
        games_flat = games.view(B * S, G, F_game)
        season_profiles_flat = self.game_encoder(games_flat, shot_profiles)
        season_profiles = season_profiles_flat.view(B, S, -1)

        # Season Level & Final Prediction
        fused_seasons = torch.cat((seasons, season_profiles), dim=-1)
        _, hidden = self.career_gru(fused_seasons)
        final_career_state = hidden.squeeze(0)
    
        predicted_scores = self.regressor(final_career_state)
        
        return predicted_scores

def draw_radar_chart(player_name, actual_scores, predicted_scores, target_names, path=None):
    # Number of variables
    num_vars = len(target_names)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The plot is circular, so we need to "complete the loop"
    actual_scores = np.concatenate((actual_scores, [actual_scores[0]]))
    predicted_scores = np.concatenate((predicted_scores, [predicted_scores[0]]))
    angles += angles[:1]
    
    # Initialize the spider plot
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
    
    # Draw one axe per variable and add labels
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(target_names, fontsize=10, fontweight='bold')
    
    # Set y-axis bounds (Assuming scores are 0-100)
    ax.set_ylim(0, 100)
    
    # Plot Actual
    ax.plot(angles, actual_scores, color='blue', linewidth=2, label='Actual Score')
    ax.fill(angles, actual_scores, color='blue', alpha=0.1)
    
    # Plot Predicted
    ax.plot(angles, predicted_scores, color='red', linewidth=2, linestyle='dashed', label='Predicted Score')
    ax.fill(angles, predicted_scores, color='red', alpha=0.1)
    
    plt.title(f"Performance Forecast: {player_name}", size=14, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    if path:
        plt.savefig(path, dpi=300)
        plt.close()
    else:
        plt.show()

radar_score_model = MultiLevelForecastingNetwork(
    shot_dim=len(shot_features),
    game_dim=len(game_features),
    season_dim=len(season_features),
    shot_hidden=32, 
    game_hidden=32, 
    rnn_hidden=32,
    output_dim=len(TARGET_COLS)
).to(device)

model_path = PROJECT_ROOT / "src" / "Player Performance Analysis" / 'best_radar_score_model.pth'
radar_score_model.load_state_dict(torch.load(model_path, map_location=device))
radar_score_model.eval()

HARDCODE_TEST_ID = [1629027, 202696]
radar_chart_count = 1

with torch.no_grad():
    for idx, row in enumerate(test_dataset.samples):
        if row['PLAYER_ID'] in HARDCODE_TEST_ID:
            # Fetch the specific pre-processed tensors for this player from the dataset
            shots, games, seasons, target = test_dataset[idx]

            shots = shots.unsqueeze(0).to(device)
            games = games.unsqueeze(0).to(device)
            seasons = seasons.unsqueeze(0).to(device)

            prediction = radar_score_model(shots, games, seasons)
            actual_vals = target.numpy()
            predicted_vals = prediction.squeeze(0).cpu().numpy()

            draw_radar_chart(
                player_name=row['PLAYER_NAME'], 
                actual_scores=actual_vals, 
                predicted_scores=predicted_vals, 
                target_names=TARGET_COLS,
                path=VISUAL_DIR / "Player Performance" / f"radar_score_example_{radar_chart_count}.png"
            )

            radar_chart_count += 1

print(f"Saved {radar_chart_count-1} Plots")