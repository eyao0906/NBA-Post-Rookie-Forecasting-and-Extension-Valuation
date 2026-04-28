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
test_dataset = NBASlidingWindowDataset(test_df, data_cache, ['Game_Score'])

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
    
game_score_model = MultiLevelForecastingNetwork(
    shot_dim=len(shot_features),
    game_dim=len(game_features),
    season_dim=len(season_features),
    shot_hidden=32, 
    game_hidden=32, 
    rnn_hidden=32,
    output_dim=1
).to(device)

model_path = PROJECT_ROOT / "src" / "Player Performance Analysis" / 'best_game_score_forecast_model.pth'
game_score_model.load_state_dict(torch.load(model_path, map_location=device))
game_score_model.eval()

HARDCODE_TEST_ID = [1629027, 202696]

# Function to specifically force Dropout layers to stay active during testing
def enable_mc_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

# Enable MC Dropout
enable_mc_dropout(game_score_model)

N_ITERATIONS = 50 # Run every player 50 times to build a distribution
mc_results = []
selected_players_data = []

print("Running Monte Carlo Dropout Inference...")

with torch.no_grad():
    for idx, row in enumerate(test_dataset.samples):
        shots, games, seasons, target = test_dataset[idx]
        
        shots = shots.unsqueeze(0).to(device)
        games = games.unsqueeze(0).to(device)
        seasons = seasons.unsqueeze(0).to(device)
        target_val = target.item()
        
        # Collect 50 different predictions for this single player
        preds = []
        for _ in range(N_ITERATIONS):
            pred = game_score_model(shots, games, seasons).item()
            preds.append(pred)
            
        # Calculate Mean (Prediction) and Standard Deviation (Uncertainty)
        pred_mean = np.mean(preds)
        pred_std = np.std(preds)
        
        result_dict = {
            'PLAYER_NAME': row['PLAYER_NAME'],
            'Actual': target_val,
            'Predicted_Mean': pred_mean,
            'Uncertainty_Std': pred_std,
            'Residual': target_val - pred_mean
        }

        mc_results.append(result_dict)

        if row['PLAYER_ID'] in HARDCODE_TEST_ID:
            selected_players_data.append(result_dict)

df_mc = pd.DataFrame(mc_results)
df_selected = pd.DataFrame(selected_players_data)

# Visualizations

# Residual Plot (Accuracy & Bias)
plt.figure(figsize=(8, 6))

sns.scatterplot(
    x='Predicted_Mean', 
    y='Residual', 
    data=df_mc, 
    alpha=0.6, 
    color='purple'
)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.title("Residual Plot: Model Accuracy & Bias", fontsize=14)
plt.xlabel("Predicted Game Score", fontsize=12)
plt.ylabel("Error (Actual - Predicted)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
# Save Plot A
residual_path = VISUAL_DIR / "Player Performance" / "game_score_forecast_residual_plot.png"
plt.savefig(residual_path, dpi=300)
plt.close()
print(f"Saved Residual Plot to {residual_path}")



# Actual vs Predicted with Uncertainty Bands
plt.figure(figsize=(8, 6))

# We use 1.96 * Std to represent a 95% Confidence Interval
plt.errorbar(
    x=df_mc['Actual'], 
    y=df_mc['Predicted_Mean'], 
    yerr=1.96 * df_mc['Uncertainty_Std'], 
    fmt='o', # points
    color='blue',
    ecolor='lightblue', # error bar color
    elinewidth=2, 
    capsize=3,
    alpha=0.7,
    label='Prediction ± 95% Confidence'
)

# Perfect prediction line
min_val = min(df_mc['Actual'].min(), df_mc['Predicted_Mean'].min())
max_val = max(df_mc['Actual'].max(), df_mc['Predicted_Mean'].max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')

plt.title("Forecasts with MC Dropout Uncertainty Bounds", fontsize=14)
plt.xlabel("Actual Game Score", fontsize=12)
plt.ylabel("Predicted Game Score", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
# Save Plot B
uncertainty_path = VISUAL_DIR / "Player Performance" / "game_score_forecast_mc_uncertainty_plot.png"
plt.savefig(uncertainty_path, dpi=300)
plt.close()
print(f"Saved Uncertainty Plot to {uncertainty_path}")

# Actual vs Predicted of select players
df_table = df_selected[['PLAYER_NAME', 'Predicted_Mean', 'Actual']].copy()
df_table.columns = ['PLAYER_NAME', 'Predicted Game Score', 'Actual Game Score']

def save_dataframe_as_png(df, filename):
    # Hide axes, use the DataFrame values and column labels
    fig, ax = plt.subplots(figsize=(10, 2)) # Adjust figsize for layout
    ax.axis('off')
    
    # Create the table with better formatting
    the_table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='left', # Align text to the left for standard tables
        edges='horizontal' # Use cleaner horizontal lines only
    )
    
    # Scale the table slightly for readability
    the_table.scale(1.1, 1.8)
    
    # Save the figure as a high-quality PNG
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

table_path = VISUAL_DIR / "Player Performance" / "example_player_score_table.png"
save_dataframe_as_png(df_table, table_path)
print(f"Saved Target Player Table to {table_path}")