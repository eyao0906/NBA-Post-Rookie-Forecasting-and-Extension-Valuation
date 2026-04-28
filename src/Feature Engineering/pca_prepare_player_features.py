import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RANDOM_STATE = 42

BASE_FEATURES = [
    'seasons_played','total_games','total_minutes','mean_min','median_min','min_std',
    'pct_10plus_min','pct_20plus_min','pct_25plus_min','pct_30plus_min',
    'avg_rest_days','b2b_rate','long_rest_rate','plus_minus_mean','plus_minus_std',
    'points_game_std','minutes_game_std',
    'fg_pct_4yr','fg3_pct_4yr','ft_pct_4yr','ts_pct_4yr','fg3a_rate_4yr','ftr_4yr','ast_tov_ratio_4yr',
    'pts_per36_4yr','reb_per36_4yr','ast_per36_4yr','stl_per36_4yr','blk_per36_4yr','tov_per36_4yr',
    'fga_per36_4yr','fta_per36_4yr','usage_proxy_per36_4yr',
    'games_slope','mpg_slope','pts_per36_slope','reb_per36_slope','ast_per36_slope','stl_per36_slope','blk_per36_slope',
    'tov_per36_slope','fga_per36_slope','usage_proxy_per36_slope','ts_pct_slope','fg3a_rate_slope','ftr_slope',
    'plus_minus_mean_slope','pct_25plus_min_slope','pct_30plus_min_slope',
    'mean_games_per_season','min_games_in_any_season','max_games_in_any_season','std_games_across_seasons',
    'recent_games_share','recent_minutes_share','late_vs_early_minutes_ratio',
]

SEASON_BLOCK_METRICS = ['mpg', 'pts_per36', 'reb_per36', 'ast_per36', 'ts_pct', 'usage_proxy_per36']
ID_COLS = ['Player_ID', 'COHORT_PLAYER_NAME', 'DRAFT_YEAR']


def choose_features(df: pd.DataFrame) -> list[str]:
    season_features = []
    for s in ['s1', 's2', 's3', 's4']:
        for metric in SEASON_BLOCK_METRICS:
            col = f'{s}_{metric}'
            if col in df.columns:
                season_features.append(col)
    features = [c for c in BASE_FEATURES + season_features if c in df.columns]
    if not features:
        raise ValueError('No PCA features were found in the input table.')
    return features


def winsorize_frame(df: pd.DataFrame, cols: list[str], lower_q: float = 0.01, upper_q: float = 0.99):
    clipped = df.copy()
    bounds = {}
    for c in cols:
        s = clipped[c]
        if pd.api.types.is_numeric_dtype(s) and s.notna().sum() > 10:
            lo, hi = s.quantile([lower_q, upper_q]).tolist()
            clipped[c] = s.clip(lo, hi)
            bounds[c] = {'lower': float(lo), 'upper': float(hi)}
        else:
            bounds[c] = {'lower': None, 'upper': None}
    return clipped, bounds


def main():
    parser = argparse.ArgumentParser(description='Standardize player features and fit PCA.')
    parser.add_argument('--input', type=Path, default=DATA_DIR / 'player_feature_table_1999_2019.csv')
    parser.add_argument('--output-dir', type=Path, default=DATA_DIR / 'pca_player_features')
    parser.add_argument('--variance-threshold', type=float, default=0.85)
    parser.add_argument('--winsor-lower', type=float, default=0.01)
    parser.add_argument('--winsor-upper', type=float, default=0.99)
    args = parser.parse_args()

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    features = choose_features(df)

    X_raw = df[features].copy()
    X_wins, winsor_bounds = winsorize_frame(
        X_raw, features, lower_q=args.winsor_lower, upper_q=args.winsor_upper
    )

    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_wins)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    pca_full = PCA(random_state=RANDOM_STATE)
    pca_full.fit(X_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cum_var, args.variance_threshold) + 1)

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    pca_cols = [f'pca_{i+1}' for i in range(n_components)]
    pca_scores = df[[c for c in ID_COLS if c in df.columns]].copy()
    pca_scores = pd.concat([pca_scores, pd.DataFrame(X_pca, columns=pca_cols, index=df.index)], axis=1)
    pca_scores.to_csv(outdir / 'player_feature_table_pca_scores.csv', index=False)

    var_df = pd.DataFrame({
        'component': pca_cols,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_explained_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
    })
    var_df.to_csv(outdir / 'pca_explained_variance.csv', index=False)

    metadata = {
        'input_file': str(args.input),
        'n_players': int(len(df)),
        'n_features_selected': int(len(features)),
        'selected_features': features,
        'n_components': int(n_components),
        'variance_threshold': float(args.variance_threshold),
        'cumulative_explained_variance_ratio': float(np.cumsum(pca.explained_variance_ratio_)[-1]),
        'winsor_lower': float(args.winsor_lower),
        'winsor_upper': float(args.winsor_upper),
    }
    (outdir / 'pca_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    joblib.dump(
        {
            'selected_features': features,
            'winsor_bounds': winsor_bounds,
            'imputer': imputer,
            'scaler': scaler,
            'pca': pca,
            'metadata': metadata,
        },
        outdir / 'pca_artifacts.joblib',
    )

    print(f'Saved PCA outputs to: {outdir.resolve()}')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
