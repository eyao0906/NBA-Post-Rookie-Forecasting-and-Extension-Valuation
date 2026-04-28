import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'
RANDOM_STATE = 42

PRESENTATION_FEATURES = [
    'mean_min', 'pts_per36_4yr', 'reb_per36_4yr', 'ast_per36_4yr', 'blk_per36_4yr',
    'ts_pct_4yr', 'fg3a_rate_4yr', 'ftr_4yr', 'usage_proxy_per36_4yr',
    'mpg_slope', 'pts_per36_slope', 'recent_minutes_share', 'late_vs_early_minutes_ratio'
]
ID_COLS = ['Player_ID', 'COHORT_PLAYER_NAME', 'DRAFT_YEAR']


def detect_pca_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith('pca_') and c.split('_')[-1].isdigit()]
    cols = sorted(cols, key=lambda x: int(x.split('_')[-1]))
    if not cols:
        raise ValueError('No PCA columns detected in PCA scores file.')
    return cols


def nearest_representatives(pca_scores: np.ndarray, labels: np.ndarray, names: pd.Series, n: int = 10):
    rows = []
    for cluster_id in sorted(np.unique(labels)):
        idx = np.where(labels == cluster_id)[0]
        cluster_points = pca_scores[idx]
        centroid = cluster_points.mean(axis=0, keepdims=True)
        d = np.sqrt(((cluster_points - centroid) ** 2).sum(axis=1))
        order = np.argsort(d)
        for rank, local_idx in enumerate(order[:n], start=1):
            i = idx[local_idx]
            rows.append({
                'cluster_id': int(cluster_id),
                'representative_rank': rank,
                'player_name': names.iloc[i],
                'distance_in_pca_space': float(d[local_idx]),
            })
    return pd.DataFrame(rows)


def assign_archetype_names(cluster_summary: pd.DataFrame) -> dict[int, str]:
    # Robust enough for this dataset; avoids hard-coding raw cluster IDs.
    remaining = set(cluster_summary['cluster_id'].tolist())
    mapping = {}

    # 1) Fringe / low-opportunity = lowest mean minutes.
    fringe = cluster_summary.sort_values('mean_min').iloc[0]['cluster_id']
    fringe = int(fringe)
    mapping[fringe] = 'Fringe / Low-Opportunity Players'
    remaining.remove(fringe)

    remain_df = cluster_summary[cluster_summary['cluster_id'].isin(remaining)].copy()

    # 2) Primary creators = highest playmaking with meaningful scoring.
    creator = remain_df.assign(score=remain_df['ast_per36_4yr'] + 0.15 * remain_df['pts_per36_4yr'])\
                      .sort_values('score', ascending=False).iloc[0]['cluster_id']
    creator = int(creator)
    mapping[creator] = 'High-Usage Primary Creators'
    remaining.remove(creator)

    remain_df = cluster_summary[cluster_summary['cluster_id'].isin(remaining)].copy()

    # 3) Interior bigs = strongest interior profile and lowest 3PA tendency.
    interior = remain_df.assign(
        score=remain_df['reb_per36_4yr'] + remain_df['blk_per36_4yr'] + 0.5 * remain_df['ftr_4yr'] - 2.0 * remain_df['fg3a_rate_4yr']
    ).sort_values('score', ascending=False).iloc[0]['cluster_id']
    interior = int(interior)
    mapping[interior] = 'Low-Usage Interior Bigs'
    remaining.remove(interior)

    remain_df = cluster_summary[cluster_summary['cluster_id'].isin(remaining)].copy()

    # 4) Scoring bigs / two-way forwards = stronger frontcourt scoring/size profile.
    scoring_big = remain_df.assign(
        score=remain_df['pts_per36_4yr'] + 0.6 * remain_df['reb_per36_4yr'] + 0.8 * remain_df['blk_per36_4yr'] - 0.4 * remain_df['ast_per36_4yr']
    ).sort_values('score', ascending=False).iloc[0]['cluster_id']
    scoring_big = int(scoring_big)
    mapping[scoring_big] = 'Scoring Bigs / Two-Way Forwards'
    remaining.remove(scoring_big)

    # 5) Leftover = perimeter wings / connectors.
    last_cluster = int(next(iter(remaining)))
    mapping[last_cluster] = 'Perimeter Wings & Connectors'

    return mapping


def main():
    parser = argparse.ArgumentParser(description='Run K-means on saved PCA scores and build archetype outputs.')
    parser.add_argument('--features-input', type=Path, default=DATA_DIR / 'player_feature_table_1999_2019.csv')
    parser.add_argument('--pca-input', type=Path, default=DATA_DIR / 'pca_player_features' / 'player_feature_table_pca_scores.csv')
    parser.add_argument('--pca-artifacts', type=Path, default=DATA_DIR / 'pca_player_features' / 'pca_artifacts.joblib')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT  / 'kmeans_k5_outputs_split')
    parser.add_argument('--n-clusters', type=int, default=5)
    parser.add_argument('--n-pca-export', type=int, default=5)
    args = parser.parse_args()

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    base_df = pd.read_csv(args.features_input)
    pca_df = pd.read_csv(args.pca_input)
    _ = joblib.load(args.pca_artifacts)  # validates existence / compatibility

    merge_cols = [c for c in ID_COLS if c in base_df.columns and c in pca_df.columns]
    if 'Player_ID' not in merge_cols:
        raise ValueError('Player_ID must exist in both the feature table and PCA score table.')

    pca_cols = detect_pca_cols(pca_df)
    X_pca = pca_df[pca_cols].to_numpy()

    kmeans = KMeans(n_clusters=args.n_clusters, random_state=RANDOM_STATE, n_init=50)
    labels = kmeans.fit_predict(X_pca)
    silhouette = float(silhouette_score(X_pca, labels)) if len(np.unique(labels)) > 1 else float('nan')

    own_dist = np.sqrt(((X_pca - kmeans.cluster_centers_[labels]) ** 2).sum(axis=1))
    all_dist = np.sqrt(((X_pca[:, None, :] - kmeans.cluster_centers_[None, :, :]) ** 2).sum(axis=2))

    cluster_labels = pca_df[[c for c in ID_COLS if c in pca_df.columns]].copy()
    cluster_labels['cluster_id'] = labels
    cluster_labels['own_cluster_distance'] = own_dist
    for j in range(args.n_clusters):
        cluster_labels[f'cluster_{j}_onehot'] = (labels == j).astype(int)
        cluster_labels[f'dist_to_cluster_{j}'] = all_dist[:, j]

    cluster_labels = pd.concat([cluster_labels, pca_df[pca_cols]], axis=1)

    clustered = base_df.merge(cluster_labels, on=merge_cols, how='left', validate='one_to_one')

    summary_cols = [c for c in PRESENTATION_FEATURES if c in clustered.columns]
    if not summary_cols:
        raise ValueError('No presentation features found for archetype profiling.')

    cluster_summary = (
        clustered.groupby('cluster_id', dropna=False)[summary_cols]
        .mean()
        .reset_index()
        .sort_values('cluster_id')
    )

    name_map = assign_archetype_names(cluster_summary)
    clustered['cluster_archetype_k5'] = clustered['cluster_id'].map(name_map)
    clustered['kmeans_cluster_k5'] = clustered['cluster_id'].astype(int)
    clustered['own_cluster_distance_k5'] = clustered['own_cluster_distance']
    clustered = clustered.drop(columns=['cluster_id', 'own_cluster_distance'])

    for j in range(args.n_clusters):
        clustered.rename(columns={
            f'cluster_{j}_onehot': f'cluster_{j}_onehot_k5',
            f'dist_to_cluster_{j}': f'dist_to_cluster_{j}_k5',
        }, inplace=True)
    rename_pca = {col: f'{col}_k5' for col in pca_cols[:args.n_pca_export]}
    clustered.rename(columns=rename_pca, inplace=True)

    cluster_sizes = (
        clustered.groupby(['kmeans_cluster_k5', 'cluster_archetype_k5'], dropna=False)
        .size()
        .reset_index(name='n_players')
        .sort_values('kmeans_cluster_k5')
    )

    cluster_summary_named = (
        clustered.groupby(['kmeans_cluster_k5', 'cluster_archetype_k5'], dropna=False)[summary_cols]
        .mean()
        .reset_index()
        .sort_values('kmeans_cluster_k5')
    )

    z = clustered[summary_cols].copy()
    z = (z - z.mean()) / z.std(ddof=0)
    z['kmeans_cluster_k5'] = clustered['kmeans_cluster_k5']
    z['cluster_archetype_k5'] = clustered['cluster_archetype_k5']
    zscore_summary = (
        z.groupby(['kmeans_cluster_k5', 'cluster_archetype_k5'], dropna=False)[summary_cols]
        .mean()
        .reset_index()
        .sort_values('kmeans_cluster_k5')
    )

    reps = nearest_representatives(X_pca, labels, clustered['COHORT_PLAYER_NAME'], n=10)
    reps['cluster_archetype_k5'] = reps['cluster_id'].map(name_map)

    predictor_cols = [
        'Player_ID', 'COHORT_PLAYER_NAME', 'DRAFT_YEAR', 'kmeans_cluster_k5', 'cluster_archetype_k5',
        'own_cluster_distance_k5',
    ]
    predictor_cols += [f'cluster_{j}_onehot_k5' for j in range(args.n_clusters)]
    predictor_cols += [f'dist_to_cluster_{j}_k5' for j in range(args.n_clusters)]
    predictor_cols += [f'{col}_k5' for col in pca_cols[:args.n_pca_export]]
    modeling_predictors = clustered[[c for c in predictor_cols if c in clustered.columns]].copy()

    clustered.to_csv(outdir / 'player_feature_table_1999_2019_clustered_k5.csv', index=False)
    modeling_predictors.to_csv(outdir / 'cluster_modeling_predictors_k5.csv', index=False)
    cluster_sizes.to_csv(outdir / 'cluster_sizes_k5.csv', index=False)
    cluster_summary_named.to_csv(outdir / 'cluster_summary_k5.csv', index=False)
    zscore_summary.to_csv(outdir / 'cluster_summary_zscores_k5.csv', index=False)
    reps.to_csv(outdir / 'cluster_representative_players_k5.csv', index=False)

    joblib.dump(
        {
            'kmeans': kmeans,
            'cluster_name_map': name_map,
            'pca_columns_used': pca_cols,
            'n_clusters': args.n_clusters,
        },
        outdir / 'kmeans_artifacts_k5.joblib',
    )

    metadata = {
        'features_input': str(args.features_input),
        'pca_input': str(args.pca_input),
        'n_players': int(len(clustered)),
        'n_pca_components_used': int(len(pca_cols)),
        'n_clusters': int(args.n_clusters),
        'silhouette_score': silhouette,
        'cluster_sizes': {str(int(k)): int(v) for k, v in zip(cluster_sizes['kmeans_cluster_k5'], cluster_sizes['n_players'])},
        'cluster_name_map': {str(int(k)): v for k, v in name_map.items()},
    }
    (outdir / 'kmeans_metadata_k5.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    # Visuals
    pc1_name, pc2_name = pca_cols[0], pca_cols[1]
    pc1_var = float(np.var(X_pca[:, 0]) / np.var(X_pca, axis=0).sum()) if X_pca.shape[1] > 1 else float('nan')
    pc2_var = float(np.var(X_pca[:, 1]) / np.var(X_pca, axis=0).sum()) if X_pca.shape[1] > 1 else float('nan')

    plt.figure(figsize=(11, 8))
    plt.scatter(pca_df[pc1_name], pca_df[pc2_name], c=labels, s=22, alpha=0.75)
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], s=220, marker='X', edgecolor='black')
    for cid, (x, y) in enumerate(centers[:, :2]):
        plt.text(x, y, f"{cid}: {name_map.get(cid, cid)}", fontsize=9, ha='left', va='bottom')
    plt.xlabel(f'{pc1_name} ({pc1_var * 100:.1f}% relative PCA variance)')
    plt.ylabel(f'{pc2_name} ({pc2_var * 100:.1f}% relative PCA variance)')
    plt.title('PCA Projection of Player Archetypes (K-means, k=5)')
    plt.tight_layout()
    plt.savefig(outdir / 'cluster_pca_scatter_k5.png', dpi=220, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(cluster_sizes['cluster_archetype_k5'], cluster_sizes['n_players'])
    plt.xticks(rotation=25, ha='right')
    plt.ylabel('Players')
    plt.title('Cluster Sizes (k=5)')
    plt.tight_layout()
    plt.savefig(outdir / 'cluster_size_bar_k5.png', dpi=220, bbox_inches='tight')
    plt.close()

    heat = zscore_summary.set_index('cluster_archetype_k5')[summary_cols]
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(heat.to_numpy(), aspect='auto')
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_xticks(np.arange(len(summary_cols)))
    ax.set_xticklabels(summary_cols, rotation=45, ha='right')
    ax.set_title('Archetype Profile Heatmap (Cluster Mean Z-scores)')
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f'{heat.iloc[i, j]:.1f}', ha='center', va='center', fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(outdir / 'cluster_profile_heatmap_k5.png', dpi=220, bbox_inches='tight')
    plt.close()

    if {'pts_per36_4yr', 'ast_per36_4yr'}.issubset(clustered.columns):
        plt.figure(figsize=(10, 7))
        plt.scatter(clustered['pts_per36_4yr'], clustered['ast_per36_4yr'], c=labels, s=24, alpha=0.7)
        for cid in sorted(clustered['kmeans_cluster_k5'].unique()):
            sub = clustered[clustered['kmeans_cluster_k5'] == cid]
            plt.text(sub['pts_per36_4yr'].median(), sub['ast_per36_4yr'].median(), name_map[cid], fontsize=9)
        plt.xlabel('PTS per 36 (Years 1-4)')
        plt.ylabel('AST per 36 (Years 1-4)')
        plt.title('Archetype Separation on Scoring vs Playmaking')
        plt.tight_layout()
        plt.savefig(outdir / 'cluster_pts_ast_scatter_k5.png', dpi=220, bbox_inches='tight')
        plt.close()

    lines = []
    lines.append('# K-means Archetype Clustering (k=5)')
    lines.append('')
    lines.append(f"- Players clustered: **{len(clustered):,}**")
    lines.append(f"- PCA components used: **{len(pca_cols)}**")
    lines.append(f"- Silhouette score in PCA space: **{silhouette:.3f}**")
    lines.append('')
    lines.append('## Cluster names')
    for cid in sorted(name_map):
        lines.append(f"- **{cid}** → {name_map[cid]}")
    lines.append('')
    lines.append('## Recommended downstream predictors')
    lines.append('- one-hot cluster indicators: `cluster_0_onehot_k5` ... `cluster_4_onehot_k5`')
    lines.append('- distances to all centroids: `dist_to_cluster_0_k5` ... `dist_to_cluster_4_k5`')
    lines.append('- optional PCA features: `pca_1_k5` ...')
    lines.append('- `own_cluster_distance_k5` as a prototype-fit / ambiguity measure')
    (outdir / 'kmeans_archetype_summary_k5.md').write_text('\n'.join(lines), encoding='utf-8')

    print(f'Saved K-means outputs to: {outdir.resolve()}')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
