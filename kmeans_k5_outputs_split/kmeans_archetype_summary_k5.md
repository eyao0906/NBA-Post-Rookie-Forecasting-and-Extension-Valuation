# K-means Archetype Clustering (k=5)

- Players clustered: **1,058**
- PCA components used: **24**
- Silhouette score in PCA space: **0.107**

## Cluster names
- **0** → High-Usage Primary Creators
- **1** → Low-Usage Interior Bigs
- **2** → Perimeter Wings & Connectors
- **3** → Fringe / Low-Opportunity Players
- **4** → Scoring Bigs / Two-Way Forwards

## Recommended downstream predictors
- one-hot cluster indicators: `cluster_0_onehot_k5` ... `cluster_4_onehot_k5`
- distances to all centroids: `dist_to_cluster_0_k5` ... `dist_to_cluster_4_k5`
- optional PCA features: `pca_1_k5` ...
- `own_cluster_distance_k5` as a prototype-fit / ambiguity measure