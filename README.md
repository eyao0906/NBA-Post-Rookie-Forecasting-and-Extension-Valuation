# CaseStudy3 — NBA Post-Rookie Forecasting and Extension Valuation

This project provides decisional support on an NBA rookie player's extension contract valuation using first-contract (Years 1–4) game-by-game sequences.

## Targets
1. **Performance Forecast**: A forward-looking module for Years 5--7 production and broad outcome classification.

2. **Player Archetype \& Comparable Profile**: A player-intelligence module that identifies what kind of player this is, how stable that identity appears to be, and which historical players provide the best realistic precedent.

3. **Salary Decision Support**: A decision layer translating projected player value and risk into extension-oriented pay guidance.

## Structure
- `Data Scraping/`: Data scraping scripts
- `src/`: Python/R implementation scripts
    - `src\Feature Engineering`: Preliminary feature engineering and preparation
    - `src\Player Archetypes Analysis`: Player archetype workflow
    - `src\Salary Decision Support`: Salary Decision Workflow
- `data/`: raw/intermediate/processed data snapshots
    - `data\hoopshype_salaries`: Raw historical salary data per player
    - `data\SalaryData`: Raw historical salary cap data per year & Raw players' historical salary data
    - `data\pca_player_features`: PCA player feature table
    - `data\Shot Chart Details Raw`: Raw ShotChartDetails data, per shot per player per game
    - `data\Shot Feature`: Shot data feature tables
- `report/`: final write-up artifacts
- `visual/`: generated charts
    - `visual\Player Archetype`: Player archetype visuals
        - `visual\Player Archetype\player_dossier_demo`: Player dossier card outputs
- `Output/`: generated table results
    - `Output\Player Archetype Analysis`: Player archetype table results
        - `Output\Player Archetype Analysis\player_dossier_demo`: Player dossier card table results
    - `Output\Salary Decision Support`: Player salary decision support output tables

## Pipeline Stages
1. Build cohort (draft years 1999–2019)
2. Data scraping (game-by-game boxscores, ShotChartDetails, players' historical salary data, historical salary cap)
3. Feature Engineering
4. Deliverable 1 Performance Forecast
5. Deliverable 2 Player Archetype
6. Deliverable 3 Salary Decision Support

# Data Scraping
Run `Data Scraping/Data_Fetching.py` for game-by-game players' boxscores data.

Run `Data Scraping/Fetch_Shots.py` for Raw ShortChardDetails, per shot per player (all players, including Hall of Fame players) per game data

Run `Data Scraping/Data_Fetching.py` for game-by-game player boxscores data.

Run `Data Scraping/HOF_Fetching.py` for game-by-game Hall of Fame players' boxscores data

Run `Data Scraping/scrape_hoopshype_salaries_1990_2025.py` for historical salary data per player per year

Historical NBA salary cap data manually downloaded from
`https://www.basketball-reference.com/contracts/salary-cap-history.html`

Run`Data Scraping/scrape_hoopshype_salaries_1990_2025.R` for to obtain the specific performance of all NBA players each year between 1999 and 2025. This document is specifically used for the first part of player classification.

*(Important note: Please run this file locally. Due to numerous unknown errors in the HoopR installation package, the same equation may run at some times but not at others. The result is highly volatile due to API and live data. Furthermore, the HoopR installation package appears to have experienced significant API errors in its 3.0.0 update; therefore, please use version 2.1.0. You may face 2.1.0 is also not working, please try more earlier vision. Also, since this file is only intended for obtaining data one time, running it once locally is sufficient. Our analysis is based on the final run results from April 14.)*

# Feature Engineering
## Shot-style and legacy archetype-prep workflow

This section documents the **upstream feature-engineering scripts** that prepare shot-style artifacts and the original boxscore-based PCA / K-means artifacts used elsewhere in the project. These scripts live under:

* `STAT-946-Case-Study-3\src\Feature Enigineering`

They serve two purposes:

1. build reusable **shot-style summary and tensor artifacts** from Seasons 1–4 ShotChartDetail data, and
2. build the original **boxscore-based PCA and K-means archetype backbone** from `player_feature_table_1999_2019.csv`.   

### Important workflow note

The boxscore-only PCA and K-means outputs should be treated as **legacy upstream artifacts**, not the final Deliverable 2 archetype space once the fused role-style table exists. Shot summaries and shot embeddings are merged into the player-level feature table first, and then PCA and K-means are refit on that fused table.  

---

### 1. Build drafted-player shot-style feature tables

#### Script

* `src\Feature Enigineering\build_shot_feature_table.py`

#### Purpose

Builds cleaned and auditable shot-style feature tables from drafted-player Seasons 1–4 ShotChartDetail data. The script first cleans the raw shot rows, then aggregates them to interpretable feature tables at the player-season level and the player 4-year level. It also writes an audit table and a shot-count table so the shot pipeline can be verified before any tensor or embedding step is run. 

#### Inputs

* `data\raw_shotchart_S1_to_S4_main.csv`
* `data\cohort_1999_2019.csv`

#### Default output location

* `data\Shot Feature`

#### Outputs

* `data\Shot Feature\player_season_shot_features.csv`
* `data\Shot Feature\player_shot_features_4yr.csv`
* `data\Shot Feature\shotchart_audit_summary.csv`
* `data\Shot Feature\player_season_shot_counts.csv`

Optional:

* a smaller-unit player-game table can also be written if `--game-out` is supplied.  

#### What each output is used for

* `player_season_shot_features.csv` is the main **interpretable shot-style summary table** with one row per `Player_ID × season_num`. It is used for diagnostics, descriptive reporting, and later season-level interpretation.
* `player_shot_features_4yr.csv` is the **player-level 4-year aggregate shot-style table**. It is the reusable player-level summary that can later be merged into a fused archetype table.
* `shotchart_audit_summary.csv` documents cleaning, missingness, duplicate-event checks, and low-volume shot coverage so the shot data preparation step is auditable.
* `player_season_shot_counts.csv` records shot volume and basic coverage by player-season, which is important for deciding whether a player-season is reliable enough for downstream tensorization or embedding. 

---

### 2. Build Hall of Fame shot-style feature tables

#### Script

* `src\Feature Enigineering\build_hof_shot_feature_table.py`

#### Purpose

Builds the Hall of Fame auxiliary version of the shot-style feature tables using the same Seasons 1–4 shot-style logic as the main drafted-player cohort. This keeps the HOF ceiling-comp library on the same feature definition as the main player pool. 

#### Inputs

* `data\raw_shotchart_S1_to_S4_hof_shotstyle.csv`
* `data\cohort_HOF.csv`

#### Default output location

* `data\Shot Feature`

#### Outputs

* `data\Shot Feature\hof_player_season_shot_features.csv`
* `data\Shot Feature\hof_player_shot_features_4yr.csv`
* `data\Shot Feature\hof_shotchart_audit_summary.csv`
* `data\Shot Feature\hof_player_season_shot_counts.csv` 

#### What each output is used for

* `hof_player_season_shot_features.csv` is the HOF season-level shot-style summary table.
* `hof_player_shot_features_4yr.csv` is the HOF player-level 4-year shot-style summary table.
* `hof_shotchart_audit_summary.csv` is the HOF-specific cleaning and coverage audit.
* `hof_player_season_shot_counts.csv` records HOF shot volume by player-season. 

---

### 3. Build player-season spatial shot tensors

#### Script

* `src\Feature Enigineering\build_shotchart_tensors.py`

#### Purpose

Converts cleaned player-season shot events into image-like half-court tensors for the deep shot-style representation stage. The script bins shots into a fixed spatial grid and produces a 3-channel tensor per player-season. 

#### Inputs

* `data\raw_shotchart_S1_to_S4_main.csv`

#### Default output location

* `data\Shot Feature\shot_tensors`

#### Core tensor design

Each player-season tensor contains 3 channels:

1. attempt density
2. made-shot density
3. smoothed local make-rate surface

The default representation is:

* half-court only
* `25 × 25` spatial grid
* Gaussian smoothing
* minimum shot threshold for export eligibility.  

#### Outputs

* `data\Shot Feature\shot_tensors\player_season_shot_tensors.npz`
* `data\Shot Feature\shot_tensors\player_shot_tensor_index.csv`
* `data\Shot Feature\shot_tensors\player_season_shot_counts.csv`
* `data\Shot Feature\shot_tensors\shot_tensor_metadata.json` 

#### What each output is used for

* `player_season_shot_tensors.npz` stores the actual tensor array that will be fed into the CNN autoencoder.
* `player_shot_tensor_index.csv` maps each tensor back to `PLAYER_ID`, `season_num`, shot counts, and related metadata so the tensor stage remains auditable.
* `player_season_shot_counts.csv` preserves the full coverage table, including player-seasons that may be filtered out from tensor export because of low shot volume.
* `shot_tensor_metadata.json` records grid settings, coordinate ranges, channel meanings, smoothing parameters, and the minimum-shot threshold, so the tensorization step can be reproduced exactly. 

This is the bridge from interpretable shot tables to the computationally intensive shot-style representation model. It transforms raw shot events into the fixed spatial objects needed for CNN-based embedding learning. 

---

### 4. Prepare PCA scores from the original player feature table

#### Script

* `src\Feature Enigineering\pca_prepare_player_features.py`

#### Purpose

Standardizes the original player-level boxscore feature table and fits PCA on that table. This is the original, boxscore-only dimension-reduction step used before K-means clustering. The script automatically selects eligible numeric features, winsorizes them, median-imputes missing values, standardizes them, and then keeps enough principal components to hit the chosen explained-variance threshold. 

#### Inputs

* `data\player_feature_table_1999_2019.csv`

#### Default output location

* `data\pca_player_features`

#### Outputs

* `data\pca_player_features\player_feature_table_pca_scores.csv`
* `data\pca_player_features\pca_explained_variance.csv`
* `data\pca_player_features\pca_metadata.json`
* `data\pca_player_features\pca_artifacts.joblib` 

#### What each output is used for

* `player_feature_table_pca_scores.csv` is the PCA score table used as the direct input to the original K-means clustering script.
* `pca_explained_variance.csv` records component-level explained variance and cumulative explained variance for verification.
* `pca_metadata.json` stores the chosen variance threshold, selected features, number of retained components, and winsorization settings.
* `pca_artifacts.joblib` stores the fitted preprocessing and PCA objects needed to transform data consistently later. 

---

### 5. Run K-means on saved PCA scores

#### Script

* `src\Feature Enigineering\kmeans_from_pca_player_features.py`

#### Purpose

Runs K-means on the saved PCA scores and builds the original boxscore-based archetype outputs. It assigns cluster labels, computes own-cluster and cross-cluster distances, exports modeling predictors, summarizes cluster profiles, and assigns basketball-facing cluster names.  

#### Inputs

* `data\player_feature_table_1999_2019.csv`
* `data\pca_player_features\player_feature_table_pca_scores.csv`
* `data\pca_player_features\pca_artifacts.joblib`

#### Default output location

* `kmeans_k5_outputs_split`

#### Outputs

* `kmeans_k5_outputs_split\player_feature_table_1999_2019_clustered_k5.csv`
* `kmeans_k5_outputs_split\cluster_modeling_predictors_k5.csv`
* `kmeans_k5_outputs_split\cluster_sizes_k5.csv`
* `kmeans_k5_outputs_split\cluster_summary_k5.csv`
* `kmeans_k5_outputs_split\cluster_summary_zscores_k5.csv`
* `kmeans_k5_outputs_split\cluster_representative_players_k5.csv`
* `kmeans_k5_outputs_split\kmeans_artifacts_k5.joblib`
* `kmeans_k5_outputs_split\kmeans_metadata_k5.json` 

#### What each output is used for

* `player_feature_table_1999_2019_clustered_k5.csv` is the fully clustered player feature table with cluster labels and distance fields attached.
* `cluster_modeling_predictors_k5.csv` is the compact downstream modeling table containing cluster label, one-hot indicators, centroid distances, own-cluster distance, and exported PCA coordinates.
* `cluster_sizes_k5.csv` records player counts by cluster.
* `cluster_summary_k5.csv` records cluster mean summaries in basketball-facing feature space.
* `cluster_summary_zscores_k5.csv` records standardized cluster profiles for interpretation.
* `cluster_representative_players_k5.csv` records the nearest representative players to each cluster center.
* `kmeans_artifacts_k5.joblib` stores the fitted K-means model and cluster name map.
* `kmeans_metadata_k5.json` stores clustering metadata such as silhouette score, cluster sizes, and archetype labels. 

These are the original macro-archetype artifacts that later workflows can consume.

---

## Recommended execution order for this upstream feature-engineering block

Run these scripts in the order below:

```text
build_player_feature_table.py
build_shot_feature_table.py
build_hof_shot_feature_table.py
build_shotchart_tensors.py
pca_prepare_player_features.py
kmeans_from_pca_player_features.py
```

* the shot-style summary tables should be built before tensorization so coverage can be audited first,
* PCA depends on the player-level feature table,
* and K-means depends on the saved PCA score table.   

## Interpretation summary

This upstream block produces:

* interpretable shot-style summaries,
* reproducible shot tensor artifacts,
* the original boxscore-only PCA layer,
* and the original boxscore-only K-means archetype layer.

# Deliverable 1: Player Performance

This section documents the reproducible workflow of **Deliverable 1 (Player Performance Prediction Module)** from the NBA Rookie Contract Extension Project. This module aims to provide a forward-looking assessment of a player's performance in their fifth through seventh years of the NBA using data from their first four NBA seasons. Specifically, this section is responsible for transforming early-career data into predictive frameworks that can be used for decision-making, including future performance expectations, outcome classifications, and confidence-based risk information.

The overall workflow in this section is organized around four linked tasks: 

1. **Early-career player information** is cleaned and summarized into player-level and sequence-level inputs using Seasons 1–4 only. 
2. **An expectation versus realization framework** is used to construct interpretable outcome labels, where players are grouped into Bust, Neutral, and Sleeper categories through **PCA and K-means clustering**. 
3. Supervised models, including an **interpretable multinomial logistic regression and a sequence-based LSTM model** are trained to classify the player based on the historical 1-4 years value, and , and 
4. Final outputs are converted into **player-level clustering summaries** that can be handed forward into the archetype and salary-decision modules.

## 1. Deliverable 1 objective

Deliverable 1 answers the business question:

**Given what is known from the player’s first four NBA seasons, what is the most likely Years 5–7 performance trajectory, how should that trajectory be classified, and how much uncertainty should be attached to that forecast?**

The final Deliverable 1 package is expected to produce:

1. a Years 5–7 performance forecast,
2. a player-level outcome classification,
3. a Bust / Neutral / Sleeper labeling framework,
4. class probabilities for each player,
5. a confidence-based risk signal,
6. held-out test-set evaluation for forecast credibility,
7. player-level comparison outputs for baseline versus sequence models,
8. stakeholder-facing summary tables and visuals for decision support.

---

## 2. Workflow Structure

### 2.1 Player Performance Forecasting

Run `/src/Player Performance Analysis/Player_Train_Test_Split.ipynb` to create train/test splits
Run `/src/Player Performance Analysis/Player_Performance_Analysis.ipynb` to calculate the Game Score
Run `/src/Player Performance Analysis/Player_Performance_Forecast.ipynb` to train the Game Score model
Run `/src/Player Performance Analysis/Player_Performance_Forecast_Radar.ipynb` to train the radar Score model
Run `/src/Player Performance Analysis/player_performance_graphs.py` and `/src/Player Performance Analysis/player_radar_chart.py` to create the results for the report

---

### 2.2 Player Performance Clustering

Please looking at the `/src/ Player_Performance_Clustering / Player_Performance_Clustering.ipynb`

This file contains the full workflow for building the Bust / Neutral / Sleeper player outcome labels and then testing whether those labels can be predicted from a player’s first four NBA seasons. The file is not just one clustering step. It covers data cleaning, player-level feature construction, unsupervised labeling, supervised modeling, model comparison, and final output generation.

This document contains the following detailed process:

1. Load the raw player-season dataset

The main raw input is the historical NBA player-season statistics file: `/data/all_player_stats_1999-2025.csv`. This file contains one row per player-season. It is the base dataset used throughout the file. 

And, `/data/player_train_test_split_with_score.csv`. This split file provides a fixed train/test membership and previously prepared player-level score information.

2. Clean the season labels and filter out low-information player-seasons

Reading data from `/data/all_player_stats_1999-2025.csv`. First, we need fixe inconsistent season strings. After that, it extracts year information and sorts seasons chronologically so that each player’s career can be traced in proper order.

And then, we remove seasons that are not useful for stable player evaluation. The main filters are: GP > 20, MIN > 10, which keeps only player-seasons with enough games played and enough minutes to make the statistics meaningful.

3. Construct a season-level performance score

Then, it computes a player performance score based on a Hollinger-style game score formula. This score is used as a general summary of player impact and becomes one of the most important variables in the later analysis.

𝐺𝑚𝑆𝑐 =𝑃𝑇𝑆+0.4∗𝐹𝐺𝑀 −0.7∗𝐹𝐺𝐴−0.4∗(𝐹𝑇𝐴−𝐹𝑇𝑀)+0.7∗ 𝑂𝑅𝐸𝐵+0.3∗𝐷𝑅𝐸𝐵+𝑆𝑇𝐿+0.7∗𝐴𝑆𝑇 +0.7∗𝐵𝐿𝐾−0.4∗𝑃𝐹 −𝑇𝑂V

4. Count seasons per player and define player career windows

After cleaning the season-level table, the file groups were arranged by player and their career years were created. The rule was: we only kept records of players with at least 7 years of performance scores.

- The first 4 seasons were considered input history; 

- The seasons 5 to 7 were considered later-career realized outcomes.

This was done for subsequent dimensionality reduction and classification. It's important to note that because we used a consistent test set, for players who didn't meet the criteria but were still in the test set, we used predicted scores as a substitute. This was to ensure consistency. After research and comparison, we found that only 19 players ultimately needed additional predicted values. Therefore, this had no significant impact on our prediction classification.

The output will be: player_analysis_df_match_filled

5. Build a player-level analysis table

For each player, the file creates summaries from the first four seasons, including:

- four-year averages
- year 4 values
- improvement from year 1 to year 4
- slope across years 1 to 4

It also creates later-career summaries using years 5 to 7, including:

- average performance score from years 5 to 7
- average points, minutes, rebounds, assists, and related outcomes

And, then it will Merge the player-level table with the train/test split file, impute missing values in the player-level table and repair mismatched or missing players manually.

6. Create the clustering input variables and standardize the clustering features

Once the player-level table is finalized, then we need create the main clustering variables:

expected_score: average performance score from years 1 to 4
realized_score: average performance score from years 5 to 7
delta_score: realized score minus expected score

Before dimension reduction and clustering, the file standardizes the clustering variables so they are on comparable scales. This prevents variables with larger raw ranges from dominating the clustering results.

7. Run PCA and t-SNE + K-means clustering

Then, we need to apply PCA to the standardized clustering variables and keeps the first three principal components. It then runs K-means clustering on those PCA coordinates.

- the explained variance ratio of the PCs
- total explained variance
- cluster counts
- cluster summaries
- silhouette score

The PCA-based clusters are then interpreted using the mean delta_score:

- lowest delta cluster → Bust
- middle delta cluster → Neutral
- highest delta cluster → Sleeper

We also applies t-SNE to the same standardized clustering variables and then runs K-means on the t-SNE coordinates.

8. Compare PCA clustering and t-SNE clustering and visualize the PCA and t-SNE clusters in 3D

9. Build the multinomial logistic regression baseline model

Then, we uses multinomial logistic regression to predict the PCA label using only first-four-season features.

The predictors come from early-career variables. Later-career variables are excluded because they would leak future information. The model is trained on the fixed training set and evaluated on the fixed test set.

10. Train the LSTM sequence model

Next, we build an LSTM classifier using sequence data instead of aggregated summaries.

For each eligible player:

- only the first 4 cleaned seasons are used
- each season contributes a feature vector
- the final input shape is one 4-season sequence per player

Sequence features include many season-level basketball statistics, such as:

- games played, games started, minutes, shooting attempts and percentages. rebounds, assists, steals, blocks, turnovers. points and performance score

This lets the model learn development patterns over time, rather than only averages or slopes.

11. Compare logistic regression and LSTM directly

Compare between following evualtion to find best model:

- a model accuracy comparison table
- logistic regression confusion matrix
- LSTM confusion matrix
- player-level merged comparison table
- direct comparison summary
- cross-model confusion matrix

12. Create confidence measures from model probabilities

then, we need introduce a confidence interpretation using the maximum predicted class probability. This turns the LSTM output into a more decision-support style result: high, medium confidence or low confidence.

13. Score all eligible players with the trained LSTM and save the final all-player prediction file

Finally, we apply the trained LSTM to all eligible players who have four usable seasons. The file saves the final player-level output: `/data/final_lstm_predictions_all_players.csv`

14. Saving the Result

Please note that this file contains many transitional outputs, such as PCA and T-SNE classification images, LSTM and Logistic regression results, etc. These files are all stored in `/src/Player_Performance_Clustering/Result`. 

If you want to view these results directly, please run `/src/Player_Performance_Clustering/Clustering_Result.py`. This file will automatically save all results in `/visual/Player_Clustering/`.

---

### 2.3 Skills radar Plot

Please looking at the `/src/ Player_Performance_Clustering / Clustering_Visualize_Conclusion.ipynb`

The purpose of this file is to take the final clustering and prediction results and answer a more practical question:

**For specific players, how does their early-career skill profile compare to the average profile of the class they were assigned to?**

1. Read the final all-player prediction results

First, we need load `/data/final_lstm_predictions_all_players.csv`

This gives the final classification output from the LSTM model. We use it to inspect players, check example names, and identify which players will be used for the final visualization.

And we also need to load `/data/all_player_stats_1999-2025.csv`

This is necessary because we want to compare players based on their actual first-four-season feature profile, not just their final class label.

2. Select specific example players

We will focuse on selected players for case-study style interpretation. The main example players are:

- Trae Young
- Nikola Vucevic

3. Build skill summary categories for radar charts

We transform player statistics into a smaller set of interpretable skill categories for visualization. These radar-chart categories include:

- Scoring
- Shooting
- Playmaking
- Rebounding
- Defense
- Ball Security

For details, please read the report. We explained the meaning and results of the equations in detail in the formal report.

4. Compare each selected player to the average of a chosen class

5. Save final radar plot data

A later part of the workflow saves the final radar chart data into the result folder so a separate Python script or QMD report can redraw the plot without rerunning the file logic.

The saved file is: `/src/ Player_Performance_Clustering/Result/selected_players_radar_plot_data.csv`. This file contains the final radar-ready long-format data.

If you want to view these results directly, please run `/src/Player_Performance_Clustering/Clustering_Result.py`. This file will automatically save all results in `/visual/Player_Clustering/`.

---

## 3. Deliverable 1 outputs that matter most for the paper

If the report only needs the final Deliverable 1 artifacts, the most important outputs are:

### PCA vs T-SNE + K means (3D) Plot
- `visual\Player_Clustering\pca_3d_kmeans.png`
- `visual\Player_Clustering\tsne_3d_kmeans.png`

### Model Output
- `visual\Player_Clustering\lstm_confusion_matrix.png`
- `visual\Player_Clustering\model_accuracy_comparison.png`
- `visual\Player_Clustering\model_direct_comparison_summary.png`
- `visual\Player_Clustering\multinomial_logit_confusion_matrix.png`

### Skill radar for control group comparison
- `visual\Player_Clustering\selected_players_radar.png`

---

# Deliverable 2: Player Archetype & Comparable Profile Workflow

This section documents the **active reproducible workflow** for **Deliverable 2** in `STAT-946-Case-Study-3`, together with the **salary-ready bridge scripts** that translate Deliverable 2 outputs into **Block 2: Archetype and Comp-Market Context** for Deliverable 3.

The Deliverable 2 implementation is a dedicated pipeline under:

- `STAT-946-Case-Study-3\src\Player Archetypes Analysis`

The project framing for Deliverable 2 is a **hybrid archetype system**:

- **macro role** from saved PCA + K-means outputs,
- **shot-style subtype** from a spatial ShotChartDetail embedding pipeline,
- **identity drift** across Seasons 1–4,
- **realistic comps** from the drafted-player pool,
- **ceiling comps** from the Hall of Fame auxiliary library,
- **stakeholder-facing dossier outputs** for front-office decision support,
- and now a **salary-ready comp-market bridge** that prepares Deliverable 2 outputs for Deliverable 3.

## 1. Deliverable 2 objective

Deliverable 2 answers the business question:

> **Who is this player now, how did that identity develop across the rookie-contract window, who has looked like this before, and what is the realistic upside?**

The final Deliverable 2 package is expected to produce:

1. current archetype,
2. identity drift / role evolution,
3. realistic comparable players,
4. optional Hall of Fame ceiling comps,
5. archetype-match / prototype-fit information,
6. median later-career outcome of the comp group,
7. shot-style explanation,
8. visuals and dossier-style outputs for decision support.

In addition, the Deliverable 2 workflow now feeds Deliverable 3 through a salary-ready bridge:

9. cleaned Year-5 salary-cap targets,
10. similarity-weighted comp-market salary anchors,
11. a stakeholder-facing **Block 2: Archetype and Comp-Market Context** table for salary decision support.

---

## 2. Workflow Structure

### 2.1 Active code path

The current Deliverable 2 and salary-bridge run chain is the dedicated analysis sequence below:

1. `00_project_inventory.py`
2. `01_cohort_coverage_audit.py`
3. `02_build_macro_archetype_table.py`
4. `03_build_shotstyle_tensors.py`
5. `04_train_shot_autoencoder.py`
6. `05_cluster_shot_embeddings.py`
7. `06_build_identity_drift.py`
8. `07_build_comps.py`
9. `08_assemble_player_archetype_profiles.py`
10. `09_make_player_archetype_visuals.py`
11. `11_build_practical_player_dossiers_fixed.py`
12. `12_build_year5_salary_targets.py`
13. `13_build_block2_comp_market_context.py`
14. `10_write_workflow_log.py`

These scripts together implement the current Deliverable 2 architecture and the explicit bridge from Deliverable 2 into Deliverable 3.

### 2.2 Utility module actively used by the whole chain

- `archetype_workflow_utils.py`

This module defines the canonical project paths and shared utilities, including:

- `DATA_DIR = project_root / "data"`
- `KMEANS_DIR = project_root / "kmeans_k5_outputs_split"`
- `SHOT_OUTPUT_DIR = project_root / "Output" / "ShotChartDetail"`
- `ARCHETYPE_OUTPUT_DIR = project_root / "Output" / "Player Archetype Analysis"`
- `ARCHETYPE_VISUAL_DIR = project_root / "visual" / "Player Archetype"`

Deliverable 2 workflow writes its reproducible tables to **`Output\Player Archetype Analysis`** and its stakeholder-facing figures to **`visual\Player Archetype`**.

### 2.3 Older Feature Engineering scripts are not the active run order here

The project still contains older or upstream scripts in `src\Feature Engineering`, such as the original PCA / K-means or ShotChartDetail builders. However, the current Deliverable 2 workflow **does not document them as the active execution order**.

Instead:

- **saved PCA + K-means outputs are reused** as upstream artifacts,
- **raw shot-chart files are reused** as inputs,
- the current Deliverable 2 pipeline **rebuilds shot tensors and retrains the shot autoencoder inside `03_*.py` and `04_*.py` under `src\Player Archetypes Analysis`**,
- and the new salary bridge is built explicitly inside `12_*.py` and `13_*.py`.

So for reproducibility of Deliverable 2 and its salary-ready extension, the scripts to run are the `00_*.py` to `13_*.py` analysis scripts above, not the older Feature Engineering scripts.

### 2.4 Deliverable 3 bridge principle

Scripts `12_build_year5_salary_targets.py` and `13_build_block2_comp_market_context.py` do **not** replace Deliverable 2. They extend it.

Their job is to translate:

- realistic comps,
- archetype labels,
- subtype labels,
- drift,
- and comp quality

into a **salary-ready comp-market layer** that can be consumed by Deliverable 3.

This keeps the project logic clean:

- Deliverable 1 explains **what may happen next**,
- Deliverable 2 explains **who the player is**,
- Deliverable 3 decides **what the player is worth and how aggressively to extend**.

---

## 3. Input data and reusable upstream artifacts

### 3.1 Core drafted-player inputs

- `data\cohort_1999_2019.csv`
- `data\player_feature_table_1999_2019.csv`
- `data\player_season_feature_table_1999_2019.csv`
- `data\career_totals_targets.csv`
- `data\Shot Chart Details Raw\raw_shotchart_S1_to_S4_main.csv`
- `data\salaries_players_1990_2025_combined.csv`
- `data\Cleaned_Salary_Cap.csv`

### 3.2 Hall of Fame auxiliary inputs

- `data\cohort_HOF.csv`
- `data\HOF_career_total_targets.csv`
- `data\Shot Chart Details Raw\raw_shotchart_S1_to_S4_hof_shotstyle.csv`

### 3.3 Reused saved macro-archetype artifacts

These are **consumed**, not refit, by the active Deliverable 2 workflow:

- `kmeans_k5_outputs_split\player_feature_table_1999_2019_clustered_k5.csv`
- `kmeans_k5_outputs_split\cluster_modeling_predictors_k5.csv`
- `kmeans_k5_outputs_split\cluster_summary_k5.csv`
- `kmeans_k5_outputs_split\cluster_summary_zscores_k5.csv`
- `kmeans_k5_outputs_split\cluster_representative_players_k5.csv`

### 3.4 Important workflow principle

The macro backbone is reused from saved artifacts. The shot-style layer is rebuilt inside the current Deliverable 2 pipeline. The salary bridge is then built on top of those finalized Deliverable 2 outputs.

That split is central to reproducibility:

- macro role = reused saved artifacts,
- shot-style embedding = reproducibly rebuilt inside active Deliverable 2 scripts,
- salary-ready Block 2 context = reproducibly built after Deliverable 2 is finalized.

---

## 4. Step-wise workflow structure

### Phase 0 — Project map and path audit
#### Script
- `00_project_inventory.py`

#### Purpose
Confirms the project structure, checks whether required drafted-player, HOF, feature, clustering, shot-chart, and reusable code artifacts exist, and initializes the workflow log.

#### Inputs
- project root
- all expected data / output / code paths listed in the script inventory

#### Outputs
- `Output\Player Archetype Analysis\project_file_inventory.csv`
- `Output\Player Archetype Analysis\player_archetype_analysis_workflow_log.txt`

---

### Phase 1 — Cohort and coverage audit
#### Script
- `01_cohort_coverage_audit.py`

#### Purpose
Audits drafted-player and Hall of Fame coverage across cohort membership, logs, features, clustering support, shot-chart support, and later-career targets.

#### Inputs
- drafted-player cohort and main project tables
- HOF cohort and auxiliary tables

#### Outputs
- `Output\Player Archetype Analysis\drafted_player_coverage_audit.csv`
- `Output\Player Archetype Analysis\hof_player_coverage_audit.csv`
- `Output\Player Archetype Analysis\cohort_join_audit.csv`

---

### Phase 2 — Lock the macro archetype backbone
#### Script
- `02_build_macro_archetype_table.py`

#### Purpose
Builds the **macro role** layer by reusing the saved PCA + K-means artifacts rather than refitting clustering.

#### Inputs
- `player_feature_table_1999_2019_clustered_k5.csv`
- `cluster_modeling_predictors_k5.csv`
- `cluster_summary_k5.csv`
- `cluster_summary_zscores_k5.csv`
- `cluster_representative_players_k5.csv`

#### Outputs
- `Output\Player Archetype Analysis\archetype_macro_player_table.csv`
- `Output\Player Archetype Analysis\archetype_macro_summary_table.csv`

#### Logic
This script canonicalizes:

- `PLAYER_ID`
- `PLAYER_NAME`
- `draft_year`
- `macro_cluster_id`
- `macro_archetype`
- `own_cluster_distance`
- cluster one-hot / distance features
- prototype ambiguity measures

---

### Phase 3 — Build shot-style spatial tensors
#### Script
- `03_build_shotstyle_tensors.py`

#### Purpose
Converts raw ShotChartDetail rows into stable **player-season spatial tensors** on a common half-court grid.

#### Inputs
- `data\Shot Chart Details Raw\raw_shotchart_S1_to_S4_main.csv`
- `data\Shot Chart Details Raw\raw_shotchart_S1_to_S4_hof_shotstyle.csv`
- `data\cohort_1999_2019.csv`
- `data\cohort_HOF.csv`

#### Core tensor design
The script builds a **3-channel tensor** per player-season:

1. shot attempt density,
2. made-shot density,
3. smoothed make-rate map.

It uses:

- fixed half-court coordinates,
- a `25 x 25` grid,
- Gaussian smoothing,
- and a minimum-shot filter for embedding eligibility.

#### Outputs
- `Output\Player Archetype Analysis\draft_player_season_shot_tensors.npz`
- `Output\Player Archetype Analysis\hof_player_season_shot_tensors.npz`
- `Output\Player Archetype Analysis\draft_player_tensors.npz`
- `Output\Player Archetype Analysis\hof_player_tensors.npz`
- `Output\Player Archetype Analysis\shot_tensor_player_season.parquet`
- `Output\Player Archetype Analysis\shot_tensor_player.parquet`
- `Output\Player Archetype Analysis\shot_coverage_audit.csv`
- `Output\Player Archetype Analysis\hof_shot_coverage_audit.csv`
- `Output\Player Archetype Analysis\shot_tensor_metadata.json`
- player tensor index CSV files for drafted and HOF pools

---

### Phase 4 — Train the shot-style embedding model
#### Script
- `04_train_shot_autoencoder.py`

#### Purpose
Trains a CNN autoencoder on drafted-player player-season tensors and encodes both drafted-player and HOF tensors into a common latent shot-style space.

#### Inputs
- drafted-player and HOF player-season tensors
- drafted-player and HOF tensor index tables

#### Outputs
- `Output\Player Archetype Analysis\shot_embedding_player_season.csv`
- `Output\Player Archetype Analysis\shot_embedding_player.csv`
- `Output\Player Archetype Analysis\hof_shot_embedding_player_season.csv`
- `Output\Player Archetype Analysis\hof_shot_embedding_player.csv`
- `Output\Player Archetype Analysis\shot_autoencoder_training_history.csv`
- `Output\Player Archetype Analysis\shot_autoencoder_model.pt`
- `Output\Player Archetype Analysis\shot_autoencoder_artifacts.json`
- `Output\Player Archetype Analysis\shot_embedding_nearest_neighbors.csv`
- `visual\Player Archetype\shot_embedding_reconstruction_examples.png`
- `visual\Player Archetype\shot_autoencoder_training_curve.png`

#### Important implementation detail
Player-level embeddings are not naive averages. They are aggregated using:

- shot-volume weighting,
- and mild recency weighting toward later rookie-contract seasons.

---

### Phase 5 — Create shot-style subtypes
#### Script
- `05_cluster_shot_embeddings.py`

#### Purpose
Creates the second archetype layer by clustering **player-level shot embeddings** with a **Gaussian Mixture Model**.

#### Inputs
- `shot_embedding_player.csv`
- `shot_embedding_player_season.csv`
- `archetype_macro_player_table.csv`

#### Outputs
- `Output\Player Archetype Analysis\shot_style_cluster_summary.csv`
- `Output\Player Archetype Analysis\shot_style_representative_players.csv`
- `Output\Player Archetype Analysis\shot_style_player_table.csv`
- `Output\Player Archetype Analysis\shot_style_player_season_table.csv`
- `Output\Player Archetype Analysis\player_hybrid_archetype_table.csv`
- `visual\Player Archetype\shot_style_embedding_scatter.png`

#### Logic
This phase assigns:

- player-level subtype ID,
- subtype label,
- subtype membership probability,
- entropy / ambiguity,
- season-level subtype assignments,
- hybrid label = `macro_archetype | shot_style_subtype`.

---

### Phase 6 — Build identity drift / role evolution
#### Script
- `06_build_identity_drift.py`

#### Purpose
Measures how player identity changes from Year 1 to Year 4 in both:

- boxscore role space,
- shot-style embedding space.

#### Inputs
- `data\player_season_feature_table_1999_2019.csv`
- `Output\Player Archetype Analysis\shot_embedding_player_season.csv`
- `Output\Player Archetype Analysis\shot_style_player_season_table.csv`
- `Output\Player Archetype Analysis\player_hybrid_archetype_table.csv`

#### Outputs
- `Output\Player Archetype Analysis\player_role_drift_summary.csv`
- `Output\Player Archetype Analysis\player_shotstyle_drift_summary.csv`
- `Output\Player Archetype Analysis\player_identity_drift_table.csv`

#### Logic
The script computes:

- Year 1 to Year 4 displacement,
- total path length,
- subtype changes,
- final drift class such as `stable`, `evolving_gradually`, or `role_shifting_materially`.

#### Important limitation
Because saved season-level PCA scores were not available as a reusable artifact, boxscore drift is approximated with standardized season-level features instead of original season-level PCA coordinates.

---

### Phase 7 — Build comparable-player infrastructure
#### Script
- `07_build_comps.py`

#### Purpose
Builds the comp engine for Deliverable 2.

#### Inputs
- `Output\Player Archetype Analysis\player_identity_drift_table.csv`
- `Output\Player Archetype Analysis\hof_shot_embedding_player.csv`
- `data\career_totals_targets.csv`
- HOF auxiliary target / cohort files

#### Outputs
- `Output\Player Archetype Analysis\realistic_comps.csv`
- `Output\Player Archetype Analysis\ceiling_comps_hof.csv`
- `Output\Player Archetype Analysis\player_comp_dossier_table.csv`

#### Logic
##### Realistic comps
Built from the drafted-player pool using a weighted similarity score over:

- macro-role compatibility,
- shot-style embedding distance,
- PCA distance when available,
- prototype-fit distance,
- drift similarity.

##### Ceiling comps
Built from the HOF auxiliary pool and treated only as upside analogs, not the main comp universe.

---

### Phase 8 — Assemble final Deliverable 2 tables
#### Script
- `08_assemble_player_archetype_profiles.py`

#### Purpose
Assembles the main player-level Deliverable 2 table by merging archetype, drift, comp, and explanation layers.

#### Inputs
- `player_identity_drift_table.csv`
- `player_comp_dossier_table.csv`
- `shot_style_cluster_summary.csv`

#### Outputs
- `Output\Player Archetype Analysis\final_player_archetype_profile_table.csv`
- `Output\Player Archetype Analysis\final_player_archetype_case_study_table.csv`

#### Final table contents
The final profile table is designed to include the core stakeholder-facing columns:

- macro archetype,
- shot-style subtype,
- hybrid label,
- prototype-fit ambiguity,
- own-cluster distance,
- drift class,
- realistic comps,
- optional ceiling comp,
- comp-group outcome summaries,
- supporting shot-style explanation,
- comp-based interpretation.

---

### Phase 9 — Produce Deliverable 2 visuals
#### Script
- `09_make_player_archetype_visuals.py`

#### Purpose
Generates stakeholder-facing visuals for the archetype layer, subtype layer, drift, comp support, and sample dossier view.

#### Inputs
- `final_player_archetype_profile_table.csv`
- subtype summary / subtype table files
- drift table
- realistic comps
- raw drafted-player shot rows
- drafted-player tensor files

#### Outputs
Written to:
- `visual\Player Archetype`

Key outputs include:

- `macro_archetype_summary_chart.png`
- `shot_style_subtype_summary_chart.png`
- `embedding_neighborhood_plot.png`
- `subtype_shot_charts_on_court.png`
- `average_shot_maps_by_subtype.png`
- `representative_player_shot_scatter.png`
- `shot_style_drift_case_studies.png`
- `comp_comparison_visual.png`
- `sample_player_dossier_visual.png`
- `drift_movement_plot.png`

#### Why it matters
This is the paper / report / presentation visualization layer.

---

### Phase 11 — Build practical player dossier demo package
#### Script
- `11_build_practical_player_dossiers_fixed.py`

#### Purpose
Converts the final Deliverable 2 table into a small set of dossier-ready player cards and player-specific shot-chart figures for front-office communication.

#### Inputs
- `Output\Player Archetype Analysis\final_player_archetype_profile_table.csv`
- `Output\Player Archetype Analysis\shot_style_player_table.csv`
- `data\Shot Chart Details Raw\raw_shotchart_S1_to_S4_main.csv`

#### Outputs
##### Structured dossier package
Written to:
- `Output\Player Archetype Analysis\player_dossier_demo`

Includes:
- `report_ready_player_dossier_table.csv`
- `report_ready_player_dossier_selection.csv`
- `report_ready_player_dossier_manifest.csv`
- `report_ready_player_dossiers.md`

##### Dossier visuals
Written to:
- `visual\Player Archetype\player_dossier_demo`

Includes one generated player-specific dossier chart per selected showcase player.

For the paper, Trae Young and Nikola Vučević are used as paper-facing case-study cards through report-side selection or manual override, not as hardcoded behavior of the existing script.

---

### Phase 12 — Clean salary data and build the Year-5 target
#### Script
- `12_build_year5_salary_targets.py`

#### Purpose
Builds the clean Year-5 salary target needed for Deliverable 3 by combining historical salary rows with historical salary-cap rows.

#### Inputs
- `data\player_feature_table_1999_2019.csv`
- `data\SalaryData\salaries_players_1990_2025_combined.csv`
- `data\SalaryData\Cleaned_Salary_Cap.csv`

#### Outputs
- `Output\Player Archetype Analysis\SalaryBlock\salary_player_season_agg.csv`
- `Output\Player Archetype Analysis\SalaryBlock\salary_cap_cleaned.csv`
- `Output\Player Archetype Analysis\SalaryBlock\year5_salary_target_table.csv`
- `Output\Player Archetype Analysis\SalaryBlock\year5_salary_unmatched_diagnostic.csv`
- `Output\Player Archetype Analysis\SalaryBlock\year5_salary_merge_summary.csv`

#### Logic
This phase does the following explicitly:

- aggregates salary rows to **one player-season total**, because players can appear multiple times in the same season after team changes,
- repairs malformed salary-cap season strings before merging,
- defines Year 5 as the season starting in `draft_year + 4`,
- matches salary records by **normalized player name + season**,
- uses a small explicit alias table for obvious name variants,
- builds  
  `year5_salary_cap_pct = year_salary_total / salary_cap`.

This is the clean, auditable salary target construction step. It is the only supported input path for the salary-ready bridge.

---

### Phase 13 — Build Block 2: Archetype and Comp-Market Context
#### Script
- `13_build_block2_comp_market_context.py`

#### Purpose
Translates finalized Deliverable 2 outputs into the salary-ready **Block 2** table for Deliverable 3.

#### Inputs
- `Output\Player Archetype Analysis\final_player_archetype_profile_table.csv`
- `Output\Player Archetype Analysis\realistic_comps.csv`
- `Output\Player Archetype Analysis\SalaryBlock\year5_salary_target_table.csv`

#### Outputs
- `Output\Player Archetype Analysis\SalaryBlock\comp_salary_detail_table.csv`
- `Output\Player Archetype Analysis\SalaryBlock\deliverable3_block2_archetype_comp_market_context.csv`

#### Logic
This phase does the following:

- merges realistic comps onto the cleaned Year-5 salary target table,
- computes a similarity-weighted salary anchor using realistic comps only,
- builds:
  - comp salary match count,
  - comp salary match rate,
  - weighted mean Year-5 cap percentage,
  - weighted p25 / p50 / p75 anchor values,
  - comp support band,
  - plain-language market-context interpretation,
- keeps ceiling comps as **upside-only context**, not pricing anchors.

This is the clean bridge from Deliverable 2 into Deliverable 3. It turns archetype and comp information into a comp-market salary neighborhood that can be combined later with Deliverable 1 forecasts and Deliverable 3 risk logic.

> **Important note:** use the current patched version of `13_build_block2_comp_market_context.py`. Earlier copies may fail when `comp_salary_match_count` is missing for players with no matched comp-salary support. In addition, out of 1,058 total players, 665 matched, 393 unmatched, the unmatched share is about 37.1% which matches the natural NBA draft picks survival rate after their first four-year contract expires.

---

### Phase 10 — Write the workflow log
#### Script
- `10_write_workflow_log.py`

#### Purpose
Appends workflow structure, assumptions, warnings, quality checks, and later-review notes to the analysis workflow log.

#### Inputs
- active Deliverable 2 structure

#### Outputs
- appended content in `Output\Player Archetype Analysis\PatchLogs\player_archetype_analysis_workflow_log.txt`

#### Important note
Scripts `12_build_year5_salary_targets.py` and `13_build_block2_comp_market_context.py` already append their own phase entries to the workflow log. `10_write_workflow_log.py` should therefore be treated as the final documentation append step rather than the only logging step.

---

## 5. Recommended execution order

Run the active Deliverable 2 and salary-bridge scripts in this exact order:

```text
00_project_inventory.py
01_cohort_coverage_audit.py
02_build_macro_archetype_table.py
03_build_shotstyle_tensors.py
04_train_shot_autoencoder.py
05_cluster_shot_embeddings.py
06_build_identity_drift.py
07_build_comps.py
08_assemble_player_archetype_profiles.py
09_make_player_archetype_visuals.py
11_build_practical_player_dossiers_fixed.py
12_build_year5_salary_targets.py
13_build_block2_comp_market_context.py
10_write_workflow_log.py
```

This order matters because:

- later scripts depend on canonical outputs from earlier phases,
- visuals and dossiers depend on the final profile table,
- and the final log should reflect the completed pipeline.

---

## 6. Deliverable 2 outputs that matter most for the paper

If the report only needs the final Deliverable 2 artifacts, the most important outputs are:

### Core analytical tables
- `Output\Player Archetype Analysis\archetype_macro_player_table.csv`
- `Output\Player Archetype Analysis\player_hybrid_archetype_table.csv`
- `Output\Player Archetype Analysis\player_identity_drift_table.csv`
- `Output\Player Archetype Analysis\realistic_comps.csv`
- `Output\Player Archetype Analysis\ceiling_comps_hof.csv`
- `Output\Player Archetype Analysis\final_player_archetype_profile_table.csv`

### Supporting visuals
- `visual\Player Archetype\subtype_shot_charts_on_court.png`
- `visual\Player Archetype\representative_player_shot_scatter.png`
- `visual\Player Archetype\shot_style_drift_case_studies.png`
- `visual\Player Archetype\sample_player_dossier_visual.png`

### Dossier package
- `Output\Player Archetype Analysis\player_dossier_demo\report_ready_player_dossiers.md`
- `visual\Player Archetype\player_dossier_demo\*.png`

### Salary-ready bridge outputs
- `Output\Player Archetype Analysis\SalaryBlock\year5_salary_target_table.csv`
- `Output\Player Archetype Analysis\SalaryBlock\year5_salary_merge_summary.csv`
- `Output\Player Archetype Analysis\SalaryBlock\year5_salary_unmatched_diagnostic.csv`
- `Output\Player Archetype Analysis\SalaryBlock\comp_salary_detail_table.csv`
- `Output\Player Archetype Analysis\SalaryBlock\deliverable3_block2_archetype_comp_market_context.csv`
---

## 7. Reproducibility rules

### Rule 1
Treat `src\Player Archetypes Analysis\00_*.py` to `11_*.py` as the **authoritative Deliverable 2 pipeline**.

### Rule 2
Treat saved K-means artifacts as **reused upstream infrastructure**, not as something to refit inside Deliverable 2.

### Rule 3
Treat the shot-style embedding layer as **rebuilt inside Deliverable 2** through `03_build_shotstyle_tensors.py` and `04_train_shot_autoencoder.py`.

---

# Deliverable 3: Salary Decision Support Workflow

This section documents the **active reproducible workflow** for **Deliverable 3** in `STAT-946-Case-Study-3`.

The Deliverable 3 implementation is a dedicated salary-decision pipeline built on top of:

- **Deliverable 2 outputs** from `Output\Player Archetype Analysis`
- **salary Block 2 handoff tables** from `Output\Player Archetype Analysis\SalaryBlock`
- **Deliverable 1 forecast handoff tables** merged through the patched Block 1 bridge
- and a dedicated set of salary scripts that write all final Deliverable 3 artifacts to **`Output\Salary Decision Support`**

The project framing for Deliverable 3 is a **staged salary-decision-support system**, not a single overclaimed black-box recommendation engine.

It is designed to answer the front-office question:

> **What is this player worth at the Year-5 extension point, what is the defensible market band, and how aggressive should the team be given archetype context, comp support, and current forecast evidence?**

The final Deliverable 3 package is expected to produce:

1. a canonical Year-5 salary-cap percentage target audit,
2. a protected / fair / walk-away comp-market anchor band,
3. conservative Block-2-only provisional extension logic,
4. stakeholder-facing provisional decision cards,
5. a leakage-aware baseline Year-5 salary model,
6. a salary-model / market-band reconciliation layer,
7. an interim forecast-adjusted guidance layer,
8. an official Block 1 forecast handoff bridge,
9. a staged final framework that separates supported forecast-adjusted rows from unsupported rows,
10. and a completion note that explicitly states what is complete now versus what remains conditional.

---

## 1. Workflow structure

### 1.1 Active code path

The current Deliverable 3 execution chain is:

1. `build_deliverable3_block1_workflow_patched.py`
2. `00_project_inventory.py`
3. `01_salary_target_audit.py`
4. `02_build_market_anchor_band.py`
5. `03_build_provisional_decision_inputs.py`
6. `04_assemble_provisional_decision_cards.py`
7. `05_prepare_salary_model_table.py`
8. `06_fit_baseline_salary_models.py`
9. `07_build_salary_reconciliation_layer.py`
10. `08_build_interim_extension_guidance_patched.py`
11. `09_build_staged_extension_framework_patched.py`
12. `10_write_framework_completion_note_patched.py`

These scripts together implement the current Deliverable 3 architecture.

### 1.2 Utility module actively used by the whole chain

- `salary_workflow_utils.py`

This module defines the canonical expected-file map and shared helpers used across multiple phases, including:

- expected Deliverable 2 backbone files under `Output\Player Archetype Analysis`
- expected salary Block 2 files under `Output\Player Archetype Analysis\SalaryBlock`
- case-study demo player registry
- recursive file-resolution logic
- inventory serialization and workflow-log helpers

This is important for reproducibility because the salary pipeline does **not** assume every file is already in the perfect folder. It first resolves expected paths and records what was actually found.

### 1.3 Output location rule

The current Deliverable 3 workflow writes its generated artifacts to:

- `Output\Salary Decision Support`

This includes:

- core CSV tables,
- markdown summaries,
- metadata JSON files,
- workflow logs,
- case-study extracts,
- and the final staged framework exports.

Unlike Deliverable 2, the current Deliverable 3 bundle is primarily a **table / rules / audit** workflow rather than a figure-heavy visual workflow.

---

## 2. Upstream inputs and reusable artifacts

### 2.1 Deliverable 2 backbone inputs consumed by Deliverable 3

These are read from `Output\Player Archetype Analysis`:

- `archetype_macro_player_table.csv`
- `player_hybrid_archetype_table.csv`
- `player_identity_drift_table.csv`
- `realistic_comps.csv`
- `ceiling_comps_hof.csv` *(optional; not the main pricing anchor)*
- `final_player_archetype_profile_table.csv`

### 2.2 Salary Block 2 inputs consumed by Deliverable 3

These are read from `Output\Player Archetype Analysis\SalaryBlock`:

- `year5_salary_target_table.csv`
- `year5_salary_merge_summary.csv`
- `year5_salary_unmatched_diagnostic.csv`
- `comp_salary_detail_table.csv`
- `deliverable3_block2_archetype_comp_market_context.csv`

### 2.3 Deliverable 1 / Block 1 forecast inputs

The patched Block 1 bridge reads the current Deliverable 1 exports and rewrites them into a Deliverable-3-ready handoff. The relevant source files are:

- `final_lstm_predictions_all_players.csv`
- `player_train_test_split_with_score.csv`
- `player_performance_testset_forecast_result.csv`
- `all_player_stats_1999-2025.csv`

These are converted into:

- `deliverable3_block1_forecast_handoff.csv`
- `deliverable3_block1_join_audit.csv`
- `deliverable3_block1_case_study_examples.csv`

### 2.4 Important workflow principle

Deliverable 3 is intentionally **staged**.

That means:

- the Year-5 salary target and comp-market anchor are treated as the pricing backbone,
- the baseline salary model is treated as a disciplined secondary estimate,
- Deliverable 1 forecast support is merged later rather than forced into every early phase,
- forecast-adjusted logic is only activated on the supported subset,
- and unsupported rows remain clearly labeled as provisional or classification-overlay-only rather than pretending the whole cohort has equally strong evidence.

---

## 3. Step-wise workflow structure

### Phase Block 1 — Deliverable 1 to Deliverable 3 forecast bridge
#### Script
- `build_deliverable3_block1_workflow_patched.py`

#### Purpose
Builds the official Deliverable 3 **Block 1 forecast handoff** from current Deliverable 1 files. It standardizes the forecast-side fields needed later in Deliverable 3, resolves player identity, retains the earliest Year-1-to-Year-4 window per player, and produces the official bridge table used by the staged final framework.

#### Inputs
- current Deliverable 1 class-probability table
- current Deliverable 1 train/test score table
- current Deliverable 1 held-out performance forecast table
- all-player statistics table used to recover Year-5 to Year-7 realized boxscore context

#### Outputs
- `Output\Salary Decision Support\deliverable3_block1_forecast_handoff.csv`
- `Output\Salary Decision Support\deliverable3_block1_join_audit.csv`
- `Output\Salary Decision Support\deliverable3_block1_case_study_examples.csv`
- `Output\Salary Decision Support\deliverable3_block1_workflow_log.txt`

#### What these outputs are used for
- `deliverable3_block1_forecast_handoff.csv` is the **authoritative Deliverable 3 Block 1 bridge** used in Phase 9.
- `deliverable3_block1_join_audit.csv` verifies which players received ID resolution, class probabilities, realized component truth, and held-out test predictions.
- `deliverable3_block1_case_study_examples.csv` provides ready-to-reference demo rows for report writing and stakeholder examples.

---

### Phase 0 — Project map and path audit
#### Script
- `00_project_inventory.py`

#### Purpose
Audits the canonical Deliverable 3 file map, resolves expected upstream files from the project root, and writes an auditable inventory before any salary logic starts.

#### Inputs
- project root
- expected Deliverable 2 and SalaryBlock file map encoded in `salary_workflow_utils.py`

#### Outputs
- `Output\Salary Decision Support\project_inventory.csv`
- `Output\Salary Decision Support\project_inventory.md`
- `Output\Salary Decision Support\project_inventory_summary.json`
- `Output\Salary Decision Support\salary_decision_support_workflow_log.txt`

#### What these outputs are used for
- `project_inventory.csv` becomes the reproducibility reference for later phases that need to resolve upstream paths.
- `project_inventory.md` is a human-readable audit of what the workflow found.
- `project_inventory_summary.json` records required/optional file counts and missing roles.
- the workflow log starts the Deliverable 3 audit trail.

---

### Phase 1 — Canonical Year-5 salary target audit
#### Script
- `01_salary_target_audit.py`

#### Purpose
Treats `year5_salary_target_table.csv` as the canonical Deliverable 3 target backbone, recomputes coverage statistics, profiles matched versus unmatched salary rows, and creates a case-study holdout manifest for later phases.

#### Inputs
- `year5_salary_target_table.csv`
- `year5_salary_merge_summary.csv`
- `year5_salary_unmatched_diagnostic.csv`
- `final_player_archetype_profile_table.csv`
- optional `project_inventory.csv` from Phase 0

#### Outputs
- `Output\Salary Decision Support\salary_target_audit_summary.csv`
- `Output\Salary Decision Support\salary_target_audit_summary.md`
- `Output\Salary Decision Support\salary_target_match_by_draft_year.csv`
- `Output\Salary Decision Support\salary_target_match_type_breakdown.csv`
- `Output\Salary Decision Support\salary_target_contract_flag_summary.csv`
- `Output\Salary Decision Support\salary_target_cap_pct_distribution_summary.csv`
- `Output\Salary Decision Support\salary_target_match_by_macro_archetype.csv`
- `Output\Salary Decision Support\salary_target_match_by_identity_drift.csv`
- `Output\Salary Decision Support\salary_target_unmatched_with_profile.csv`
- `Output\Salary Decision Support\salary_demo_player_holdout_manifest.csv`
- `Output\Salary Decision Support\salary_target_audit_metadata.json`

#### What these outputs are used for
- the summary and breakdown tables verify that the salary target is usable before modeling or pricing logic begins.
- `salary_target_unmatched_with_profile.csv` is the main diagnostic file for unresolved Year-5 salary rows.
- `salary_demo_player_holdout_manifest.csv` reserves the report/demo players from later salary-model fitting so those rows can be used as stakeholder case studies instead of leaking into model calibration.

---

### Phase 2 — Build the comp-market anchor band
#### Script
- `02_build_market_anchor_band.py`

#### Purpose
Converts Block 2 comp-market context into a reusable one-row-per-player price board with:

- **protected price**
- **fair price**
- **walk-away max**

This phase audits the preassembled Block 2 anchor information against the row-level comp-salary detail table instead of blindly trusting only one upstream table.

#### Inputs
- `deliverable3_block2_archetype_comp_market_context.csv`
- `comp_salary_detail_table.csv`
- optional `salary_demo_player_holdout_manifest.csv`
- optional `project_inventory.csv`

#### Outputs
- `Output\Salary Decision Support\salary_market_anchor_band_table.csv`
- `Output\Salary Decision Support\salary_market_anchor_band_summary.csv`
- `Output\Salary Decision Support\salary_market_anchor_band_summary.md`
- `Output\Salary Decision Support\salary_market_anchor_support_breakdown.csv`
- `Output\Salary Decision Support\salary_market_anchor_fair_price_distribution.csv`
- `Output\Salary Decision Support\salary_market_anchor_band_metadata.json`

#### Core design choices
- protected price = comp p25 with recomputed-detail fallback
- fair price = weighted mean with p50 fallback
- walk-away max = comp p75 with recomputed-detail fallback
- ceiling comps are **not** used as the main pricing anchor
- no Deliverable 1 premium or discount is applied here

#### What these outputs are used for
- `salary_market_anchor_band_table.csv` is the canonical pricing-band table used by Phases 3 and 7.
- the support and distribution summaries document where the market band is strong versus weak.
- the markdown and metadata files preserve the exact band-construction logic.

---

### Phase 3 — Build provisional Block-2-only decision inputs
#### Script
- `03_build_provisional_decision_inputs.py`

#### Purpose
Builds conservative Block-2-only extension inputs from the market-anchor table and comp/archetype context. This phase deliberately stops short of final recommendation language and instead prepares structured rule-based decision features.

#### Inputs
- `salary_market_anchor_band_table.csv`
- `deliverable3_block2_archetype_comp_market_context.csv`
- optional `salary_demo_player_holdout_manifest.csv`
- optional `project_inventory.csv`

#### Outputs
- `Output\Salary Decision Support\salary_provisional_decision_inputs_table.csv`
- `Output\Salary Decision Support\salary_provisional_decision_inputs_summary.csv`
- `Output\Salary Decision Support\salary_provisional_comp_support_breakdown.csv`
- `Output\Salary Decision Support\salary_provisional_action_by_scarcity.csv`
- `Output\Salary Decision Support\salary_comp_support_logic_reference.csv`
- `Output\Salary Decision Support\salary_scarcity_wording_reference.csv`
- `Output\Salary Decision Support\salary_provisional_action_rule_reference.csv`
- `Output\Salary Decision Support\salary_provisional_decision_inputs_summary.md`
- `Output\Salary Decision Support\salary_provisional_decision_inputs_metadata.json`

#### Logic
This phase:

- maps macro archetypes to scarcity / replaceability tiers,
- classifies prototype ambiguity and anchor-width bands,
- scores comp-support strength using match count, match rate, effective comp depth, same-macro share, and band width,
- and assigns **provisional action buckets** such as:
  - `offer_now`
  - `offer_now_disciplined_band`
  - `wait_and_save_flexibility`
  - `avoid_overcommitting`

#### What these outputs are used for
- `salary_provisional_decision_inputs_table.csv` is the main structured rule table for Phase 4 and part of the later reconciliation logic.
- the rule-reference files document exactly how scarcity and support wording were assigned.
- the summary files make the Block-2-only logic auditable before any forecast adjustment is added.

---

### Phase 4 — Assemble provisional stakeholder-facing decision cards
#### Script
- `04_assemble_provisional_decision_cards.py`

#### Purpose
Transforms the Phase 3 rule table into stakeholder-facing provisional decision cards and case-study extracts suitable for report writing and communication.

#### Inputs
- `salary_provisional_decision_inputs_table.csv`

#### Outputs
- `Output\Salary Decision Support\deliverable3_provisional_decision_cards.csv`
- `Output\Salary Decision Support\deliverable3_provisional_case_study_table.csv`
- `Output\Salary Decision Support\deliverable3_provisional_decision_card_manifest.csv`
- `Output\Salary Decision Support\deliverable3_provisional_decision_cards_summary.csv`
- `Output\Salary Decision Support\deliverable3_provisional_decision_cards_summary.md`
- `Output\Salary Decision Support\deliverable3_provisional_decision_cards_metadata.json`

#### What these outputs are used for
- the card table is the first communication-ready Deliverable 3 layer.
- the case-study table isolates the reserved demo examples.
- the manifest and summary files document exactly which card fields were assembled and how many rows were produced.

#### Important boundary
These cards are still **provisional**. They are not the final Deliverable 3 recommendation layer.

---

### Phase 5 — Prepare the leakage-aware Year-5 salary modeling table
#### Script
- `05_prepare_salary_model_table.py`

#### Purpose
Builds the modeling-ready Year-5 salary table for baseline regression. It merges the canonical salary target with reusable Deliverable 2 predictors while explicitly separating true predictors from audit-only fields and training-control metadata.

#### Inputs
- `year5_salary_target_table.csv`
- `player_identity_drift_table.csv`
- `archetype_macro_player_table.csv`
- `final_player_archetype_profile_table.csv`
- optional `salary_demo_player_holdout_manifest.csv`
- optional `project_inventory.csv`

#### Outputs
- `Output\Salary Decision Support\year5_salary_modeling_table.csv`
- `Output\Salary Decision Support\year5_salary_modeling_dictionary.csv`
- `Output\Salary Decision Support\year5_salary_modeling_missingness.csv`
- `Output\Salary Decision Support\year5_salary_training_manifest.csv`
- `Output\Salary Decision Support\year5_salary_modeling_summary.csv`
- `Output\Salary Decision Support\year5_salary_modeling_summary.md`
- `Output\Salary Decision Support\year5_salary_modeling_metadata.json`

#### Logic
This phase:

- keeps `year5_salary_target_table.csv` as the canonical target backbone,
- reuses the richest available pre-Year-5 predictor family from Deliverable 2,
- builds a feature dictionary that marks fields as identifiers, targets, training metadata, numeric predictors, categorical predictors, or audit-only context,
- and creates training flags so reserved case-study players are excluded from calibration.

#### What these outputs are used for
- `year5_salary_modeling_table.csv` is the direct input to Phase 6.
- `year5_salary_modeling_dictionary.csv` documents exactly which fields are eligible for baseline modeling.
- `year5_salary_training_manifest.csv` records which rows are trainable, observed, or held out.
- `year5_salary_modeling_missingness.csv` supports verifiability by exposing missingness before fitting begins.

---

### Phase 6 — Fit baseline Year-5 salary models
#### Script
- `06_fit_baseline_salary_models.py`

#### Purpose
Fits disciplined baseline Year-5 salary models on the Phase 5 table and selects the best model using draft-year-aware grouped out-of-fold validation.

#### Inputs
- `year5_salary_modeling_table.csv`
- `year5_salary_training_manifest.csv`
- `year5_salary_modeling_dictionary.csv`

#### Outputs
- `Output\Salary Decision Support\year5_salary_baseline_oof_predictions.csv`
- `Output\Salary Decision Support\year5_salary_baseline_cv_fold_metrics.csv`
- `Output\Salary Decision Support\year5_salary_baseline_model_comparison.csv`
- `Output\Salary Decision Support\year5_salary_baseline_selected_model_coefficients.csv`
- `Output\Salary Decision Support\year5_salary_baseline_selected_model_predictions.csv`
- `Output\Salary Decision Support\year5_salary_baseline_demo_holdout_predictions.csv`
- `Output\Salary Decision Support\year5_salary_baseline_summary.csv`
- `Output\Salary Decision Support\year5_salary_baseline_summary.md`
- `Output\Salary Decision Support\year5_salary_baseline_metadata.json`
- `Output\Salary Decision Support\year5_salary_baseline_selected_model.joblib` **or** `.json` depending on the selected model

#### Logic
This phase evaluates disciplined benchmark models such as:

- global mean
- macro-archetype mean
- linear regression
- ridge regression variants

using **GroupKFold by draft year**.

#### What these outputs are used for
- `year5_salary_baseline_selected_model_predictions.csv` is the main baseline estimate table later merged in Phase 7.
- the OOF and fold-metric files provide the core validation evidence.
- the selected-model artifact preserves the fitted object for reproducibility.
- the demo holdout table gives baseline predictions for case-study players without leaking them into training.

---

### Phase 7 — Reconcile the salary model with the market band
#### Script
- `07_build_salary_reconciliation_layer.py`

#### Purpose
Builds the bridge layer that reconciles:

- the baseline salary estimate,
- the protected / fair / walk-away market band,
- the Block-2 provisional decision context,
- and the current Deliverable 1 macro forecast overlay when available.

#### Inputs
- `year5_salary_baseline_selected_model_predictions.csv`
- `salary_market_anchor_band_table.csv`
- `salary_provisional_decision_inputs_table.csv`
- optional current Deliverable 1 macro forecast table

#### Outputs
- `Output\Salary Decision Support\deliverable3_salary_reconciliation_table.csv`
- `Output\Salary Decision Support\deliverable3_salary_reconciliation_summary.csv`
- `Output\Salary Decision Support\deliverable3_salary_reconciliation_summary.md`
- `Output\Salary Decision Support\deliverable3_forecast_overlay_merge_audit.csv`
- `Output\Salary Decision Support\deliverable3_forecast_overlay_support_breakdown.csv`
- `Output\Salary Decision Support\deliverable3_salary_reconciliation_case_study_table.csv`
- `Output\Salary Decision Support\deliverable3_salary_reconciliation_metadata.json`

#### Logic
This phase:

- compares model price versus market band,
- classifies alignment buckets such as `below_protected`, `between_protected_and_fair`, `between_fair_and_walkaway`, and `above_walkaway`,
- derives pricing-anchor preference flags,
- and adds interim forecast-overlay context without yet claiming that the workflow is fully final.

#### What these outputs are used for
- `deliverable3_salary_reconciliation_table.csv` is the core bridge into Phase 8 and Phase 9.
- the merge-audit file verifies how forecast-side information was overlaid.
- the summary and case-study tables support report writing and diagnostics.

---

### Phase 8 — Build interim forecast-adjusted extension guidance
#### Script
- `08_build_interim_extension_guidance_patched.py`

#### Purpose
Upgrades the reconciliation layer into an **interim** forecast-adjusted extension-guidance table. This is still conservative and explicitly not the final fully validated framework.

#### Inputs
- `deliverable3_salary_reconciliation_table.csv`

#### Outputs
- `Output\Salary Decision Support\deliverable3_interim_extension_guidance_table.csv`
- `Output\Salary Decision Support\deliverable3_interim_extension_guidance_cards.csv`
- `Output\Salary Decision Support\deliverable3_interim_extension_case_study_table.csv`
- `Output\Salary Decision Support\deliverable3_interim_extension_summary.csv`
- `Output\Salary Decision Support\deliverable3_interim_extension_rule_reference.csv`
- `Output\Salary Decision Support\deliverable3_interim_extension_summary.md`
- `Output\Salary Decision Support\deliverable3_interim_extension_metadata.json`

#### Logic
This patched phase:

- converts the reconciliation table into forecast-support tiers,
- upgrades Block-2-only stances into interim extension buckets,
- adds negotiation posture fields,
- and preserves the main report demos, now including Andrew Bynum when present in the staged patched workflow.

#### What these outputs are used for
- the interim table and cards provide a communication-ready bridge before the final staged framework is assembled.
- the case-study extract is useful for report examples and appendix tables.
- the rule-reference file records how interim forecast-support logic was translated into action buckets.

#### Important boundary
This layer remains **interim_with_current_D1_macro_forecast** and should not be presented as the final extension recommendation.

---

### Phase 9 — Assemble the staged final Deliverable 3 framework
#### Script
- `09_build_staged_extension_framework_patched.py`

#### Purpose
Assembles the current staged final Deliverable 3 framework by replacing earlier ad hoc forecast overlays with the official Deliverable 3 Block 1 handoff.

#### Inputs
- `deliverable3_salary_reconciliation_table.csv`
- `deliverable3_block1_forecast_handoff.csv`
- `deliverable3_block1_case_study_examples.csv`

#### Outputs
- `Output\Salary Decision Support\deliverable3_staged_final_framework_table.csv`
- `Output\Salary Decision Support\deliverable3_staged_final_guidance_cards.csv`
- `Output\Salary Decision Support\deliverable3_final_case_study_table.csv`
- `Output\Salary Decision Support\deliverable3_framework_status_matrix.csv`
- `Output\Salary Decision Support\deliverable3_forecast_adjustment_supported_subset.csv`
- `Output\Salary Decision Support\deliverable3_forecast_adjustment_supported_subset_summary.csv`
- `Output\Salary Decision Support\deliverable3_forecast_adjustment_supported_subset_crosstab.csv`
- `Output\Salary Decision Support\deliverable3_staged_final_summary.csv`
- `Output\Salary Decision Support\deliverable3_staged_final_summary.md`
- `Output\Salary Decision Support\deliverable3_final_rule_reference.csv`
- `Output\Salary Decision Support\deliverable3_staged_final_metadata.json`

#### Logic
This patched phase explicitly separates:

- rows with official Block 1 class probabilities,
- rows with supported forecast-adjustment eligibility,
- rows that remain classification-overlay-only,
- and rows that still remain Block-2-only provisional outputs.

It also enforces the project’s conservative rule:

- upside can move the stance **within** the existing band,
- downside can pull the stance more cautious **within** the existing band,
- unsupported rows do **not** receive overclaimed forecast adjustments.

#### What these outputs are used for
- `deliverable3_staged_final_framework_table.csv` is the main final analytical table for Deliverable 3.
- `deliverable3_staged_final_guidance_cards.csv` is the main stakeholder-facing final card layer.
- `deliverable3_final_case_study_table.csv` is the canonical case-study output for the report.
- the supported-subset files isolate the only rows where forecast-adjusted logic is currently defensible.

---

### Phase 10 — Write the framework completion note
#### Script
- `10_write_framework_completion_note_patched.py`

#### Purpose
Writes the final status note that explains what Deliverable 3 has genuinely completed and what still remains conditional.

#### Inputs
- `deliverable3_staged_final_summary.csv`
- `deliverable3_forecast_adjustment_supported_subset_summary.csv`
- `deliverable3_final_case_study_table.csv`

#### Outputs
- `Output\Salary Decision Support\deliverable3_framework_completion_note.md`
- `Output\Salary Decision Support\deliverable3_framework_completion_note_metadata.json`
- appended entry in `Output\Salary Decision Support\salary_decision_support_workflow_log.txt`

#### What these outputs are used for
- the completion note is the final honesty check for the workflow.
- it documents that Deliverable 3 is completed as a **staged salary-decision-support framework**, not a fully unconstrained recommendation engine.
- it records the final case-study status and supported-subset metrics used in the report narrative.

---

## 4. Reproducibility and interpretation notes

### 4.1 Canonical path assumptions

For this uploaded project bundle, the salary scripts assume:

- archetype inputs are read from `Output\Player Archetype Analysis`
- salary Block 2 inputs are read from `Output\Player Archetype Analysis\SalaryBlock`
- Deliverable 3 generated outputs are written to `Output\Salary Decision Support`

### 4.2 Why there are both provisional, interim, and staged-final layers

These layers are intentional, not redundant.

- **Provisional** = Block-2-only decision structure before forecast integration
- **Interim** = forecast-adjusted layer using the current macro overlay
- **Staged final** = official current framework after the patched Block 1 handoff is merged

This is how the workflow avoids overstating evidence quality.

### 4.3 What should be treated as the main current Deliverable 3 outputs

For most downstream reporting, the key current files are:

- `deliverable3_staged_final_framework_table.csv`
- `deliverable3_staged_final_guidance_cards.csv`
- `deliverable3_final_case_study_table.csv`
- `deliverable3_framework_status_matrix.csv`
- `deliverable3_staged_final_summary.csv`
- `deliverable3_framework_completion_note.md`

### 4.4 Important boundary on interpretation

Deliverable 3 is currently strongest as a **transparent staged framework**.

It should be described as:

- salary-cap target audited,
- market-anchor band built,
- provisional and interim logic documented,
- official Block 1 handoff integrated,
- staged final framework assembled,

but **not** as:

- a fully validated single-number contract engine for the whole cohort,
- a finished durability / availability risk model,
- or a full-cohort forecast-adjusted final recommendation system with equally strong support for every player.

---

# Deliverable 3.2: Salary Forecasting and Uncertainty

Please see:

`/src/5th_Year_Salary_Analysis/Data_cleanning_For_Forecasting.ipynb`
`/src/5th_Year_Salary_Analysis/Salary_cap_Forecast.ipynb`

This section documents the active reproducible workflow for the Year-5 salary forecasting module in the project.

The salary forecasting implementation is a dedicated pipeline built on top of:

- the cleaned historical NBA player-season table from **`/ data / all_player_stats_1999-2025.csv`**
- the final player classification output from **`/ data / final_lstm_predictions_all_players.csv`**
- the merged salary and salary-cap history tables from **`/ data / SalaryData / salaries_players_1990_2025_combined.csv`** and **`/ Output / Player Archetype Analysis / SalaryBlock / salary_cap_cleaned.csv`**
- the matched archetype tables from the earlier player archetype analysis from **`/ Output / Player Archetype Analysis / SalaryBlock /  deliverable3_block2_archetype_comp_market_context.csv`**
- and a dedicated set of forecasting scripts that write final intermediate artifacts to: **`/src/5th_Year_Salary_Analysis/Result`**

It is designed to answer the front-office style question:

**Given a player's first four NBA seasons, their predicted development class, and their archetype context, how well can we forecast the player's Year-5 salary-cap percentage, and which groups appear to have higher prediction uncertainty or large-error risk?**

The final salary forecasting package is intended to produce:

- a cleaned player-level salary forecasting input table,
- a valid Year-5 target table,
- an archetype- and label-enriched forecasting dataset,
- Ridge and Elastic Net baseline forecasting models with Goal 1 and Goal 1 + Goal 2 feature versions,
- sampled player-level predicted-vs-actual plots,
- full-model predicted-vs-actual plots,
- large-error group diagnostics,
- bias-adjusted uncertainty summaries across player subgroups,
- and chart-ready result files for later visualization and QMD reporting.

---

## 1. Year-5 Salary Forecasting Workflow

### 1.1 Data cleanning steps before forecasting

Please see:

`/src/5th_Year_Salary_Analysis/Data_cleanning_For_Forecasting.ipynb`

The goal of Data_cleanning_For_Forecasting.ipynb is to build the clean, matched, player-level forecasting dataset used by the later salary models.

This file is mainly a data engineering and table preparation. It does not fit the final Ridge or Elastic Net models. Instead, it prepares the exact modelling inputs needed for salary forecasting.

The two main **final output files** from Data_cleanning_For_Forecasting.ipynb are:

1. `/src/5th_Year_Salary_Analysis/Result/df_y5.csv`

This is the Year-5 target table. It contains the player’s 5th-season salary outcome. This file is used as the Y table for the later forecasting model.

2. `/src/5th_Year_Salary_Analysis/Result/df_with_archetype_matched.csv`

This is the main cleaned forecasting feature table. It contains the first-four-year player information matched with:

- cleaned stats history
- predicted class / probabilities / confidence
- and archetype fields like macro_archetype and shot_style_subtype

This file is used as the main X-side input table for later forecasting model.

---

### 1.2 Salary Cap Forecasting

Please see:

`/src/5th_Year_Salary_Analysis/Salary_cap_Forecast.ipynb`

The goal is to forecast a player’s Year-5 salary cap percentage using information from the player’s first four seasons, and then check which player groups have higher prediction uncertainty or large errors. This file tryting to answer:

**How well can we predict a player’s Year-5 salary cap percentage from the first four seasons, and which player groups are harder to predict accurately?**

1. Input files

`/src/5th_Year_Salary_Analysis/Result/df_with_archetype_matched.csv`
`/src/5th_Year_Salary_Analysis/Result/df_y5.csv`
`/data/final_lstm_predictions_all_players.csv`
`/Output/Player Archetype Analysis/SalaryBlock/ deliverable3_block2_archetype_comp_market_context.csv`

This file is using the forecasting ready handoff tables which we built from Data_cleanning_For_Forecasting.ipynb, but those tables already carry information from both the player clustering section and the salary cleaning section.

2. Build the final regression dataset

One important part of this file is reshaping the first four seasons into a player-level wide format. That means instead of keeping one row per season, it creates one row per player with Year 1 to Year 4 information as features. Then it merges this with the Year-5 salary target from df_y5.csv.

3. Fit and compare forecasting models

The notebook then fits regularized regression models, mainly:
- Ridge
- Elastic Net

It tests them under different feature setups, especially:
- Goal 1
- Goal 1 + Goal 2

These feature sets allow the notebook to compare whether adding richer information, such as class and archetype context, improves forecasting performance. The models are then evaluated using measures like: R² and RMSE.

4. Create predicted vs actual visual plots

The notebook does not only report model metrics. It also creates player-level visual comparisons between:
- actual Year-5 salary cap percentage
- predicted Year-5 salary cap percentage

5. Analyze large-error risk across player subgroups

Another major part of this notebook is the uncertainty and error analysis. It defines a large-error condition: |error| > 0.03

Then it studies which player groups have higher large-error rates. The notebook checks these patterns across:
- age group
- predicted class
- macro archetype
- shot style subtype

It also creates combined group summaries and identifies the top groups with the highest bias-adjusted large-error percentage.

6. Save outputs for later use

At the end, the notebook saves chart-ready CSV files into: `/src/5th_Year_Salary_Analysis/Result`. These files are later used by the Python plotting script and by QMD reporting.

If you want to view these results directly, please run `/src/5th_Year_Salary_Analysis/Salary_Forecast.py`. This file will automatically save all results in `/visual/Salary_Forecast/`.

---

## 2. Deliverable 3.2 outputs that matter most for the paper

If the report only needs the final Deliverable 3.2 artifacts, the most important outputs are:

### Group Bias
- `visual\Salary_Forecast\bias_by_shot_age_macro_predclass.png`
- `visual\Salary_Forecast\bias_top5_groups_large_error.png`

### Model Output
- `visual\Salary_Forecast\elastic_net_goal12_full_model.png`
- `visual\Salary_Forecast\elastic_net_goal1_full_model.png`
- `visual\Salary_Forecast\ridge_goal12_full_model.png`
- `visual\Salary_Forecast\ridge_goal1_full_model.png`

### Control Group Comparison
- `visual\Salary_Forecast\selected_players_elastic_net.png`
- `visual\Salary_Forecast\selected_players_ridge.png`
















---

