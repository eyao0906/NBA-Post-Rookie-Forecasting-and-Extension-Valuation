from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# =========================================================
# 1. paths
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

RESULT_DIR = PROJECT_ROOT / "src" / "Player_Performance_Clustering" / "Result"
SAVE_DIR = PROJECT_ROOT / "visual" / "Player_Clustering"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 2. helper: safe read
# =========================================================
def read_csv_safe(path, **kwargs):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, **kwargs)


# =========================================================
# 3. read saved clustering files
# =========================================================
pca_plot_df = read_csv_safe(RESULT_DIR / "pca_plot_df.csv")
tsne_plot_df = read_csv_safe(RESULT_DIR / "tsne_plot_df.csv")

# =========================================================
# 4. read saved model-result files
# =========================================================
comparison_metrics = read_csv_safe(RESULT_DIR / "comparison_metrics.csv")
logit_cm = read_csv_safe(RESULT_DIR / "logit_confusion_matrix_test.csv", index_col=0)
lstm_cm = read_csv_safe(RESULT_DIR / "lstm_confusion_matrix_test.csv", index_col=0)
comparison_summary = read_csv_safe(RESULT_DIR / "comparison_summary_test.csv")

cross_model_cm_path = RESULT_DIR / "cross_model_confusion_matrix_test.csv"
cross_model_cm = pd.read_csv(cross_model_cm_path, index_col=0) if cross_model_cm_path.exists() else None


# =========================================================
# 5. clustering plots
# =========================================================
def save_pca_3d_plot(df, save_path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = {
        "Sleeper": "green",
        "Neutral": "gray",
        "Bust": "red"
    }

    for label in ["Sleeper", "Neutral", "Bust"]:
        temp = df[df["pca_label"] == label]
        if len(temp) > 0:
            ax.scatter(
                temp["PCA1"],
                temp["PCA2"],
                temp["PCA3"],
                label=label,
                s=50,
                alpha=0.7,
                c=colors[label]
            )

    ax.set_title("PCA (3D) + K-means")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.set_zlabel("PCA3")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_tsne_3d_plot(df, save_path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = {
        "Sleeper": "green",
        "Neutral": "gray",
        "Bust": "red"
    }

    for label in ["Sleeper", "Neutral", "Bust"]:
        temp = df[df["tsne_label"] == label]
        if len(temp) > 0:
            ax.scatter(
                temp["TSNE1"],
                temp["TSNE2"],
                temp["TSNE3"],
                label=label,
                s=50,
                alpha=0.7,
                c=colors[label]
            )

    ax.set_title("t-SNE (3D) + K-means")
    ax.set_xlabel("TSNE1")
    ax.set_ylabel("TSNE2")
    ax.set_zlabel("TSNE3")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# 6. table formatting helper
# clean style: no full grid, only top/header/bottom rules
# =========================================================
def format_table_df(df, round_digits=4):
    out = df.copy()

    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(
                lambda x: f"{x:.{round_digits}f}" if pd.notnull(x) else ""
            )
    return out


def save_clean_table_png(
    df,
    save_path,
    title=None,
    show_index=False,
    round_digits=4,
    font_size=12,
    header_font_size=13,
    title_font_size=15,
    row_height=0.095,
    left_cols=None,
    right_cols=None
):
    df_show = df.copy()

    if show_index:
        df_show = df_show.reset_index()

    df_show = format_table_df(df_show, round_digits=round_digits)

    n_rows = len(df_show)
    n_cols = len(df_show.columns)

    # figure size
    fig_width = max(10, 2.2 * n_cols)
    fig_height = max(2.8, 1.4 + row_height * (n_rows + 2) * 10)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # title
    y_top = 0.94
    if title:
        ax.text(
            0.5, y_top, title,
            ha="center", va="center",
            fontsize=title_font_size,
            fontweight="bold"
        )
        y_top -= 0.08

    # column widths
    col_names = list(df_show.columns)

    if left_cols is None:
        left_cols = []
    if right_cols is None:
        right_cols = []

    raw_widths = []
    for col in col_names:
        max_len = max(
            [len(str(col))] + [len(str(x)) for x in df_show[col].tolist()]
        )
        # give more width to text columns
        if col in left_cols:
            width = max_len * 1.45
        else:
            width = max_len * 1.05
        raw_widths.append(width)

    total_width = sum(raw_widths)
    widths = [w / total_width for w in raw_widths]

    x_lefts = [0]
    for w in widths[:-1]:
        x_lefts.append(x_lefts[-1] + w)

    x_rights = [x_lefts[i] + widths[i] for i in range(n_cols)]
    x_centers = [(x_lefts[i] + x_rights[i]) / 2 for i in range(n_cols)]

    # lines
    y_header = y_top
    y_header_line = y_header - 0.035
    y_bottom = y_header_line - row_height * n_rows - 0.03

    # top rule
    ax.hlines(y_header + 0.03, 0, 1, linewidth=1.2, color="black")
    # header rule
    ax.hlines(y_header_line, 0, 1, linewidth=0.8, color="black")
    # bottom rule
    ax.hlines(y_bottom, 0, 1, linewidth=1.2, color="black")

    # header text
    for i, col in enumerate(col_names):
        if col in left_cols:
            ax.text(
                x_lefts[i] + 0.003, y_header,
                str(col),
                ha="left", va="center",
                fontsize=header_font_size
            )
        elif col in right_cols:
            ax.text(
                x_rights[i] - 0.003, y_header,
                str(col),
                ha="right", va="center",
                fontsize=header_font_size
            )
        else:
            ax.text(
                x_centers[i], y_header,
                str(col),
                ha="center", va="center",
                fontsize=header_font_size
            )

    # body text
    for r in range(n_rows):
        y = y_header_line - row_height * (r + 0.5)

        for c, col in enumerate(col_names):
            val = str(df_show.iloc[r, c])

            if col in left_cols:
                ax.text(
                    x_lefts[c] + 0.003, y,
                    val,
                    ha="left", va="center",
                    fontsize=font_size
                )
            elif col in right_cols:
                ax.text(
                    x_rights[c] - 0.003, y,
                    val,
                    ha="right", va="center",
                    fontsize=font_size
                )
            else:
                ax.text(
                    x_centers[c], y,
                    val,
                    ha="center", va="center",
                    fontsize=font_size
                )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# 7. prepare tables for clean output
# =========================================================
comparison_metrics_show = comparison_metrics.copy()

logit_cm_show = logit_cm.copy()
lstm_cm_show = lstm_cm.copy()

comparison_summary_show = comparison_summary.copy()

# optional cleanup of metric names
if "metric" in comparison_summary_show.columns:
    comparison_summary_show["metric"] = comparison_summary_show["metric"].str.replace("_", " ", regex=False)


# =========================================================
# 8. save outputs
# =========================================================
save_pca_3d_plot(
    pca_plot_df,
    SAVE_DIR / "pca_3d_kmeans.png"
)

save_tsne_3d_plot(
    tsne_plot_df,
    SAVE_DIR / "tsne_3d_kmeans.png"
)

save_clean_table_png(
    comparison_metrics_show,
    SAVE_DIR / "model_accuracy_comparison.png",
    title="Model Accuracy Comparison (Test Dataset)",
    show_index=False,
    round_digits=4,
    font_size=13,
    header_font_size=14,
    left_cols=["model"],
    right_cols=["accuracy_test_sample"]
)

save_clean_table_png(
    logit_cm_show,
    SAVE_DIR / "multinomial_logit_confusion_matrix.png",
    title="Multinomial Logistic Regression Confusion Matrix (Test Dataset)",
    show_index=True,
    round_digits=0,
    font_size=13,
    header_font_size=14,
    left_cols=["index"],
    right_cols=list(logit_cm_show.columns)
)

save_clean_table_png(
    lstm_cm_show,
    SAVE_DIR / "lstm_confusion_matrix.png",
    title="LSTM Confusion Matrix (Test Dataset)",
    show_index=True,
    round_digits=0,
    font_size=13,
    header_font_size=14,
    left_cols=["index"],
    right_cols=list(lstm_cm_show.columns)
)

save_clean_table_png(
    comparison_summary_show,
    SAVE_DIR / "model_direct_comparison_summary.png",
    title="Direct Comparison Summary (Test Dataset)",
    show_index=False,
    round_digits=4,
    font_size=12,
    header_font_size=14,
    left_cols=["metric"],
    right_cols=["value"]
)

radar_plot_df = read_csv_safe(RESULT_DIR / "selected_players_radar_plot_data.csv")

def save_selected_players_radar_plot(radar_df, save_path):
    skills = ["Scoring", "Shooting", "Playmaking", "Rebounding", "Defense", "Ball Security"]
    angles = np.linspace(0, 2 * np.pi, len(skills), endpoint=False).tolist()
    angles += angles[:1]

    player_order = ["Trae Young", "Nikola Vucevic"]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 7),
        subplot_kw=dict(polar=True)
    )

    for ax, player_name in zip(axes, player_order):
        temp = radar_df[radar_df["player_name"] == player_name].copy()

        if temp.empty:
            ax.set_title(f"{player_name}\nNo data found", fontsize=15)
            ax.axis("off")
            continue

        meta = temp[["actual_class", "compare_class"]].drop_duplicates().iloc[0]
        actual_class = meta["actual_class"]
        compare_class = meta["compare_class"]

        series_names = temp["series_name"].drop_duplicates().tolist()

        for series_name in series_names:
            s = temp[temp["series_name"] == series_name].copy()
            s["skill"] = pd.Categorical(s["skill"], categories=skills, ordered=True)
            s = s.sort_values("skill")

            values = s["value"].tolist()

            if len(values) != len(skills):
                continue

            values += values[:1]

            if "avg" in series_name.lower():
                ax.plot(angles, values, linewidth=2, linestyle="--", label=series_name, color="gray")
                ax.fill(angles, values, alpha=0.08, color="gray")
            else:
                ax.plot(angles, values, linewidth=2.5, label=series_name, color="tab:blue")
                ax.fill(angles, values, alpha=0.20, color="tab:blue")

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(skills, fontsize=12)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=9)

        ax.set_title(
            f"{player_name}\nActual: {actual_class} | Compare: {compare_class}",
            fontsize=16,
            pad=18
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.10), fontsize=10)

    fig.suptitle("Skill Radar Chart: Player vs Selected Class Average", fontsize=18, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

save_selected_players_radar_plot(
    radar_plot_df,
    SAVE_DIR / "selected_players_radar.png"
)

print("\nDone. Saved these files:")
print(SAVE_DIR / "pca_3d_kmeans.png")
print(SAVE_DIR / "tsne_3d_kmeans.png")
print(SAVE_DIR / "model_accuracy_comparison.png")
print(SAVE_DIR / "multinomial_logit_confusion_matrix.png")
print(SAVE_DIR / "lstm_confusion_matrix.png")
print(SAVE_DIR / "model_direct_comparison_summary.png")
print("Saved radar plot:", SAVE_DIR / "selected_players_radar.png")