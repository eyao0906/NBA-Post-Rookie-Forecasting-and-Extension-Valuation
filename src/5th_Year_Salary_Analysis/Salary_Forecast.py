from pathlib import Path
import textwrap
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# 1. paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

RESULT_DIR = PROJECT_ROOT / "src" / "5th_Year_Salary_Analysis" / "Result"
SAVE_DIR = PROJECT_ROOT  / "visual" / "Salary_Forecast"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 2. helpers
# =========================================================
def read_csv_safe(path, **kwargs):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, **kwargs)


def get_manifest_info(manifest_df, file_name):
    temp = manifest_df[manifest_df["file_name"] == file_name]
    if temp.empty:
        return {}
    return temp.iloc[0].to_dict()


def wrap_labels(labels, width=16):
    return ["\n".join(textwrap.wrap(str(x), width=width)) for x in labels]


def save_line_plot(df, title, save_path, x_col="x_label", actual_col="actual", pred_col="predicted",
                   y_label="Salary Cap Percentage", rotate=55, percent_axis=True):
    fig, ax = plt.subplots(figsize=(11, 5.8))

    x = df[x_col].astype(str).tolist()
    y_actual = df[actual_col]
    y_pred = df[pred_col]

    ax.plot(x, y_actual, marker="o", label="Actual")
    ax.plot(x, y_pred, marker="o", label="Predicted")

    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.legend()
    ax.tick_params(axis="x", rotation=rotate, labelsize=8)

    if percent_axis:
        ax.yaxis.set_major_formatter(lambda v, pos: f"{v:.1%}")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", save_path)


def save_grouped_bar_plot(df, title, save_path, x_col="x_label",
                          ridge_col="Ridge", enet_col="Elastic Net",
                          x_label="", y_label="", rotate=15, wrap_width=20):
    fig, ax = plt.subplots(figsize=(14, 6.5))

    plot_df = df.copy()
    plot_df[x_col] = wrap_labels(plot_df[x_col], width=wrap_width)

    x = range(len(plot_df))
    width = 0.36

    ax.bar([i - width / 2 for i in x], plot_df[ridge_col], width=width, label="Ridge")
    ax.bar([i + width / 2 for i in x], plot_df[enet_col], width=width, label="Elastic Net")

    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df[x_col], rotation=rotate, ha="center")
    ax.legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", save_path)


def save_bias_four_panel(age_df, class_df, macro_df, shot_df, age_info, class_info, macro_info, shot_info, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    axes = axes.flatten()

    panel_data = [
        (age_df, age_info, "By Age Group"),
        (class_df, class_info, "By Predicted Class"),
        (macro_df, macro_info, "By Macro Archetype"),
        (shot_df, shot_info, "By Shot Style Subtype"),
    ]

    for ax, (df, info, fallback_title) in zip(axes, panel_data):
        plot_df = df.copy()
        plot_df["x_label"] = wrap_labels(plot_df["x_label"], width=18)

        x = range(len(plot_df))
        width = 0.36

        ax.bar([i - width / 2 for i in x], plot_df["Ridge"], width=width, label="Ridge")
        ax.bar([i + width / 2 for i in x], plot_df["Elastic Net"], width=width, label="Elastic Net")

        title = info.get("title", fallback_title)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(info.get("y_label", "Bias-adjusted % of players in group with large error"))
        ax.set_xlabel(info.get("x_label", ""))
        ax.set_xticks(list(x))
        ax.set_xticklabels(plot_df["x_label"], rotation=18, ha="center", fontsize=8)
        ax.legend(fontsize=9)

    fig.suptitle(
        "Bias-Adjusted Large-Error Percentage by Shot Style Subtype / Age / Macro Type / Predicted Class (|error| > 0.03)",
        fontsize=15,
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", save_path)


# =========================================================
# 3. read manifest
# =========================================================
manifest = read_csv_safe(RESULT_DIR / "plot_manifest.csv")


# =========================================================
# 4. read chart-ready csv files
# =========================================================
selected_ridge = read_csv_safe(RESULT_DIR / "selected_players_ridge_data.csv")
selected_enet = read_csv_safe(RESULT_DIR / "selected_players_elastic_net_data.csv")

ridge_goal1 = read_csv_safe(RESULT_DIR / "ridge_goal1_full_model_data.csv")
enet_goal1 = read_csv_safe(RESULT_DIR / "elastic_net_goal1_full_model_data.csv")
ridge_goal12 = read_csv_safe(RESULT_DIR / "ridge_goal12_full_model_data.csv")
enet_goal12 = read_csv_safe(RESULT_DIR / "elastic_net_goal12_full_model_data.csv")

top5_bias = read_csv_safe(RESULT_DIR / "bias_top5_groups_large_error_data.csv")

age_bias = read_csv_safe(RESULT_DIR / "bias_by_age_group_data.csv")
class_bias = read_csv_safe(RESULT_DIR / "bias_by_predicted_class_data.csv")
macro_bias = read_csv_safe(RESULT_DIR / "bias_by_macro_archetype_data.csv")
shot_bias = read_csv_safe(RESULT_DIR / "bias_by_shot_style_subtype_data.csv")


# =========================================================
# 5. manifest info lookups
# =========================================================
selected_ridge_info = get_manifest_info(manifest, "selected_players_ridge_data.csv")
selected_enet_info = get_manifest_info(manifest, "selected_players_elastic_net_data.csv")

ridge_goal1_info = get_manifest_info(manifest, "ridge_goal1_full_model_data.csv")
enet_goal1_info = get_manifest_info(manifest, "elastic_net_goal1_full_model_data.csv")
ridge_goal12_info = get_manifest_info(manifest, "ridge_goal12_full_model_data.csv")
enet_goal12_info = get_manifest_info(manifest, "elastic_net_goal12_full_model_data.csv")

top5_bias_info = get_manifest_info(manifest, "bias_top5_groups_large_error_data.csv")

age_bias_info = get_manifest_info(manifest, "bias_by_age_group_data.csv")
class_bias_info = get_manifest_info(manifest, "bias_by_predicted_class_data.csv")
macro_bias_info = get_manifest_info(manifest, "bias_by_macro_archetype_data.csv")
shot_bias_info = get_manifest_info(manifest, "bias_by_shot_style_subtype_data.csv")


# =========================================================
# 6. save the 8 final images
# =========================================================

# 1
save_line_plot(
    selected_ridge,
    title=selected_ridge_info.get("title", "Predicted vs Actual Year-5 Salary Cap Percentage - Ridge").replace('\r', ''),
    save_path=SAVE_DIR / "selected_players_ridge.png",
    x_col="x_label",
    actual_col="actual",
    pred_col="predicted",
    y_label=selected_ridge_info.get("y_label", "Salary Cap Percentage"),
    rotate=0,
    percent_axis=True
)

# 2
save_line_plot(
    selected_enet,
    title=selected_enet_info.get("title", "Predicted vs Actual Year-5 Salary Cap Percentage - Elastic Net").replace('\r', ''),
    save_path=SAVE_DIR / "selected_players_elastic_net.png",
    x_col="x_label",
    actual_col="actual",
    pred_col="predicted",
    y_label=selected_enet_info.get("y_label", "Salary Cap Percentage"),
    rotate=0,
    percent_axis=True
)

# 3
save_grouped_bar_plot(
    top5_bias,
    title=top5_bias_info.get("title", "Top 5 Groups with Highest Bias-Adjusted Large-Error Percentage").replace('\r', ''),
    save_path=SAVE_DIR / "bias_top5_groups_large_error.png",
    x_col="x_label",
    ridge_col="Ridge",
    enet_col="Elastic Net",
    x_label=top5_bias_info.get("x_label", "Group"),
    y_label=top5_bias_info.get("y_label", "Bias-adjusted percentage of players in group with large error (%)"),
    rotate=10,
    wrap_width=20
)

# 4
save_line_plot(
    ridge_goal1,
    title=ridge_goal1_info.get("title", "Ridge + Goal 1").replace('\r', ''),
    save_path=SAVE_DIR / "ridge_goal1_full_model.png",
    x_col="x_label",
    actual_col="actual",
    pred_col="predicted",
    y_label=ridge_goal1_info.get("y_label", "Salary Cap Percentage"),
    rotate=58,
    percent_axis=True
)

# 5
save_line_plot(
    enet_goal1,
    title=enet_goal1_info.get("title", "Elastic Net + Goal 1").replace('\r', ''),
    save_path=SAVE_DIR / "elastic_net_goal1_full_model.png",
    x_col="x_label",
    actual_col="actual",
    pred_col="predicted",
    y_label=enet_goal1_info.get("y_label", "Salary Cap Percentage"),
    rotate=58,
    percent_axis=True
)

# 6
save_line_plot(
    ridge_goal12,
    title=ridge_goal12_info.get("title", "Ridge + Goal 1 + Goal 2").replace('\r', ''),
    save_path=SAVE_DIR / "ridge_goal12_full_model.png",
    x_col="x_label",
    actual_col="actual",
    pred_col="predicted",
    y_label=ridge_goal12_info.get("y_label", "Salary Cap Percentage"),
    rotate=58,
    percent_axis=True
)

# 7
save_line_plot(
    enet_goal12,
    title=enet_goal12_info.get("title", "Elastic Net + Goal 1 + Goal 2").replace('\r', ''),
    save_path=SAVE_DIR / "elastic_net_goal12_full_model.png",
    x_col="x_label",
    actual_col="actual",
    pred_col="predicted",
    y_label=enet_goal12_info.get("y_label", "Salary Cap Percentage"),
    rotate=58,
    percent_axis=True
)

# 8
save_bias_four_panel(
    age_df=age_bias,
    class_df=class_bias,
    macro_df=macro_bias,
    shot_df=shot_bias,
    age_info=age_bias_info,
    class_info=class_bias_info,
    macro_info=macro_bias_info,
    shot_info=shot_bias_info,
    save_path=SAVE_DIR / "bias_by_shot_age_macro_predclass.png"
)

print("\nDone. Saved 8 images to:")
print(SAVE_DIR)