from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
import math
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

TARGET_COL = "year5_salary_cap_pct"
TRAIN_FLAG_COL = "training_eligible_flag"
OBSERVED_FLAG_COL = "target_observed_flag"
GROUP_COL = "split_group_draft_year"
HOLDOUT_COL = "holdout_for_case_study"
STATUS_COL = "modeling_row_status"
UPPER_CLIP = 0.35


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_output_dir(project_root: Path) -> Path:
    candidates = [
        project_root / "Output/Salary%20Decision%20Support",
        project_root / "Output/Salary Decision Support",
    ]
    for c in candidates:
        if c.exists():
            return ensure_dir(c)
    return ensure_dir(candidates[0])


def append_workflow_log(log_path: Path, text_block: str) -> None:
    ensure_dir(log_path.parent)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n{'=' * 80}\n")
        fh.write(f"[{timestamp}]\n")
        fh.write(text_block.rstrip() + "\n")


def find_existing_path(project_root: Path, candidates: list[str]) -> Path:
    for candidate in candidates:
        path = project_root / candidate
        if path.exists():
            return path
    for candidate in candidates:
        basename = Path(candidate).name
        matches = sorted(project_root.rglob(basename), key=lambda p: (len(p.parts), str(p)))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not locate any of: {candidates}")


def load_inputs(project_root: Path) -> dict[str, Path]:
    return {
        "modeling_table": find_existing_path(project_root, [
            "Output/Salary%20Decision%20Support/year5_salary_modeling_table.csv",
            "Output/Salary Decision Support/year5_salary_modeling_table.csv",
            "year5_salary_modeling_table.csv",
        ]),
        "training_manifest": find_existing_path(project_root, [
            "Output/Salary%20Decision%20Support/year5_salary_training_manifest.csv",
            "Output/Salary Decision Support/year5_salary_training_manifest.csv",
            "year5_salary_training_manifest.csv",
        ]),
        "modeling_dictionary": find_existing_path(project_root, [
            "Output/Salary%20Decision%20Support/year5_salary_modeling_dictionary.csv",
            "Output/Salary Decision Support/year5_salary_modeling_dictionary.csv",
            "year5_salary_modeling_dictionary.csv",
        ]),
    }


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "medae": float(median_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
                ]),
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def build_model_specs(numeric_features: list[str], categorical_features: list[str]) -> dict[str, object]:
    pre = build_preprocessor(numeric_features, categorical_features)
    return {
        "global_mean": "global_mean",
        "macro_archetype_mean": "macro_archetype_mean",
        "linear_regression": Pipeline([
            ("preprocessor", clone(pre)),
            ("model", LinearRegression()),
        ]),
        "ridge_alpha_1": Pipeline([
            ("preprocessor", clone(pre)),
            ("model", Ridge(alpha=1.0)),
        ]),
        "ridge_alpha_10": Pipeline([
            ("preprocessor", clone(pre)),
            ("model", Ridge(alpha=10.0)),
        ]),
    }


def predict_custom(model_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    global_mean = float(train_df[TARGET_COL].mean())
    if model_name == "global_mean":
        pred = np.repeat(global_mean, len(test_df))
    elif model_name == "macro_archetype_mean":
        mapping = train_df.groupby("cluster_archetype_k5", dropna=False)[TARGET_COL].mean()
        pred = test_df["cluster_archetype_k5"].map(mapping).fillna(global_mean).to_numpy(dtype=float)
    else:
        raise ValueError(model_name)
    return np.clip(pred, 0.0, UPPER_CLIP)


def evaluate_models(df_train: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]):
    model_specs = build_model_specs(numeric_features, categorical_features)
    groups = df_train[GROUP_COL]
    n_splits = min(5, groups.nunique())
    splitter = GroupKFold(n_splits=n_splits)

    oof = df_train[["PLAYER_ID", "PLAYER_NAME", "draft_year", GROUP_COL, TARGET_COL]].copy().reset_index(drop=True)
    for name in model_specs:
        oof[f"pred_{name}"] = np.nan

    fold_records = []
    feature_cols = numeric_features + categorical_features

    for fold_id, (tr_idx, te_idx) in enumerate(splitter.split(df_train, df_train[TARGET_COL], groups), start=1):
        train_df = df_train.iloc[tr_idx].copy()
        test_df = df_train.iloc[te_idx].copy()
        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]
        y_train = train_df[TARGET_COL].to_numpy(dtype=float)
        y_test = test_df[TARGET_COL].to_numpy(dtype=float)

        for name, spec in model_specs.items():
            if isinstance(spec, str):
                pred = predict_custom(name, train_df, test_df)
            else:
                model = clone(spec)
                model.fit(X_train, y_train)
                pred = np.clip(model.predict(X_test).astype(float), 0.0, UPPER_CLIP)
            oof.loc[te_idx, f"pred_{name}"] = pred
            fold_records.append({
                "model_name": name,
                "fold_id": fold_id,
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "test_draft_years": ", ".join(map(str, sorted(test_df[GROUP_COL].unique()))),
                **metrics_dict(y_test, pred),
            })

    comparison = []
    for name in model_specs:
        pred_col = f"pred_{name}"
        pred = oof[pred_col].to_numpy(dtype=float)
        y = oof[TARGET_COL].to_numpy(dtype=float)
        comparison.append({
            "model_name": name,
            "n_oof": int(len(oof)),
            **metrics_dict(y, pred),
        })
    comparison_df = pd.DataFrame(comparison).sort_values(["rmse", "mae", "medae", "model_name"]).reset_index(drop=True)
    comparison_df["rank_rmse"] = np.arange(1, len(comparison_df) + 1)
    fold_df = pd.DataFrame(fold_records)
    return oof, fold_df, comparison_df


def fit_final_selected(df_train: pd.DataFrame, df_all: pd.DataFrame, selected_model: str, numeric_features: list[str], categorical_features: list[str]):
    model_specs = build_model_specs(numeric_features, categorical_features)
    spec = model_specs[selected_model]
    feature_cols = numeric_features + categorical_features

    if isinstance(spec, str):
        if selected_model == "global_mean":
            fitted = {"model_name": selected_model, "global_mean": float(df_train[TARGET_COL].mean())}
            pred_all = np.repeat(fitted["global_mean"], len(df_all))
        else:
            mapping = df_train.groupby("cluster_archetype_k5", dropna=False)[TARGET_COL].mean().to_dict()
            global_mean = float(df_train[TARGET_COL].mean())
            pred_all = df_all["cluster_archetype_k5"].map(mapping).fillna(global_mean).to_numpy(dtype=float)
            fitted = {"model_name": selected_model, "mapping": mapping, "global_mean": global_mean}
        pred_all = np.clip(pred_all, 0.0, UPPER_CLIP)
        coef_df = pd.DataFrame(columns=["feature_name", "coefficient", "abs_coefficient", "selected_model"])
        artifact_kind = "json"
    else:
        fitted = clone(spec)
        fitted.fit(df_train[feature_cols], df_train[TARGET_COL].to_numpy(dtype=float))
        pred_all = np.clip(fitted.predict(df_all[feature_cols]).astype(float), 0.0, UPPER_CLIP)
        pre = fitted.named_steps["preprocessor"]
        model = fitted.named_steps["model"]
        names = pre.get_feature_names_out()
        coef = np.ravel(model.coef_)
        coef_df = pd.DataFrame({
            "feature_name": names,
            "coefficient": coef,
            "abs_coefficient": np.abs(coef),
            "selected_model": selected_model,
        }).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
        artifact_kind = "joblib"
    return fitted, pred_all, coef_df, artifact_kind


def write_markdown(path: Path, comparison_df: pd.DataFrame, demo_df: pd.DataFrame, selected_model: str, train_n: int):
    lines = [
        "# Year-5 Salary Baseline Models",
        "",
        f"Selected model: **{selected_model}**",
        "",
        f"Training rows used: **{train_n}**",
        "",
        "## Grouped out-of-fold comparison",
        "",
        comparison_df.to_markdown(index=False),
    ]
    if not demo_df.empty:
        demo_cols = [c for c in ["PLAYER_NAME", "draft_year", "baseline_selected_model", "predicted_year5_salary_cap_pct", "holdout_reason"] if c in demo_df.columns]
        lines += ["", "## Demo holdout predictions", "", demo_df[demo_cols].to_markdown(index=False)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Phase 6 — fit baseline salary models.")
    parser.add_argument("--project-root", type=str, default=".")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_dir = get_output_dir(project_root)
    log_path = output_dir / "salary_decision_support_workflow_log.txt"

    inputs = load_inputs(project_root)
    modeling = pd.read_csv(inputs["modeling_table"])
    dictionary = pd.read_csv(inputs["modeling_dictionary"])

    rec = dictionary[dictionary["recommended_for_baseline_model"] == 1].copy()
    numeric_features = [c for c in rec.loc[rec["field_role"] == "numeric_predictor", "column_name"].tolist() if c in modeling.columns]
    categorical_features = [c for c in rec.loc[rec["field_role"] == "categorical_predictor", "column_name"].tolist() if c in modeling.columns]

    for col in categorical_features:
        modeling[col] = modeling[col].astype("object")

    train_df = modeling[(modeling[TRAIN_FLAG_COL] == 1) & (modeling[OBSERVED_FLAG_COL] == 1)].copy().reset_index(drop=True)

    oof_df, fold_df, comparison_df = evaluate_models(train_df, numeric_features, categorical_features)
    selected_model = str(comparison_df.iloc[0]["model_name"])
    fitted, pred_all, coef_df, artifact_kind = fit_final_selected(train_df, modeling, selected_model, numeric_features, categorical_features)

    pred_all_df = modeling[[
        "PLAYER_ID", "PLAYER_NAME", "draft_year", TARGET_COL, OBSERVED_FLAG_COL,
        TRAIN_FLAG_COL, HOLDOUT_COL, "holdout_reason", GROUP_COL, STATUS_COL,
    ]].copy()
    pred_all_df["baseline_selected_model"] = selected_model
    pred_all_df["predicted_year5_salary_cap_pct"] = pred_all
    pred_all_df["prediction_status"] = np.where(
        pred_all_df[TRAIN_FLAG_COL] == 1,
        "baseline_prediction_from_training_schema",
        "baseline_prediction_outside_training_pool",
    )
    demo_df = pred_all_df[pred_all_df[HOLDOUT_COL] == 1].copy().reset_index(drop=True)

    paths = {
        "oof": output_dir / "year5_salary_baseline_oof_predictions.csv",
        "fold": output_dir / "year5_salary_baseline_cv_fold_metrics.csv",
        "comparison": output_dir / "year5_salary_baseline_model_comparison.csv",
        "coefficients": output_dir / "year5_salary_baseline_selected_model_coefficients.csv",
        "predictions": output_dir / "year5_salary_baseline_selected_model_predictions.csv",
        "demo": output_dir / "year5_salary_baseline_demo_holdout_predictions.csv",
        "summary_csv": output_dir / "year5_salary_baseline_summary.csv",
        "summary_md": output_dir / "year5_salary_baseline_summary.md",
        "metadata": output_dir / "year5_salary_baseline_metadata.json",
        "artifact": output_dir / f"year5_salary_baseline_selected_model.{artifact_kind}",
    }

    oof_df.to_csv(paths["oof"], index=False)
    fold_df.to_csv(paths["fold"], index=False)
    comparison_df.to_csv(paths["comparison"], index=False)
    coef_df.to_csv(paths["coefficients"], index=False)
    pred_all_df.to_csv(paths["predictions"], index=False)
    demo_df.to_csv(paths["demo"], index=False)

    best = comparison_df.iloc[0]
    summary_df = pd.DataFrame([
        {"metric_group": "workflow_status", "metric": "training_rows", "value": int(len(train_df)), "notes": "Observed-target rows used in baseline fitting."},
        {"metric_group": "workflow_status", "metric": "draft_year_groups", "value": int(train_df[GROUP_COL].nunique()), "notes": "Unique draft-year groups in outer GroupKFold validation."},
        {"metric_group": "workflow_status", "metric": "numeric_features", "value": int(len(numeric_features)), "notes": "Recommended numeric predictors from Phase 5."},
        {"metric_group": "workflow_status", "metric": "categorical_features", "value": int(len(categorical_features)), "notes": "Recommended categorical predictors from Phase 5."},
        {"metric_group": "workflow_status", "metric": "selected_model", "value": selected_model, "notes": "Best grouped OOF model by RMSE, then MAE."},
        {"metric_group": "selected_model_oof", "metric": "rmse", "value": float(best['rmse']), "notes": "Grouped OOF RMSE for the selected baseline model."},
        {"metric_group": "selected_model_oof", "metric": "mae", "value": float(best['mae']), "notes": "Grouped OOF MAE for the selected baseline model."},
        {"metric_group": "selected_model_oof", "metric": "medae", "value": float(best['medae']), "notes": "Grouped OOF median AE for the selected baseline model."},
        {"metric_group": "selected_model_oof", "metric": "r2", "value": float(best['r2']), "notes": "Grouped OOF R^2 for the selected baseline model."},
    ])
    summary_df.to_csv(paths["summary_csv"], index=False)
    write_markdown(paths["summary_md"], comparison_df, demo_df, selected_model, len(train_df))

    if artifact_kind == "joblib":
        joblib.dump(fitted, paths["artifact"])
    else:
        paths["artifact"].write_text(json.dumps(fitted, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata = {
        "script": "06_fit_baseline_salary_models.py",
        "project_root": str(project_root),
        "inputs": {k: str(v) for k, v in inputs.items()},
        "selected_model": selected_model,
        "artifact_kind": artifact_kind,
        "outputs": [p.name for p in paths.values()],
        "notes": [
            "Outer validation uses GroupKFold with draft-year groups.",
            "Trae Young and Nikola Vučević remain outside the training pool when marked in the Phase 5 manifest.",
            "This phase fits disciplined baseline models only and does not yet integrate Deliverable 1 forecast adjustments.",
        ],
    }
    paths["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    log_text = f"""
Phase 6 complete — baseline Year-5 salary models fitted.

Inputs used:
- modeling_table: {inputs['modeling_table']}
- training_manifest: {inputs['training_manifest']}
- modeling_dictionary: {inputs['modeling_dictionary']}

Main outputs:
- {paths['oof'].name}
- {paths['fold'].name}
- {paths['comparison'].name}
- {paths['coefficients'].name}
- {paths['predictions'].name}
- {paths['demo'].name}
- {paths['summary_csv'].name}
- {paths['summary_md'].name}
- {paths['artifact'].name}

Workflow logic:
- used the Phase 5 modeling-prep table as the baseline salary-model backbone,
- restricted fitting to training-eligible observed-target rows only,
- evaluated disciplined benchmark models with draft-year-aware GroupKFold outer validation,
- selected the best baseline model by grouped out-of-fold RMSE,
- refit the selected model on the full eligible training sample and generated player-level baseline predictions,
- preserved case-study holdout players outside training while still returning their predicted salary-cap percentages.

Important boundary:
- this phase fits baseline salary models only and must not be treated as final Deliverable 3 guidance.
""".strip()
    append_workflow_log(log_path, log_text)

    print(f"Selected model: {selected_model}")
    print(comparison_df.to_string(index=False))
    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
