from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from salary_workflow_utils import append_workflow_log, ensure_dir, write_json

SCRIPT_NAME = '05_prepare_salary_model_table.py'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Prepare a baseline-ready Year-5 salary modeling table by merging the canonical '
            'Year-5 salary target with reusable Deliverable 2 predictor backbones. This phase '
            'is intentionally leakage-aware and keeps Trae Young / Nikola Vučević out of '
            'future training calibration via the holdout manifest.'
        )
    )
    parser.add_argument('--project-root', type=Path, default=Path('.'), help='Project root containing Output/, src/, and visual/.')
    parser.add_argument('--output-dir', type=Path, default=None, help='Optional override for Output/Salary Decision Support.')
    parser.add_argument('--inventory-csv', type=Path, default=None, help='Optional inventory CSV from Phase 0.')
    parser.add_argument('--holdout-csv', type=Path, default=None, help='Optional holdout manifest from Phase 1.')
    parser.add_argument('--log-name', type=str, default='salary_decision_support_workflow_log.txt', help='Workflow log file name written under the output directory.')
    return parser.parse_args()


def load_inventory(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


INVENTORY_ROLE_TO_BASENAME = {
    'year5_salary_target': 'year5_salary_target_table.csv',
    'identity_drift_table': 'player_identity_drift_table.csv',
    'macro_archetype_backbone': 'archetype_macro_player_table.csv',
    'final_player_profile': 'final_player_archetype_profile_table.csv',
}


def resolve_from_inventory(inventory_df: pd.DataFrame | None, role: str) -> Path | None:
    if inventory_df is None or 'role' not in inventory_df.columns or 'resolved_path' not in inventory_df.columns:
        return None
    subset = inventory_df.loc[inventory_df['role'] == role, 'resolved_path']
    if subset.empty:
        return None
    candidate = Path(str(subset.iloc[0]))
    return candidate if candidate.exists() else None


def resolve_input(project_root: Path, inventory_df: pd.DataFrame | None, role: str, explicit_output_dir: Path | None = None) -> Path:
    inv_path = resolve_from_inventory(inventory_df, role)
    if inv_path is not None:
        return inv_path.resolve()

    basename = INVENTORY_ROLE_TO_BASENAME[role]
    candidates: list[Path] = []
    if explicit_output_dir is not None:
        candidates.extend([
            explicit_output_dir / basename,
            explicit_output_dir / '..' / basename,
        ])
    candidates.extend([
        project_root / 'Output' / 'Player Archetype Analysis' / basename,
        project_root / 'Output' / 'Player Archetype Analysis' / 'SalaryBlock' / basename,
        project_root / basename,
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    matches = list(project_root.rglob(basename))
    if matches:
        matches.sort(key=lambda p: (len(p.parts), str(p).lower()))
        return matches[0].resolve()
    raise FileNotFoundError(f'Could not resolve required file for role={role} ({basename}).')


HOLDOUT_BASENAME = 'salary_demo_player_holdout_manifest.csv'


def resolve_holdout(project_root: Path, explicit_path: Path | None, output_dir: Path) -> Path | None:
    if explicit_path is not None and explicit_path.exists():
        return explicit_path.resolve()
    candidates = [
        output_dir / HOLDOUT_BASENAME,
        project_root / 'Output' / 'Salary Decision Support' / HOLDOUT_BASENAME,
        project_root / HOLDOUT_BASENAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    matches = list(project_root.rglob(HOLDOUT_BASENAME))
    if matches:
        matches.sort(key=lambda p: (len(p.parts), str(p).lower()))
        return matches[0].resolve()
    return None


def require_columns(df: pd.DataFrame, required: list[str], table_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f'{table_name} is missing required columns: {missing}')


TARGET_AUDIT_COLUMNS = {
    'year5_season_start', 'year5_season_string', 'salary_player_name_example', 'year_salary_total', 'salary_cap', 'luxury_tax',
    'raw_salary_rows', 'raw_team_count', 'any_team_option', 'any_player_option', 'any_qualifying_offer', 'any_two_way_contract',
    'any_terminated', 'salary_name_match_flag', 'cap_match_flag', 'year5_salary_match_flag', 'salary_match_type', 'name_key_raw', 'name_key'
}


IDENTIFIER_COLUMNS = {'PLAYER_ID', 'PLAYER_NAME', 'draft_year'}
TARGET_COLUMNS = {'year5_salary_cap_pct'}


def classify_field_role(column: str) -> tuple[str, int, str]:
    if column in IDENTIFIER_COLUMNS:
        return 'identifier', 0, 'Player identifier only; not a modeling feature.'
    if column in TARGET_COLUMNS:
        return 'target', 0, 'Canonical Year-5 salary-cap percentage target.'
    if column in {'holdout_for_case_study', 'holdout_reason', 'training_eligible_flag', 'target_observed_flag', 'split_group_draft_year', 'modeling_row_status'}:
        return 'training_meta', 0, 'Training / split control metadata; not a predictive feature.'
    if column in TARGET_AUDIT_COLUMNS:
        return 'target_audit_context', 0, 'Built from the Year-5 salary merge; keep for audit, exclude from predictive modeling.'
    if column.startswith('cluster_') and 'onehot' in column:
        return 'numeric_predictor', 1, 'Saved macro-role one-hot indicator from Deliverable 2.'
    if column.startswith('dist_to_cluster_'):
        return 'numeric_predictor', 1, 'Distance-to-centroid predictor from saved macro-role backbone.'
    if column.startswith('pca_'):
        return 'numeric_predictor', 1, 'Saved PCA score from Deliverable 2 macro-role backbone.'
    if column.startswith('shot_emb_'):
        return 'numeric_predictor', 1, 'Shot-style embedding dimension from Deliverable 2.'
    if column.startswith('shot_style_prob_'):
        return 'numeric_predictor', 1, 'Soft shot-style subtype membership probability.'
    if column in {'own_cluster_distance', 'prototype_fit_rank_distance', 'prototype_margin_second_minus_first', 'prototype_ambiguity_ratio', 'prototype_fit_ambiguity'}:
        return 'numeric_predictor', 1, 'Prototype-fit / cluster-fit predictor available before Year 5.'
    if column in {
        'box_role_y1_to_y4_displacement', 'box_role_total_path_length', 'observed_boxscore_seasons',
        'shotstyle_y1_to_y4_displacement', 'shotstyle_total_path_length', 'observed_shot_seasons',
        'shot_style_subtype_changes', 'seasons_with_embeddings', 'total_shot_attempts_covered', 'shot_style_entropy',
        'shot_style_subtype_probability'
    }:
        return 'numeric_predictor', 1, 'Early-career drift / shot-coverage / subtype-confidence predictor.'
    if column in {'macro_cluster_id', 'macro_archetype', 'cluster_archetype_k5', 'shot_style_subtype_id', 'shot_style_subtype', 'hybrid_archetype_label', 'identity_drift_class', 'embedding_aggregation'}:
        return 'categorical_predictor', 1, 'Pre-Year5 categorical predictor; encode later if used in the baseline model.'
    if column in {'DRAFT_YEAR'}:
        return 'redundant_meta', 0, 'Redundant draft-year field from upstream table; exclude from modeling.'
    return 'context_or_other', 0, 'Unclassified context field; review before using in a predictive model.'


NUMERIC_CONTEXT_DEFAULTS = {
    'training_eligible_flag', 'target_observed_flag', 'holdout_for_case_study', 'split_group_draft_year'
}


def build_dictionary(df: pd.DataFrame, source_map: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in df.columns:
        field_role, recommended, note = classify_field_role(column)
        rows.append({
            'column_name': column,
            'dtype': str(df[column].dtype),
            'source_table': source_map.get(column, 'derived_or_unknown'),
            'field_role': field_role,
            'recommended_for_baseline_model': int(recommended),
            'notes': note,
        })
    dict_df = pd.DataFrame(rows)
    role_order = {
        'identifier': 0,
        'target': 1,
        'training_meta': 2,
        'numeric_predictor': 3,
        'categorical_predictor': 4,
        'target_audit_context': 5,
        'redundant_meta': 6,
        'context_or_other': 7,
    }
    dict_df['_order'] = dict_df['field_role'].map(role_order).fillna(99)
    dict_df = dict_df.sort_values(['_order', 'column_name']).drop(columns=['_order']).reset_index(drop=True)
    return dict_df


def build_missingness(df: pd.DataFrame, dict_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    role_lookup = dict_df.set_index('column_name')[['field_role', 'recommended_for_baseline_model', 'source_table']].to_dict('index')
    n_rows = len(df)
    for column in df.columns:
        missing_count = int(df[column].isna().sum())
        info = role_lookup.get(column, {})
        rows.append({
            'column_name': column,
            'missing_count': missing_count,
            'missing_pct': float(missing_count / n_rows) if n_rows else np.nan,
            'non_missing_count': int(n_rows - missing_count),
            'field_role': info.get('field_role', 'unknown'),
            'recommended_for_baseline_model': info.get('recommended_for_baseline_model', 0),
            'source_table': info.get('source_table', 'unknown'),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(['recommended_for_baseline_model', 'missing_pct', 'column_name'], ascending=[False, False, True]).reset_index(drop=True)


def markdown_summary(summary_df: pd.DataFrame, note_lines: list[str]) -> str:
    lines = [
        '# Year-5 Salary Modeling Table Summary',
        '',
        'This phase prepares a leakage-aware baseline modeling table for Year-5 salary-cap regression.',
        'It uses the canonical target plus reusable Deliverable 2 predictors already available in the project bundle.',
        '',
        '## Summary metrics',
        '',
        '| Metric group | Metric | Value | Notes |',
        '|---|---|---:|---|',
    ]
    for _, row in summary_df.iterrows():
        value = row['value']
        if isinstance(value, float):
            if float(value).is_integer():
                value_str = str(int(value))
            else:
                value_str = f'{value:.6f}'
        else:
            value_str = str(value)
        lines.append(f"| {row['metric_group']} | {row['metric']} | {value_str} | {row['notes']} |")
    lines.extend(['', '## Notes', ''])
    lines.extend([f'- {line}' for line in note_lines])
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else project_root / 'Output' / 'Salary Decision Support'
    ensure_dir(output_dir)
    workflow_log_path = output_dir / args.log_name

    inventory_df = load_inventory(args.inventory_csv.resolve() if args.inventory_csv is not None else None)

    year5_target_path = resolve_input(project_root, inventory_df, 'year5_salary_target', output_dir)
    identity_drift_path = resolve_input(project_root, inventory_df, 'identity_drift_table', output_dir)
    macro_backbone_path = resolve_input(project_root, inventory_df, 'macro_archetype_backbone', output_dir)
    final_profile_path = resolve_input(project_root, inventory_df, 'final_player_profile', output_dir)
    holdout_path = resolve_holdout(project_root, args.holdout_csv.resolve() if args.holdout_csv is not None else None, output_dir)

    target_df = pd.read_csv(year5_target_path)
    drift_df = pd.read_csv(identity_drift_path)
    macro_df = pd.read_csv(macro_backbone_path)
    profile_df = pd.read_csv(final_profile_path)
    holdout_df = pd.read_csv(holdout_path) if holdout_path is not None and holdout_path.exists() else pd.DataFrame(columns=['PLAYER_ID', 'holdout_for_case_study', 'holdout_reason'])

    require_columns(target_df, ['PLAYER_ID', 'PLAYER_NAME', 'draft_year', 'year5_salary_cap_pct', 'year5_salary_match_flag'], 'year5_salary_target_table.csv')
    require_columns(drift_df, ['PLAYER_ID', 'PLAYER_NAME', 'draft_year', 'macro_archetype', 'identity_drift_class'], 'player_identity_drift_table.csv')
    require_columns(holdout_df, ['PLAYER_ID'], 'salary_demo_player_holdout_manifest.csv')

    holdout_cols = ['PLAYER_ID']
    for col in ['holdout_for_case_study', 'holdout_reason']:
        if col in holdout_df.columns:
            holdout_cols.append(col)
    holdout_use = holdout_df[holdout_cols].drop_duplicates('PLAYER_ID')

    # Start from canonical target backbone.
    modeling_df = target_df.copy()

    # Merge the richest reusable Deliverable 2 predictor backbone available now.
    drift_add = drift_df.drop(columns=[c for c in ['PLAYER_NAME', 'draft_year', 'macro_archetype'] if c in drift_df.columns])
    modeling_df = modeling_df.merge(drift_add, on='PLAYER_ID', how='left', validate='one_to_one')

    # Pull compact human-readable reference fields only when they are missing from drift backbone.
    profile_keep = [
        c for c in ['PLAYER_ID', 'supporting_shot_style_explanation', 'comp_based_interpretation']
        if c in profile_df.columns
    ]
    if profile_keep:
        modeling_df = modeling_df.merge(profile_df[profile_keep].drop_duplicates('PLAYER_ID'), on='PLAYER_ID', how='left', validate='one_to_one')

    # Macro backbone is used only for sanity checks and to backfill if a saved column is unexpectedly absent.
    if 'prototype_ambiguity_ratio' not in modeling_df.columns and 'prototype_ambiguity_ratio' in macro_df.columns:
        macro_fill = macro_df[['PLAYER_ID', 'prototype_ambiguity_ratio']].drop_duplicates('PLAYER_ID')
        modeling_df = modeling_df.merge(macro_fill, on='PLAYER_ID', how='left', validate='one_to_one')

    modeling_df = modeling_df.merge(holdout_use, on='PLAYER_ID', how='left')
    if 'holdout_for_case_study' not in modeling_df.columns:
        modeling_df['holdout_for_case_study'] = 0
    modeling_df['holdout_for_case_study'] = modeling_df['holdout_for_case_study'].fillna(0).astype(int)
    if 'holdout_reason' not in modeling_df.columns:
        modeling_df['holdout_reason'] = np.nan

    modeling_df['target_observed_flag'] = modeling_df['year5_salary_match_flag'].fillna(0).astype(int)
    modeling_df['training_eligible_flag'] = ((modeling_df['target_observed_flag'] == 1) & (modeling_df['holdout_for_case_study'] == 0)).astype(int)
    modeling_df['split_group_draft_year'] = modeling_df['draft_year']

    def row_status(row: pd.Series) -> str:
        if int(row['target_observed_flag']) == 0:
            return 'target_missing_not_trainable'
        if int(row['holdout_for_case_study']) == 1:
            return 'demo_holdout_excluded_from_training'
        return 'trainable_target_observed'

    modeling_df['modeling_row_status'] = modeling_df.apply(row_status, axis=1)

    # Prefer a clean column order: identifiers / target / training meta / predictors / audit/context.
    front = [
        'PLAYER_ID', 'PLAYER_NAME', 'draft_year', 'year5_salary_cap_pct', 'target_observed_flag', 'training_eligible_flag',
        'holdout_for_case_study', 'holdout_reason', 'split_group_draft_year', 'modeling_row_status'
    ]
    remaining = [c for c in modeling_df.columns if c not in front]
    modeling_df = modeling_df[front + remaining]

    # Build source map for data dictionary.
    source_map: dict[str, str] = {}
    for col in target_df.columns:
        source_map[col] = 'year5_salary_target_table.csv'
    for col in drift_df.columns:
        source_map.setdefault(col, 'player_identity_drift_table.csv')
    for col in profile_keep:
        if col != 'PLAYER_ID':
            source_map.setdefault(col, 'final_player_archetype_profile_table.csv')
    for col in ['holdout_for_case_study', 'holdout_reason']:
        if col in modeling_df.columns:
            source_map[col] = 'salary_demo_player_holdout_manifest.csv'
    for col in ['target_observed_flag', 'training_eligible_flag', 'split_group_draft_year', 'modeling_row_status']:
        source_map[col] = 'derived_phase5'

    dict_df = build_dictionary(modeling_df, source_map)
    missingness_df = build_missingness(modeling_df, dict_df)

    train_manifest = modeling_df[[
        'PLAYER_ID', 'PLAYER_NAME', 'draft_year', 'year5_salary_cap_pct', 'target_observed_flag',
        'holdout_for_case_study', 'holdout_reason', 'training_eligible_flag', 'split_group_draft_year', 'modeling_row_status'
    ]].copy()

    summary_rows = [
        {'metric_group': 'workflow_status', 'metric': 'players_total', 'value': int(len(modeling_df)), 'notes': 'Total player rows carried into the modeling-prep table.'},
        {'metric_group': 'workflow_status', 'metric': 'columns_total', 'value': int(modeling_df.shape[1]), 'notes': 'Total columns in the modeling-prep table.'},
        {'metric_group': 'target_status', 'metric': 'target_observed_rows', 'value': int(modeling_df['target_observed_flag'].sum()), 'notes': 'Rows with observed Year-5 salary-cap target available.'},
        {'metric_group': 'target_status', 'metric': 'target_missing_rows', 'value': int((modeling_df['target_observed_flag'] == 0).sum()), 'notes': 'Rows without an observed Year-5 salary-cap target.'},
        {'metric_group': 'training_status', 'metric': 'holdout_rows', 'value': int(modeling_df['holdout_for_case_study'].sum()), 'notes': 'Reserved demo rows excluded from model fitting.'},
        {'metric_group': 'training_status', 'metric': 'training_eligible_rows', 'value': int(modeling_df['training_eligible_flag'].sum()), 'notes': 'Rows eligible for baseline salary-model training after excluding demo holdouts.'},
        {'metric_group': 'feature_inventory', 'metric': 'recommended_numeric_predictors', 'value': int(((dict_df['field_role'] == 'numeric_predictor') & (dict_df['recommended_for_baseline_model'] == 1)).sum()), 'notes': 'Numeric predictors available before Year 5.'},
        {'metric_group': 'feature_inventory', 'metric': 'recommended_categorical_predictors', 'value': int(((dict_df['field_role'] == 'categorical_predictor') & (dict_df['recommended_for_baseline_model'] == 1)).sum()), 'notes': 'Categorical predictors available before Year 5 that will need encoding later.'},
        {'metric_group': 'feature_inventory', 'metric': 'max_missing_pct_recommended_predictors', 'value': float(missingness_df.loc[missingness_df['recommended_for_baseline_model'] == 1, 'missing_pct'].max()), 'notes': 'Maximum missingness across recommended predictor columns.'},
    ]
    summary_df = pd.DataFrame(summary_rows)

    note_lines = [
        'This phase uses the richest reusable Deliverable 2 predictor backbone currently available in the uploaded project bundle: player_identity_drift_table.csv.',
        'No comp-salary anchors or provisional action buckets are used as model predictors here, because those would contaminate the baseline salary-estimation module with target-adjacent context.',
        'Trae Young and Nikola Vučević remain reserved as case-study demos and are explicitly excluded from training via training_eligible_flag = 0.',
        'The original flattened Seasons 1–4 player-feature table is not part of the current uploaded Deliverable 3 file bundle, so this phase reuses the saved macro-role / shot-style / drift backbone instead.'
    ]

    table_path = output_dir / 'year5_salary_modeling_table.csv'
    dictionary_path = output_dir / 'year5_salary_modeling_dictionary.csv'
    missingness_path = output_dir / 'year5_salary_modeling_missingness.csv'
    manifest_path = output_dir / 'year5_salary_training_manifest.csv'
    summary_path = output_dir / 'year5_salary_modeling_summary.csv'
    summary_md_path = output_dir / 'year5_salary_modeling_summary.md'
    metadata_path = output_dir / 'year5_salary_modeling_metadata.json'

    modeling_df.to_csv(table_path, index=False)
    dict_df.to_csv(dictionary_path, index=False)
    missingness_df.to_csv(missingness_path, index=False)
    train_manifest.to_csv(manifest_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    summary_md_path.write_text(markdown_summary(summary_df, note_lines), encoding='utf-8')

    metadata = {
        'script': SCRIPT_NAME,
        'project_root': str(project_root),
        'inputs': {
            'year5_salary_target': str(year5_target_path),
            'identity_drift_table': str(identity_drift_path),
            'macro_archetype_backbone': str(macro_backbone_path),
            'final_player_profile': str(final_profile_path),
            'holdout_manifest': str(holdout_path) if holdout_path is not None else None,
        },
        'outputs': [
            'year5_salary_modeling_table.csv',
            'year5_salary_modeling_dictionary.csv',
            'year5_salary_modeling_missingness.csv',
            'year5_salary_training_manifest.csv',
            'year5_salary_modeling_summary.csv',
            'year5_salary_modeling_summary.md',
        ],
        'notes': note_lines,
    }
    write_json(metadata_path, metadata)

    log_text = f"""
Phase 5 complete — Year-5 salary modeling table prepared.

Inputs used:
- year5_salary_target: {year5_target_path}
- identity_drift_table: {identity_drift_path}
- macro_archetype_backbone: {macro_backbone_path}
- final_player_profile: {final_profile_path}
- holdout_manifest: {holdout_path if holdout_path is not None else 'not found'}

Main outputs:
- year5_salary_modeling_table.csv
- year5_salary_modeling_dictionary.csv
- year5_salary_modeling_missingness.csv
- year5_salary_training_manifest.csv
- year5_salary_modeling_summary.csv
- year5_salary_modeling_summary.md

Workflow logic:
- treated year5_salary_target_table.csv as the canonical target backbone,
- reused player_identity_drift_table.csv as the richest currently available pre-Year5 predictor backbone,
- kept salary-merge audit fields for traceability but marked them as excluded from baseline predictive modeling,
- kept Trae Young and Nikola Vučević excluded from training via the holdout manifest,
- documented the feature dictionary and missingness so a later baseline regression can be built without silent leakage.

Important boundary:
- this phase prepares the baseline salary-model table only;
- it does not fit the model yet and does not replace the protected / fair / walk-away market band logic.
"""
    append_workflow_log(workflow_log_path, log_text)


if __name__ == '__main__':
    main()
