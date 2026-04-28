from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_output_dir(project_root: Path) -> Path:
    candidates = [
        project_root / 'Output/Salary%20Decision%20Support',
        project_root / 'Output/Salary Decision Support',
    ]
    for c in candidates:
        if c.exists():
            return ensure_dir(c)
    return ensure_dir(candidates[0])


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
    raise FileNotFoundError(f'Could not locate any of: {candidates}')


def append_workflow_log(log_path: Path, text_block: str) -> None:
    ensure_dir(log_path.parent)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with log_path.open('a', encoding='utf-8') as fh:
        fh.write(f"\n{'=' * 80}\n")
        fh.write(f'[{timestamp}]\n')
        fh.write(text_block.rstrip() + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description='Write Deliverable 3 completion note')
    parser.add_argument('--project-root', default='.')
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_dir = get_output_dir(project_root)

    summary_path = find_existing_path(project_root, [
        'Output/Salary Decision Support/deliverable3_staged_final_summary.csv',
        'Output/Salary%20Decision%20Support/deliverable3_staged_final_summary.csv',
        'deliverable3_staged_final_summary.csv',
    ])
    subset_summary_path = find_existing_path(project_root, [
        'Output/Salary Decision Support/deliverable3_forecast_adjustment_supported_subset_summary.csv',
        'Output/Salary%20Decision%20Support/deliverable3_forecast_adjustment_supported_subset_summary.csv',
        'deliverable3_forecast_adjustment_supported_subset_summary.csv',
    ])
    case_path = find_existing_path(project_root, [
        'Output/Salary Decision Support/deliverable3_final_case_study_table.csv',
        'Output/Salary%20Decision%20Support/deliverable3_final_case_study_table.csv',
        'deliverable3_final_case_study_table.csv',
    ])

    summary = pd.read_csv(summary_path)
    subset = pd.read_csv(subset_summary_path)
    case_df = pd.read_csv(case_path)

    metric_map = dict(zip(summary['metric'], summary['value']))
    subset_map = dict(zip(subset['metric'], subset['value']))

    lines = [
        '# Deliverable 3 Framework Completion Note',
        '',
        'Deliverable 3 is now completed as a **staged salary-decision-support framework** rather than as a single overclaimed final recommendation engine.',
        '',
        '## Complete now',
        '- Year-5 salary-cap target backbone is built and audited.',
        '- Protected / fair / walk-away comp-market anchor band is built and audited.',
        '- Comp-support, scarcity, and Block-2-only provisional action logic are built for the full cohort.',
        '- Baseline salary model and market-band reconciliation are built.',
        '- Official Block 1 forecast handoff is integrated into the Deliverable 3 framework.',
        '- Forecast-adjusted logic is activated on the supported subset only.',
        '- Final stakeholder-facing case-study rows for Trae Young and Nikola Vučević are assembled.',
        '',
        '## Still conditional / not fully validated',
        '- Full-cohort final extension guidance is still not validated.',
        '- Durability / availability risk remains a pending separate module.',
        '- The current uncertainty input is the Block 1 confidence proxy, not a richer interval package.',
        '- Forecast-adjusted stance logic is only defensible on the supported subset, not the whole cohort.',
        '',
        '## Core status counts',
        '',
        f"- Framework rows: {int(metric_map.get('framework_rows', 0))}",
        f"- Rows with market-anchor support: {int(metric_map.get('market_anchor_supported_rows', 0))}",
        f"- Rows with official Block 1 probabilities merged: {int(metric_map.get('forecast_probabilities_available_rows', 0))}",
        f"- Rows eligible for supported forecast-adjusted logic: {int(metric_map.get('supported_forecast_adjustment_rows', 0))}",
        f"- Salary-model training rows after case-study holdout exclusion: {int(metric_map.get('salary_model_training_sample_rows', 0))}",
        '',
        '## Supported forecast-adjustment evaluation subset',
        '',
        f"- Supported rows: {int(subset_map.get('supported_subset_rows', 0))}",
        f"- Supported rows with observed salary target: {int(subset_map.get('supported_subset_with_salary_target', 0))}",
        f"- Label accuracy on supported subset: {subset_map.get('supported_subset_label_accuracy', 'NA')}",
        f"- Performance RMSE on supported subset: {subset_map.get('supported_subset_perf_rmse', 'NA')}",
        f"- Performance MAE on supported subset: {subset_map.get('supported_subset_perf_mae', 'NA')}",
        '',
        '## Case-study status',
        '',
    ]
    for _, row in case_df.iterrows():
        lines += [
            f"### {row['PLAYER_NAME']}",
            f"- Staged stance: {row['staged_extension_stance_label']}",
            f"- Market band: protected {row['protected_price_cap_pct']:.4f}, fair {row['fair_price_cap_pct']:.4f}, walk-away {row['walk_away_max_cap_pct']:.4f}",
            f"- Negotiation posture: open {row['staged_open_cap_pct']:.4f}, target {row['staged_target_cap_pct']:.4f}, hard max {row['staged_hard_max_cap_pct']:.4f}",
            f"- Forecast support: {row['forecast_support_status']}",
            f"- Note: {row['block1_support_note']}",
            '',
        ]

    note_path = output_dir / 'deliverable3_framework_completion_note.md'
    note_path.write_text('\n'.join(lines), encoding='utf-8')

    metadata = {
        'script': '10_write_framework_completion_note.py',
        'project_root': str(project_root),
        'inputs': {
            'staged_summary': str(summary_path),
            'supported_subset_summary': str(subset_summary_path),
            'final_case_study_table': str(case_path),
        },
        'outputs': [note_path.name],
    }
    (output_dir / 'deliverable3_framework_completion_note_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    append_workflow_log(
        output_dir / 'salary_decision_support_workflow_log.txt',
        f'''
Phase 10 complete — Deliverable 3 framework completion note written.

Inputs used:
- staged_summary: {summary_path}
- supported_subset_summary: {subset_summary_path}
- final_case_study_table: {case_path}

Main outputs:
- {note_path.name}
- deliverable3_framework_completion_note_metadata.json

Purpose:
- document what Deliverable 3 is honestly complete now,
- state what remains conditional,
- and keep the staged system reviewable for report writing and future continuation.
        '''
    )


if __name__ == '__main__':
    main()
