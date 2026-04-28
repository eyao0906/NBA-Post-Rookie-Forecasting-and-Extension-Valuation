from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from salary_workflow_utils import (
    DEMO_PLAYERS,
    append_workflow_log,
    ensure_dir,
    inventory_markdown,
    records_to_rows,
    resolve_all_expected_files,
    summarize_inventory,
    write_json,
)


SCRIPT_NAME = "00_project_inventory.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the Deliverable 3 project file map, resolve expected inputs from the "
            "project root, and write an auditable inventory for downstream salary workflow phases."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root containing src/, data/, Output/, and visual/ directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override for Output/Salary Decision Support.",
    )
    parser.add_argument(
        "--visual-dir",
        type=Path,
        default=None,
        help="Optional override for visual/Salary Decision Support.",
    )
    parser.add_argument(
        "--log-name",
        type=str,
        default="salary_decision_support_workflow_log.txt",
        help="Workflow log file name written under the output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = (args.output_dir.resolve() if args.output_dir else project_root / "Output" / "Salary Decision Support")
    visual_dir = (args.visual_dir.resolve() if args.visual_dir else project_root / "visual" / "Salary Decision Support")
    ensure_dir(output_dir)
    ensure_dir(visual_dir)

    inventory_records = resolve_all_expected_files(project_root)
    summary = summarize_inventory(inventory_records)

    inventory_csv_path = output_dir / "project_inventory.csv"
    inventory_md_path = output_dir / "project_inventory.md"
    inventory_json_path = output_dir / "project_inventory_summary.json"
    workflow_log_path = output_dir / args.log_name

    inventory_df = pd.DataFrame(records_to_rows(inventory_records))
    inventory_df.sort_values(["phase_group", "required", "role"], ascending=[True, False, True], inplace=True)
    inventory_df.to_csv(inventory_csv_path, index=False)

    inventory_md_path.write_text(
        inventory_markdown(inventory_records, project_root),
        encoding="utf-8",
    )

    write_json(
        inventory_json_path,
        {
            "script": SCRIPT_NAME,
            "project_root": str(project_root),
            "output_dir": str(output_dir),
            "visual_dir": str(visual_dir),
            "demo_players_reserved_for_case_study": DEMO_PLAYERS,
            **summary,
        },
    )

    log_text = f"""
Phase 0 — path audit and project map
Script: {SCRIPT_NAME}
Project root: {project_root}
Output directory: {output_dir}
Visual directory: {visual_dir}

Purpose:
- resolve the canonical Deliverable 3 file map from project-root relative paths,
- search recursively when files are not found at the expected location,
- write an auditable inventory before downstream salary modeling or decision logic starts,
- keep Trae Young and Nikola Vučević reserved as case-study demo players for later phases.

Outputs written:
- {inventory_csv_path.name}
- {inventory_md_path.name}
- {inventory_json_path.name}

Summary:
- required files found: {summary['required_found']} / {summary['required_total']}
- optional files found: {summary['optional_found']} / {summary['optional_total']}
- path-corrected resolutions: {summary['corrected_paths']}
- missing required roles: {', '.join(summary['missing_required_roles']) if summary['missing_required_roles'] else 'none'}
""".strip()
    append_workflow_log(workflow_log_path, log_text)

    print("Inventory complete.")
    print(f"Project root: {project_root}")
    print(f"Required files found: {summary['required_found']} / {summary['required_total']}")
    print(f"Optional files found: {summary['optional_found']} / {summary['optional_total']}")
    print(f"Inventory CSV: {inventory_csv_path}")
    print(f"Workflow log: {workflow_log_path}")


if __name__ == "__main__":
    main()
