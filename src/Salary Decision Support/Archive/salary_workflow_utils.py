from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence
import csv
import json
import os
from datetime import datetime


DEMO_PLAYERS = [
    "Trae Young",
    "Nikola Vucevic",
    "Nikola Vucevic",
    "Nikola Vučević",
]


@dataclass
class ExpectedFile:
    phase_group: str
    role: str
    relative_path: str
    required: bool = True
    kind: str = "csv"
    notes: str = ""


@dataclass
class ResolvedFile:
    phase_group: str
    role: str
    expected_path: str
    resolved_path: str | None
    exists: bool
    path_corrected: bool
    correction_reason: str
    required: bool
    kind: str
    notes: str
    file_size_bytes: int | None = None
    modified_time: str | None = None
    row_count: int | None = None
    column_count: int | None = None
    columns_preview: str | None = None


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_expected_files() -> list[ExpectedFile]:
    """Canonical Deliverable 3 file map, using project-root relative paths."""
    return [
        ExpectedFile(
            phase_group="deliverable2_backbone",
            role="macro_archetype_backbone",
            relative_path="Output/Player Archetype Analysis/archetype_macro_player_table.csv",
            notes="Saved PCA + K-means macro-role backbone from Deliverable 2.",
        ),
        ExpectedFile(
            phase_group="deliverable2_backbone",
            role="hybrid_archetype_table",
            relative_path="Output/Player Archetype Analysis/player_hybrid_archetype_table.csv",
            notes="Merged macro role + shot-style subtype table.",
        ),
        ExpectedFile(
            phase_group="deliverable2_backbone",
            role="identity_drift_table",
            relative_path="Output/Player Archetype Analysis/player_identity_drift_table.csv",
            notes="Player-level drift outputs used as Block 2 risk proxies.",
        ),
        ExpectedFile(
            phase_group="deliverable2_backbone",
            role="realistic_comp_engine",
            relative_path="Output/Player Archetype Analysis/realistic_comps.csv",
            notes="Row-level realistic comparable-player table.",
        ),
        ExpectedFile(
            phase_group="deliverable2_backbone",
            role="ceiling_comp_engine",
            relative_path="Output/Player Archetype Analysis/ceiling_comps_hof.csv",
            required=False,
            notes="Optional ceiling-comp table; not a primary pricing anchor.",
        ),
        ExpectedFile(
            phase_group="deliverable2_backbone",
            role="final_player_profile",
            relative_path="Output/Player Archetype Analysis/final_player_archetype_profile_table.csv",
            notes="Main stakeholder-facing Deliverable 2 table feeding Deliverable 3.",
        ),
        ExpectedFile(
            phase_group="salary_block",
            role="year5_salary_target",
            relative_path="Output/Player Archetype Analysis/SalaryBlock/year5_salary_target_table.csv",
            notes="Canonical Year-5 salary-cap target table.",
        ),
        ExpectedFile(
            phase_group="salary_block",
            role="year5_salary_merge_summary",
            relative_path="Output/Player Archetype Analysis/SalaryBlock/year5_salary_merge_summary.csv",
            notes="Coverage summary for the salary merge.",
        ),
        ExpectedFile(
            phase_group="salary_block",
            role="year5_salary_unmatched_diagnostic",
            relative_path="Output/Player Archetype Analysis/SalaryBlock/year5_salary_unmatched_diagnostic.csv",
            notes="Diagnostic table for unmatched Year-5 salary rows.",
        ),
        ExpectedFile(
            phase_group="salary_block",
            role="comp_salary_detail",
            relative_path="Output/Player Archetype Analysis/SalaryBlock/comp_salary_detail_table.csv",
            notes="Row-level comp salary audit file.",
        ),
        ExpectedFile(
            phase_group="salary_block",
            role="deliverable3_block2_context",
            relative_path="Output/Player Archetype Analysis/SalaryBlock/deliverable3_block2_archetype_comp_market_context.csv",
            notes="Main current-state handoff table for Deliverable 3.",
        ),
        ExpectedFile(
            phase_group="dossier_comm_layer",
            role="report_ready_dossier_table",
            relative_path="Output/Player Archetype Analysis/player_dossier_demo/report_ready_player_dossier_table.csv",
            required=False,
            notes="Communication-layer table for case-study demos.",
        ),
        ExpectedFile(
            phase_group="dossier_comm_layer",
            role="report_ready_dossier_selection",
            relative_path="Output/Player Archetype Analysis/player_dossier_demo/report_ready_player_dossier_selection.csv",
            required=False,
            notes="Case-study subset selection table.",
        ),
        ExpectedFile(
            phase_group="dossier_comm_layer",
            role="report_ready_dossier_manifest",
            relative_path="Output/Player Archetype Analysis/player_dossier_demo/report_ready_player_dossier_manifest.csv",
            required=False,
            notes="Manifest for dossier build artifacts.",
        ),
        ExpectedFile(
            phase_group="dossier_comm_layer",
            role="report_ready_dossiers_markdown",
            relative_path="Output/Player Archetype Analysis/player_dossier_demo/report_ready_player_dossiers.md",
            required=False,
            kind="markdown",
            notes="Markdown communication export from Deliverable 2.",
        ),
    ]


def best_recursive_match(project_root: Path, expected_relative_path: str) -> tuple[Path | None, str]:
    expected_parts = Path(expected_relative_path).parts
    basename = expected_parts[-1]
    matches = list(project_root.rglob(basename))
    if not matches:
        return None, "missing"

    def score(path: Path) -> tuple[int, int, int, str]:
        path_parts = path.parts
        suffix_matches = 0
        for i in range(1, min(len(path_parts), len(expected_parts)) + 1):
            if path_parts[-i].lower() == expected_parts[-i].lower():
                suffix_matches += 1
            else:
                break
        phase_bonus = 1 if "Salary Decision Support" in str(path) else 0
        return (-suffix_matches, -phase_bonus, len(path.parts), str(path).lower())

    matches.sort(key=score)
    best = matches[0]

    expected_name = str(Path(expected_relative_path)).replace("\\", "/")
    resolved_name = str(best.relative_to(project_root)).replace("\\", "/")
    if resolved_name == expected_name:
        return best, "exact"
    return best, "recursive_match"


def inspect_tabular_file(path: Path) -> tuple[int | None, int | None, str | None]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if header is None:
                return 0, 0, None
            row_count = 0
            for _ in reader:
                row_count += 1
            column_count = len(header)
            preview = ", ".join(header[:12])
            return row_count, column_count, preview
    except UnicodeDecodeError:
        with path.open("r", encoding="latin-1", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if header is None:
                return 0, 0, None
            row_count = 0
            for _ in reader:
                row_count += 1
            column_count = len(header)
            preview = ", ".join(header[:12])
            return row_count, column_count, preview
    except Exception:
        return None, None, None


def resolve_expected_file(project_root: Path, spec: ExpectedFile) -> ResolvedFile:
    expected_path = project_root / spec.relative_path
    correction_reason = "exact"
    resolved_path: Path | None = None
    path_corrected = False

    if expected_path.exists():
        resolved_path = expected_path
    else:
        resolved_path, correction_reason = best_recursive_match(project_root, spec.relative_path)
        path_corrected = resolved_path is not None

    exists = resolved_path is not None and resolved_path.exists()
    file_size_bytes = None
    modified_time = None
    row_count = None
    column_count = None
    columns_preview = None

    if exists and resolved_path is not None:
        stat = resolved_path.stat()
        file_size_bytes = int(stat.st_size)
        modified_time = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
        if spec.kind.lower() == "csv":
            row_count, column_count, columns_preview = inspect_tabular_file(resolved_path)

    return ResolvedFile(
        phase_group=spec.phase_group,
        role=spec.role,
        expected_path=str(expected_path),
        resolved_path=str(resolved_path) if resolved_path is not None else None,
        exists=exists,
        path_corrected=path_corrected and correction_reason != "exact",
        correction_reason=correction_reason,
        required=spec.required,
        kind=spec.kind,
        notes=spec.notes,
        file_size_bytes=file_size_bytes,
        modified_time=modified_time,
        row_count=row_count,
        column_count=column_count,
        columns_preview=columns_preview,
    )


def resolve_all_expected_files(project_root: Path) -> list[ResolvedFile]:
    return [resolve_expected_file(project_root, spec) for spec in get_expected_files()]


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def append_workflow_log(log_path: Path, text_block: str) -> None:
    ensure_dir(log_path.parent)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n{'=' * 80}\n")
        fh.write(f"[{timestamp}]\n")
        fh.write(text_block.rstrip() + "\n")


def relative_to_root(path: Path | None, project_root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(project_root)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def summarize_inventory(records: Sequence[ResolvedFile]) -> dict:
    required_total = sum(1 for r in records if r.required)
    required_found = sum(1 for r in records if r.required and r.exists)
    optional_total = sum(1 for r in records if not r.required)
    optional_found = sum(1 for r in records if not r.required and r.exists)
    corrected_paths = sum(1 for r in records if r.exists and r.path_corrected)
    missing_required_roles = [r.role for r in records if r.required and not r.exists]
    return {
        "required_total": required_total,
        "required_found": required_found,
        "optional_total": optional_total,
        "optional_found": optional_found,
        "corrected_paths": corrected_paths,
        "missing_required_roles": missing_required_roles,
    }


def inventory_markdown(records: Sequence[ResolvedFile], project_root: Path) -> str:
    summary = summarize_inventory(records)
    lines = [
        "# Deliverable 3 Project Inventory",
        "",
        f"Project root: `{project_root}`",
        "",
        "## Summary",
        "",
        f"- Required files found: {summary['required_found']} / {summary['required_total']}",
        f"- Optional files found: {summary['optional_found']} / {summary['optional_total']}",
        f"- Path-corrected resolutions: {summary['corrected_paths']}",
        f"- Demo players reserved for case-study use: {', '.join(DEMO_PLAYERS)}",
        "",
        "## Inventory",
        "",
        "| Phase group | Role | Required | Exists | Path corrected | Resolved path | Rows | Cols |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for rec in records:
        resolved_rel = relative_to_root(Path(rec.resolved_path), project_root) if rec.resolved_path else None
        lines.append(
            f"| {rec.phase_group} | {rec.role} | {'yes' if rec.required else 'no'} | "
            f"{'yes' if rec.exists else 'no'} | {'yes' if rec.path_corrected else 'no'} | "
            f"{resolved_rel or ''} | {rec.row_count if rec.row_count is not None else ''} | "
            f"{rec.column_count if rec.column_count is not None else ''} |"
        )
    if summary["missing_required_roles"]:
        lines.extend([
            "",
            "## Missing required roles",
            "",
            *[f"- {role}" for role in summary["missing_required_roles"]],
        ])
    return "\n".join(lines) + "\n"


def records_to_rows(records: Iterable[ResolvedFile]) -> list[dict]:
    return [asdict(record) for record in records]
