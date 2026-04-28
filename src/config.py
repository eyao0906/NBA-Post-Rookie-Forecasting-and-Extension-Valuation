"""Central configuration for CaseStudy3 pipeline."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    # Cohort boundaries
    draft_start_year: int = 1999
    draft_end_year: int = 2019
    min_games_first4: int = 0
    max_games_first4: int = 328  # 82 * 4

    # API behavior
    api_sleep_seconds: float = 1.0
    api_max_retries: int = 4
    api_backoff_seconds: float = 1.5

    # Model defaults
    test_size: float = 0.2
    random_state: int = 946
    epochs: int = 60
    batch_size: int = 16


def ensure_dirs(data_dir: Path, visual_dir: Path, report_dir: Path) -> None:
    (data_dir / "raw").mkdir(parents=True, exist_ok=True)
    (data_dir / "intermediate").mkdir(parents=True, exist_ok=True)
    (data_dir / "processed").mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
