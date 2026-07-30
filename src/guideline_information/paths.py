"""Filesystem layout for guideline information runs and pilots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.utils.io import DATA_DIR


DEFAULT_INFORMATION_DIR = DATA_DIR.parent / "guideline_information"


def new_run_id(now: datetime | None = None) -> str:
    value = now or datetime.now(UTC)
    return value.strftime("%Y%m%dT%H%M%SZ")


@dataclass(frozen=True)
class PilotPaths:
    root_dir: Path
    pilot_id: str

    @property
    def pilot_dir(self) -> Path:
        return self.root_dir / "pilots" / self.pilot_id

    @property
    def manifest(self) -> Path:
        return self.pilot_dir / "pilot_manifest.json"

    @property
    def selected_documents(self) -> Path:
        return self.pilot_dir / "selected_documents.jsonl"

    @property
    def selected_sections(self) -> Path:
        return self.pilot_dir / "selected_sections.jsonl"

    @property
    def expected_coverage(self) -> Path:
        return self.pilot_dir / "expected_coverage.json"

    @classmethod
    def create(cls, root_dir: str | Path = DEFAULT_INFORMATION_DIR, pilot_id: str = "") -> "PilotPaths":
        if not pilot_id:
            raise ValueError("pilot_id is required")
        paths = cls(Path(root_dir), pilot_id)
        paths.pilot_dir.mkdir(parents=True, exist_ok=True)
        return paths


@dataclass(frozen=True)
class InformationRunPaths:
    root_dir: Path
    run_id: str

    @property
    def run_dir(self) -> Path:
        return self.root_dir / "runs" / self.run_id

    @property
    def manifest(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def candidates(self) -> Path:
        return self.run_dir / "candidates.jsonl"

    @property
    def extraction_results(self) -> Path:
        return self.run_dir / "extraction_results.jsonl"

    @property
    def verification_results(self) -> Path:
        return self.run_dir / "verification_results.jsonl"

    @property
    def validation_results(self) -> Path:
        return self.run_dir / "validation_results.jsonl"

    @property
    def route_results(self) -> Path:
        return self.run_dir / "route_results.jsonl"


    @property
    def model_responses(self) -> Path:
        return self.run_dir / "model_responses.jsonl"

    @property
    def normalization_results(self) -> Path:
        return self.run_dir / "normalization_results.jsonl"

    @property
    def canonical_validation_results(self) -> Path:
        return self.run_dir / "canonical_validation_results.jsonl"

    @property
    def extraction_failures(self) -> Path:
        return self.run_dir / "extraction_failures.jsonl"

    @property
    def verification_failures(self) -> Path:
        return self.run_dir / "verification_failures.jsonl"

    @property
    def sample_manifest(self) -> Path:
        return self.run_dir / "sample_manifest.json"

    @property
    def review_samples(self) -> Path:
        return self.run_dir / "review_samples.jsonl"

    @property
    def review_csv(self) -> Path:
        return self.run_dir / "review_samples.csv"

    @property
    def review_fields_csv(self) -> Path:
        return self.run_dir / "review_fields.csv"

    @property
    def review_decisions(self) -> Path:
        return self.run_dir / "review_decisions.jsonl"

    @property
    def adjudications(self) -> Path:
        return self.run_dir / "adjudications.jsonl"

    @property
    def gold_examples(self) -> Path:
        return self.run_dir / "gold_examples.jsonl"

    @property
    def evaluation_report_json(self) -> Path:
        return self.run_dir / "evaluation_report.json"

    @property
    def evaluation_report_md(self) -> Path:
        return self.run_dir / "evaluation_report.md"

    @property
    def error_examples(self) -> Path:
        return self.run_dir / "error_examples.jsonl"

    @classmethod
    def create(cls, root_dir: str | Path = DEFAULT_INFORMATION_DIR, run_id: str | None = None) -> "InformationRunPaths":
        paths = cls(Path(root_dir), run_id or new_run_id())
        paths.run_dir.mkdir(parents=True, exist_ok=False)
        return paths

    @classmethod
    def ensure(cls, root_dir: str | Path = DEFAULT_INFORMATION_DIR, run_id: str | None = None) -> "InformationRunPaths":
        paths = cls(Path(root_dir), run_id or new_run_id())
        paths.run_dir.mkdir(parents=True, exist_ok=True)
        return paths

    @classmethod
    def existing(cls, root_dir: str | Path = DEFAULT_INFORMATION_DIR, run_id: str = "") -> "InformationRunPaths":
        if not run_id:
            raise ValueError("run_id is required")
        return cls(Path(root_dir), run_id)
