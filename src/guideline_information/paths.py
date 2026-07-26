"""Filesystem layout for guideline information runs."""

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
    def sample_manifest(self) -> Path:
        return self.run_dir / "sample_manifest.json"

    @property
    def review_samples(self) -> Path:
        return self.run_dir / "review_samples.jsonl"

    @property
    def review_decisions(self) -> Path:
        return self.run_dir / "review_decisions.jsonl"

    @property
    def adjudications(self) -> Path:
        return self.run_dir / "adjudications.jsonl"

    @property
    def gold_examples(self) -> Path:
        return self.run_dir / "gold_examples.jsonl"

    @classmethod
    def create(cls, root_dir: str | Path = DEFAULT_INFORMATION_DIR, run_id: str | None = None) -> "InformationRunPaths":
        paths = cls(Path(root_dir), run_id or new_run_id())
        paths.run_dir.mkdir(parents=True, exist_ok=False)
        return paths

    @classmethod
    def existing(cls, root_dir: str | Path = DEFAULT_INFORMATION_DIR, run_id: str = "") -> "InformationRunPaths":
        if not run_id:
            raise ValueError("run_id is required")
        return cls(Path(root_dir), run_id)
