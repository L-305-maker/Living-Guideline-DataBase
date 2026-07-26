"""Thin orchestration helpers for the information extraction phase."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.guideline_information.extraction.candidate_builder import CandidateBuilder
from src.guideline_information.models import RecommendationCandidate
from src.utils.io import read_jsonl


def load_section_records(sections_dir: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(Path(sections_dir).glob("*.jsonl")):
        records.extend(read_jsonl(path))
    return records


def build_candidates_from_sections(sections_dir: str | Path, pipeline_run_id: str) -> list[RecommendationCandidate]:
    return CandidateBuilder().build(load_section_records(sections_dir), pipeline_run_id)
