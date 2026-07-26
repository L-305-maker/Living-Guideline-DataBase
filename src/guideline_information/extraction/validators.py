"""Validation helpers for extraction records."""

from __future__ import annotations

from typing import Any

from src.guideline_information.models import ExtractionResult
from src.utils.io import read_jsonl


def validate_extraction_result(record: dict[str, Any]) -> ExtractionResult:
    return ExtractionResult.model_validate(record)


def read_valid_jsonl(path: str) -> list[dict[str, Any]]:
    return list(read_jsonl(path))
