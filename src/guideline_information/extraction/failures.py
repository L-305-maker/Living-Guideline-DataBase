"""Structured adapter failure records."""

from __future__ import annotations

from typing import Any

from src.guideline_information.enums import FailureCode


class AdapterFailure(ValueError):
    def __init__(
        self,
        message: str,
        *,
        failure_code: FailureCode,
        stage: str,
        field_path: str = "",
        retryable: bool = False,
        exception_type: str = "",
        response_record: dict[str, Any] | None = None,
        normalization_record: dict[str, Any] | None = None,
        canonical_validation_record: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_code = failure_code
        self.stage = stage
        self.field_path = field_path
        self.retryable = retryable
        self.exception_type = exception_type or type(self).__name__
        self.response_record = response_record or {}
        self.normalization_record = normalization_record or {}
        self.canonical_validation_record = canonical_validation_record or {}

    def failure_record(self, *, candidate_id: str, extraction_id: str = "") -> dict[str, Any]:
        return {
            "candidate_id": candidate_id,
            "extraction_id": extraction_id,
            "failure_code": self.failure_code.value,
            "exception_type": self.exception_type,
            "stage": self.stage,
            "field_path": self.field_path,
            "retryable": self.retryable,
            "error": str(self),
        }

