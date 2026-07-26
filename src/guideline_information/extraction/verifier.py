"""Verifier protocol definitions."""

from __future__ import annotations

from typing import Protocol

from src.guideline_information.models import ExtractionResult, VerificationResult


class RecommendationVerifier(Protocol):
    def verify(self, extraction: ExtractionResult) -> VerificationResult:
        ...
