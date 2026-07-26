"""Extractor protocol definitions."""

from __future__ import annotations

from typing import Protocol

from src.guideline_information.models import ExtractionResult, RecommendationCandidate


class RecommendationExtractor(Protocol):
    def extract(self, candidate: RecommendationCandidate) -> ExtractionResult:
        ...
