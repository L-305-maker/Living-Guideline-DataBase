"""Pluggable review sample selectors."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Iterable

from src.guideline_information.enums import RecommendedRoute, SelectorName
from src.guideline_information.ids import make_record_id
from src.guideline_information.models import RecommendationCandidate, ReviewSample, VerificationResult


@dataclass(frozen=True)
class SelectorInput:
    candidates: list[RecommendationCandidate]
    verification_by_candidate: dict[str, VerificationResult]


class BaseSelector:
    name: SelectorName

    def filter_candidates(self, data: SelectorInput) -> list[RecommendationCandidate]:
        return data.candidates

    def select(self, data: SelectorInput, budget: int, seed: int) -> list[RecommendationCandidate]:
        pool = self.filter_candidates(data)
        rng = random.Random(seed)
        shuffled = list(pool)
        rng.shuffle(shuffled)
        return shuffled[: max(0, budget)]


class RandomAuditSelector(BaseSelector):
    name = SelectorName.RANDOM_AUDIT

    def filter_candidates(self, data: SelectorInput) -> list[RecommendationCandidate]:
        return [item for item in data.candidates if item.candidate_score >= 0.8]


class UncertaintySelector(BaseSelector):
    name = SelectorName.UNCERTAINTY

    def filter_candidates(self, data: SelectorInput) -> list[RecommendationCandidate]:
        return [item for item in data.candidates if item.candidate_score < 0.8 or bool(item.candidate_profile.get("missing_fields"))]


class ExtractorVerifierDisagreementSelector(BaseSelector):
    name = SelectorName.EXTRACTOR_VERIFIER_DISAGREEMENT

    def filter_candidates(self, data: SelectorInput) -> list[RecommendationCandidate]:
        return [
            item
            for item in data.candidates
            if data.verification_by_candidate.get(item.candidate_id)
            and data.verification_by_candidate[item.candidate_id].recommended_route == RecommendedRoute.ADJUDICATION
        ]


class RuleModelConflictSelector(BaseSelector):
    name = SelectorName.RULE_MODEL_CONFLICT

    def filter_candidates(self, data: SelectorInput) -> list[RecommendationCandidate]:
        return [item for item in data.candidates if bool(item.candidate_profile.get("rule_model_conflict"))]


class HardNegativeSelector(BaseSelector):
    name = SelectorName.HARD_NEGATIVE
    needles = ("method", "research recommendation", "rationale", "executive summary", "good practice")

    def filter_candidates(self, data: SelectorInput) -> list[RecommendationCandidate]:
        return [item for item in data.candidates if any(token in item.candidate_text.lower() for token in self.needles)]


class MultiClaimSelector(BaseSelector):
    name = SelectorName.MULTI_CLAIM

    def filter_candidates(self, data: SelectorInput) -> list[RecommendationCandidate]:
        return [item for item in data.candidates if item.candidate_text.lower().count(" or ") + item.candidate_text.lower().count(" and ") >= 2]


class FormatStratifiedSelector(BaseSelector):
    name = SelectorName.FORMAT_STRATIFIED

    def filter_candidates(self, data: SelectorInput) -> list[RecommendationCandidate]:
        return sorted(data.candidates, key=lambda item: (item.doc_id, item.section_path, item.candidate_id))


def build_review_samples(
    *,
    candidates: Iterable[RecommendationCandidate],
    selectors: list[BaseSelector],
    budgets: dict[str, int],
    sample_batch_id: str,
    seed: int = 0,
    verification_by_candidate: dict[str, VerificationResult] | None = None,
) -> tuple[list[ReviewSample], dict]:
    data = SelectorInput(list(candidates), verification_by_candidate or {})
    selected: list[ReviewSample] = []
    seen_candidates: set[str] = set()
    for offset, selector in enumerate(selectors):
        for candidate in selector.select(data, budgets.get(selector.name.value, 0), seed + offset):
            if candidate.candidate_id in seen_candidates:
                continue
            seen_candidates.add(candidate.candidate_id)
            selected.append(
                ReviewSample(
                    sample_id=make_record_id("sample", sample_batch_id, selector.name.value, candidate.candidate_id),
                    candidate_id=candidate.candidate_id,
                    sample_batch_id=sample_batch_id,
                    selection_strategy=selector.name.value,
                    selection_reasons=list(candidate.candidate_signals),
                    priority=len(selected),
                    source_revision_id=candidate.source_revision_id,
                    source_text_snapshot=candidate.candidate_text,
                )
            )
    manifest = {
        "total_candidates": len(data.candidates),
        "selected_samples": len(selected),
        "selector_budgets": budgets,
        "random_seed": seed,
        "dedupe_rule": "one sample per candidate_id per batch",
        "sampling_strategy_version": "review_selectors_v1",
        "input_pipeline_run": sorted({item.pipeline_run_id for item in data.candidates}),
        "built_at": datetime.now(UTC).isoformat(),
    }
    return selected, manifest
