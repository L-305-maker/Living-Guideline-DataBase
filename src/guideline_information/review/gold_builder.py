"""Build gold examples from approved review outcomes."""

from __future__ import annotations

from src.guideline_information.enums import ReviewDecisionType, ReviewStatus
from src.guideline_information.ids import make_record_id
from src.guideline_information.models import (
    AdjudicationResult,
    GoldRecommendationExample,
    RecommendationCandidate,
    ReviewDecision,
    ReviewSample,
)


def build_gold_from_review(
    sample: ReviewSample,
    candidate: RecommendationCandidate,
    review: ReviewDecision,
    *,
    allow_single_reviewer_gold: bool = False,
) -> GoldRecommendationExample:
    if sample.review_status != ReviewStatus.APPROVED:
        raise ValueError("Gold requires an approved sample")
    if not allow_single_reviewer_gold:
        raise ValueError("Single reviewer gold is not allowed")
    if review.decision not in {ReviewDecisionType.ACCEPT, ReviewDecisionType.EDIT}:
        raise ValueError("Review decision is not gold eligible")
    annotation = review.corrected_annotation or {"recommendation_text": candidate.candidate_text}
    return GoldRecommendationExample(
        gold_example_id=make_record_id("gold", sample.sample_id, review.review_id, sample.source_revision_id),
        candidate_id=candidate.candidate_id,
        source_revision_id=sample.source_revision_id,
        final_annotation=annotation,
        provenance_review_ids=[review.review_id],
    )


def build_gold_from_adjudication(
    sample: ReviewSample,
    candidate: RecommendationCandidate,
    adjudication: AdjudicationResult,
) -> GoldRecommendationExample:
    if sample.review_status not in {ReviewStatus.ADJUDICATED, ReviewStatus.APPROVED}:
        raise ValueError("Gold from adjudication requires adjudicated or approved sample")
    if adjudication.final_decision not in {ReviewDecisionType.ACCEPT, ReviewDecisionType.EDIT}:
        raise ValueError("Adjudication decision is not gold eligible")
    annotation = adjudication.final_annotation or {"recommendation_text": candidate.candidate_text}
    return GoldRecommendationExample(
        gold_example_id=make_record_id("gold", sample.sample_id, adjudication.adjudication_id, sample.source_revision_id),
        candidate_id=candidate.candidate_id,
        source_revision_id=sample.source_revision_id,
        final_annotation=annotation,
        provenance_review_ids=adjudication.source_review_ids,
        adjudication_id=adjudication.adjudication_id,
    )
