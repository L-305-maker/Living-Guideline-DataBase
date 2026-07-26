"""Adjudication helpers."""

from __future__ import annotations

from src.guideline_information.enums import ReviewDecisionType
from src.guideline_information.ids import make_record_id
from src.guideline_information.models import AdjudicationResult, ReviewDecision, ReviewSample


def create_adjudication(
    sample: ReviewSample,
    reviews: list[ReviewDecision],
    *,
    adjudicator_id: str,
    final_decision: ReviewDecisionType,
    final_annotation: dict,
    reason: str = "",
) -> AdjudicationResult:
    if len(reviews) < 2:
        raise ValueError("Adjudication requires at least two review decisions")
    review_ids = [review.review_id for review in reviews]
    return AdjudicationResult(
        adjudication_id=make_record_id("adjudication", sample.sample_id, review_ids, final_decision, final_annotation),
        sample_id=sample.sample_id,
        source_review_ids=review_ids,
        adjudicator_id=adjudicator_id,
        final_decision=final_decision,
        final_annotation=final_annotation,
        adjudication_reason=reason,
    )
