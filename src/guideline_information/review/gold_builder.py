"""Build gold examples from approved review outcomes."""

from __future__ import annotations

from src.guideline_information.enums import AnnotationTier, ReviewDecisionType, ReviewOrigin, ReviewStatus
from src.guideline_information.ids import make_record_id
from src.guideline_information.models import (
    AdjudicationResult,
    GoldRecommendationExample,
    RecommendationCandidate,
    ExtractionResult,
    ReviewDecision,
    ReviewSample,
    VerificationResult,
)


def build_gold_from_review(
    sample: ReviewSample,
    candidate: RecommendationCandidate,
    review: ReviewDecision,
    *,
    extraction: ExtractionResult | None = None,
    verification: VerificationResult | None = None,
    allow_single_reviewer_gold: bool = False,
) -> GoldRecommendationExample:
    if sample.review_status != ReviewStatus.APPROVED:
        raise ValueError("Gold requires an approved sample")
    if not allow_single_reviewer_gold:
        raise ValueError("Single reviewer gold is not allowed")
    _require_source_revision_match(sample, candidate)
    _require_human_review(review)
    _require_real_extraction(extraction)
    if review.decision not in {ReviewDecisionType.ACCEPT, ReviewDecisionType.EDIT}:
        raise ValueError("Review decision is not gold eligible")
    annotation = review.corrected_annotation or {"recommendation_text": candidate.candidate_text}
    return GoldRecommendationExample(
        gold_example_id=make_record_id("gold", sample.sample_id, review.review_id, sample.source_revision_id),
        candidate_id=candidate.candidate_id,
        source_revision_id=sample.source_revision_id,
        final_annotation=annotation,
        provenance_review_ids=[review.review_id],
        extraction_id=extraction.extraction_id if extraction else sample.extraction_id,
        verification_id=verification.verification_id if verification else sample.verification_id,
        prompt_version=extraction.prompt_version if extraction else "",
        model_version=extraction.model_version if extraction else "",
        annotation_tier=AnnotationTier.SILVER_SINGLE_REVIEW,
    )


def build_gold_from_double_agreement(
    sample: ReviewSample,
    candidate: RecommendationCandidate,
    reviews: list[ReviewDecision],
    *,
    extraction: ExtractionResult | None = None,
    verification: VerificationResult | None = None,
) -> GoldRecommendationExample:
    if len(reviews) < 2:
        raise ValueError("Double agreement gold requires two reviews")
    _require_source_revision_match(sample, candidate)
    _require_real_extraction(extraction)
    for review in reviews:
        _require_human_review(review)
    first = reviews[0]
    if any(review.decision != first.decision or review.corrected_annotation != first.corrected_annotation for review in reviews[1:]):
        raise ValueError("Reviews do not agree; adjudication is required")
    if first.decision not in {ReviewDecisionType.ACCEPT, ReviewDecisionType.EDIT}:
        raise ValueError("Review decision is not gold eligible")
    annotation = first.corrected_annotation or {"recommendation_text": candidate.candidate_text}
    return GoldRecommendationExample(
        gold_example_id=make_record_id("gold", sample.sample_id, [review.review_id for review in reviews], sample.source_revision_id),
        candidate_id=candidate.candidate_id,
        source_revision_id=sample.source_revision_id,
        final_annotation=annotation,
        provenance_review_ids=[review.review_id for review in reviews],
        extraction_id=extraction.extraction_id if extraction else sample.extraction_id,
        verification_id=verification.verification_id if verification else sample.verification_id,
        prompt_version=extraction.prompt_version if extraction else "",
        model_version=extraction.model_version if extraction else "",
        annotation_tier=AnnotationTier.GOLD_DOUBLE_AGREEMENT,
    )


def build_gold_from_adjudication(
    sample: ReviewSample,
    candidate: RecommendationCandidate,
    adjudication: AdjudicationResult,
    *,
    extraction: ExtractionResult | None = None,
    verification: VerificationResult | None = None,
    source_reviews: list[ReviewDecision] | None = None,
) -> GoldRecommendationExample:
    if sample.review_status not in {ReviewStatus.ADJUDICATED, ReviewStatus.APPROVED}:
        raise ValueError("Gold from adjudication requires adjudicated or approved sample")
    _require_source_revision_match(sample, candidate)
    _require_real_extraction(extraction)
    if source_reviews is not None:
        source_review_ids = {review.review_id for review in source_reviews}
        if set(adjudication.source_review_ids) - source_review_ids:
            raise ValueError("Adjudication references missing source reviews")
        for review in source_reviews:
            _require_human_review(review)
    if adjudication.final_decision not in {ReviewDecisionType.ACCEPT, ReviewDecisionType.EDIT}:
        raise ValueError("Adjudication decision is not gold eligible")
    annotation = adjudication.final_annotation or {"recommendation_text": candidate.candidate_text}
    return GoldRecommendationExample(
        gold_example_id=make_record_id("gold", sample.sample_id, adjudication.adjudication_id, sample.source_revision_id),
        candidate_id=candidate.candidate_id,
        source_revision_id=sample.source_revision_id,
        final_annotation=annotation,
        provenance_review_ids=adjudication.source_review_ids,
        extraction_id=extraction.extraction_id if extraction else sample.extraction_id,
        verification_id=verification.verification_id if verification else sample.verification_id,
        adjudication_id=adjudication.adjudication_id,
        prompt_version=extraction.prompt_version if extraction else "",
        model_version=extraction.model_version if extraction else "",
        annotation_tier=AnnotationTier.GOLD_ADJUDICATED,
    )


def _require_human_review(review: ReviewDecision) -> None:
    if review.review_origin != ReviewOrigin.HUMAN:
        raise ValueError("Gold requires HUMAN review_origin")


def _require_real_extraction(extraction: ExtractionResult | None) -> None:
    if extraction is None:
        return
    model_name = (extraction.model_name or extraction.model_version or "").lower()
    if model_name.startswith("fake") or "fake" in model_name:
        raise ValueError("Gold cannot be built from fake client extraction")


def _require_source_revision_match(sample: ReviewSample, candidate: RecommendationCandidate) -> None:
    if sample.source_revision_id != candidate.source_revision_id:
        raise ValueError("Source revision mismatch")
