"""Pydantic domain snapshots for guideline information processing."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.guideline_information.enums import (
    AnnotationTier,
    RecommendedRoute,
    RecommendationType,
    ReviewDecisionType,
    ReviewOrigin,
    ReviewStatus,
    SourceType,
    SpanCoordinateSpace,
    SpanIssueCode,
    SpanSupportStatus,
)


SCHEMA_VERSION = "guideline_information_v1"


def utc_now() -> datetime:
    return datetime.now(UTC)


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("*", mode="after")
    @classmethod
    def require_aware_datetime(cls, value: Any) -> Any:
        if isinstance(value, datetime) and value.tzinfo is None:
            raise ValueError("datetime fields must be timezone-aware")
        return value


class FieldEvidence(FrozenModel):
    value_original: str = ""
    value_normalized: str = ""
    source_type: SourceType = SourceType.NOT_STATED
    source_block_key: str = ""
    source_revision_id: str = ""
    span_coordinate_space: SpanCoordinateSpace = SpanCoordinateSpace.CANDIDATE_TEXT
    quote: str = ""
    span_start: int | None = None
    span_end: int | None = None
    confidence_signal: str = ""


class RecommendationCandidate(FrozenModel):
    candidate_id: str
    pipeline_run_id: str
    doc_id: str
    source_block_key: str
    source_revision_id: str
    source_id_algorithm: str
    section_path: list[str] = Field(default_factory=list)
    candidate_text: str
    context_before: str = ""
    context_after: str = ""
    title: str = ""
    source_institution: str = ""
    publication_date: str = ""
    candidate_signals: list[str] = Field(default_factory=list)
    candidate_profile: dict[str, Any] = Field(default_factory=dict)
    candidate_score: float = 0.0
    schema_version: str = SCHEMA_VERSION


class ExtractionResult(FrozenModel):
    extraction_id: str
    candidate_id: str
    extractor_name: str
    model_name: str = ""
    model_version: str = ""
    prompt_version: str = ""
    schema_version: str = SCHEMA_VERSION
    is_formal_recommendation: bool
    recommendation_type: RecommendationType = RecommendationType.UNRESOLVED
    recommendation_text: str = ""
    direction: str | None = None
    strength: str | None = None
    certainty: str | None = None
    population: str | None = None
    interventions: list[str] = Field(default_factory=list)
    dosage: str | None = None
    duration: str | None = None
    conditions: list[str] = Field(default_factory=list)
    field_evidence: dict[str, list[FieldEvidence]] = Field(default_factory=dict)
    validation_errors: list[str] = Field(default_factory=list)
    model_request_id: str = ""
    raw_response_text: str = ""
    parsed_with_repair: bool = False
    usage: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = 0
    created_at: datetime = Field(default_factory=utc_now)


class VerificationResult(FrozenModel):
    verification_id: str
    extraction_id: str
    candidate_id: str = ""
    verifier_name: str
    verifier_version: str = ""
    model_name: str = ""
    prompt_version: str = ""
    schema_version: str = SCHEMA_VERSION
    agrees_is_formal_recommendation: bool | None = None
    field_agreements: dict[str, bool] = Field(default_factory=dict)
    field_conflicts: dict[str, str] = Field(default_factory=dict)
    unsupported_fields: list[str] = Field(default_factory=list)
    logic_errors: list[str] = Field(default_factory=list)
    recommended_route: RecommendedRoute = RecommendedRoute.HUMAN_REVIEW
    model_request_id: str = ""
    raw_response_text: str = ""
    parsed_with_repair: bool = False
    usage: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = 0
    created_at: datetime = Field(default_factory=utc_now)


class ValidationResult(FrozenModel):
    validation_id: str
    candidate_id: str
    extraction_id: str
    source_revision_id: str
    field_statuses: dict[str, list[SpanSupportStatus]] = Field(default_factory=dict)
    field_issue_codes: dict[str, list[SpanIssueCode]] = Field(default_factory=dict)
    corrected_spans: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    unsupported_fields: list[str] = Field(default_factory=list)
    invalid_fields: list[str] = Field(default_factory=list)
    overall_status: SpanSupportStatus = SpanSupportStatus.SUPPORTED
    recommended_route: RecommendedRoute = RecommendedRoute.AUTO_ACCEPT
    schema_version: str = SCHEMA_VERSION
    created_at: datetime = Field(default_factory=utc_now)


class RouteResult(FrozenModel):
    route_id: str
    candidate_id: str
    extraction_id: str
    verification_id: str
    validation_id: str
    route: RecommendedRoute
    reasons: list[str] = Field(default_factory=list)
    schema_version: str = SCHEMA_VERSION
    created_at: datetime = Field(default_factory=utc_now)


class ReviewSample(FrozenModel):
    sample_id: str
    candidate_id: str
    extraction_id: str | None = None
    verification_id: str | None = None
    sample_batch_id: str
    selection_strategy: str
    selection_reasons: list[str] = Field(default_factory=list)
    priority: int = 0
    review_status: ReviewStatus = ReviewStatus.PENDING
    assigned_reviewer_ids: list[str] = Field(default_factory=list)
    source_revision_id: str
    source_text_snapshot: str
    context_before: str = ""
    context_after: str = ""
    readonly_snapshot: dict[str, Any] = Field(default_factory=dict)
    state_history: list[dict[str, Any]] = Field(default_factory=list)
    schema_version: str = SCHEMA_VERSION
    created_at: datetime = Field(default_factory=utc_now)


class ReviewDecision(FrozenModel):
    review_id: str
    sample_id: str
    reviewer_id: str
    review_round: int
    decision: ReviewDecisionType
    review_origin: ReviewOrigin = ReviewOrigin.SYNTHETIC_TEST
    corrected_annotation: dict[str, Any] = Field(default_factory=dict)
    field_decisions: dict[str, str] = Field(default_factory=dict)
    issue_codes: list[str] = Field(default_factory=list)
    comment: str = ""
    schema_version: str = SCHEMA_VERSION
    submitted_at: datetime = Field(default_factory=utc_now)


class AdjudicationResult(FrozenModel):
    adjudication_id: str
    sample_id: str
    source_review_ids: list[str]
    adjudicator_id: str
    final_decision: ReviewDecisionType
    final_annotation: dict[str, Any] = Field(default_factory=dict)
    adjudication_reason: str = ""
    schema_version: str = SCHEMA_VERSION
    created_at: datetime = Field(default_factory=utc_now)


class GoldRecommendationExample(FrozenModel):
    gold_example_id: str
    candidate_id: str
    source_revision_id: str
    final_annotation: dict[str, Any]
    provenance_review_ids: list[str] = Field(default_factory=list)
    extraction_id: str | None = None
    verification_id: str | None = None
    adjudication_id: str | None = None
    prompt_version: str = ""
    model_version: str = ""
    annotation_tier: AnnotationTier = AnnotationTier.SILVER_SINGLE_REVIEW
    gold_schema_version: str = SCHEMA_VERSION
    schema_version: str = SCHEMA_VERSION
    approved_at: datetime = Field(default_factory=utc_now)
