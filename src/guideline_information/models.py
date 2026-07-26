"""Pydantic domain snapshots for guideline information processing."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.guideline_information.enums import RecommendedRoute, ReviewDecisionType, ReviewStatus


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
    recommendation_text: str = ""
    direction: str | None = None
    strength: str | None = None
    certainty: str | None = None
    population: str | None = None
    interventions: list[str] = Field(default_factory=list)
    dosage: str | None = None
    duration: str | None = None
    field_evidence: dict[str, list[str]] = Field(default_factory=dict)
    validation_errors: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class VerificationResult(FrozenModel):
    verification_id: str
    extraction_id: str
    verifier_name: str
    verifier_version: str = ""
    schema_version: str = SCHEMA_VERSION
    field_agreements: dict[str, bool] = Field(default_factory=dict)
    field_conflicts: dict[str, str] = Field(default_factory=dict)
    unsupported_fields: list[str] = Field(default_factory=list)
    logic_errors: list[str] = Field(default_factory=list)
    recommended_route: RecommendedRoute = RecommendedRoute.HUMAN_REVIEW
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
    state_history: list[dict[str, Any]] = Field(default_factory=list)
    schema_version: str = SCHEMA_VERSION
    created_at: datetime = Field(default_factory=utc_now)


class ReviewDecision(FrozenModel):
    review_id: str
    sample_id: str
    reviewer_id: str
    review_round: int
    decision: ReviewDecisionType
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
    adjudication_id: str | None = None
    gold_schema_version: str = SCHEMA_VERSION
    schema_version: str = SCHEMA_VERSION
    approved_at: datetime = Field(default_factory=utc_now)
