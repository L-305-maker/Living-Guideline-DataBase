"""Deterministic routing for pilot extraction results."""

from __future__ import annotations

import random

from src.guideline_information.enums import RecommendedRoute, RecommendationType, SpanSupportStatus
from src.guideline_information.ids import make_record_id
from src.guideline_information.models import ExtractionResult, RecommendationCandidate, RouteResult, ValidationResult, VerificationResult


def route_candidate(
    candidate: RecommendationCandidate,
    extraction: ExtractionResult,
    verification: VerificationResult,
    validation: ValidationResult,
    *,
    random_audit_rate: float = 0.0,
    seed: int = 0,
) -> RouteResult:
    reasons: list[str] = []
    route = RecommendedRoute.AUTO_ACCEPT
    if extraction.recommendation_type in {
        RecommendationType.NOT_RECOMMENDATION,
        RecommendationType.RATIONALE,
        RecommendationType.EVIDENCE_SUMMARY,
        RecommendationType.RESEARCH_RECOMMENDATION,
    } or not extraction.is_formal_recommendation:
        route = RecommendedRoute.REJECT
        reasons.append("not_formal_recommendation")
    if verification.recommended_route == RecommendedRoute.REJECT:
        route = RecommendedRoute.REJECT
        reasons.append("verifier_reject")
    if verification.agrees_is_formal_recommendation is False and extraction.is_formal_recommendation:
        route = RecommendedRoute.HUMAN_REVIEW
        reasons.append("extractor_verifier_conflict")
    if verification.unsupported_fields or verification.logic_errors or verification.field_conflicts:
        route = RecommendedRoute.HUMAN_REVIEW
        reasons.append("verifier_found_issues")
    if validation.overall_status in {
        SpanSupportStatus.UNSUPPORTED,
        SpanSupportStatus.INVALID_SPAN,
        SpanSupportStatus.SOURCE_REVISION_MISMATCH,
    }:
        route = RecommendedRoute.HUMAN_REVIEW
        reasons.append("invalid_evidence_span")
    if extraction.parsed_with_repair or verification.parsed_with_repair:
        route = RecommendedRoute.HUMAN_REVIEW
        reasons.append("model_json_repaired")
    if _looks_multi_claim(candidate.candidate_text):
        route = RecommendedRoute.HUMAN_REVIEW
        reasons.append("multi_claim")
    if route == RecommendedRoute.AUTO_ACCEPT and random_audit_rate > 0:
        rng = random.Random(f"{seed}:{candidate.candidate_id}")
        if rng.random() < random_audit_rate:
            route = RecommendedRoute.RANDOM_AUDIT
            reasons.append("random_audit")
    if not reasons:
        reasons.append("all_checks_passed")
    return RouteResult(
        route_id=make_record_id("route", candidate.candidate_id, extraction.extraction_id, verification.verification_id, validation.validation_id),
        candidate_id=candidate.candidate_id,
        extraction_id=extraction.extraction_id,
        verification_id=verification.verification_id,
        validation_id=validation.validation_id,
        route=route,
        reasons=reasons,
    )


def _looks_multi_claim(text: str) -> bool:
    lower = text.lower()
    return lower.count(" or ") + lower.count(" and ") + lower.count("; ") >= 3
