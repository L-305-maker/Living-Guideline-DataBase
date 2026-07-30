"""Deterministic Evidence Span validation."""

from __future__ import annotations

from src.guideline_information.enums import (
    RecommendedRoute,
    SourceType,
    SpanCoordinateSpace,
    SpanIssueCode,
    SpanSupportStatus,
)
from src.guideline_information.extraction.text_normalization import find_unique, normalized_unique_match
from src.guideline_information.ids import make_record_id
from src.guideline_information.models import ExtractionResult, RecommendationCandidate, ValidationResult

CRITICAL_STATUSES = {
    SpanSupportStatus.UNSUPPORTED,
    SpanSupportStatus.INVALID_SPAN,
    SpanSupportStatus.SOURCE_REVISION_MISMATCH,
}
ALLOWED_COORDINATES_BY_FIELD = {
    "recommendation_text": {SpanCoordinateSpace.CANDIDATE_TEXT, SpanCoordinateSpace.SOURCE_BLOCK},
    "direction": {SpanCoordinateSpace.CANDIDATE_TEXT, SpanCoordinateSpace.SOURCE_BLOCK, SpanCoordinateSpace.CONTEXT_AFTER},
    "strength": {SpanCoordinateSpace.CANDIDATE_TEXT, SpanCoordinateSpace.SOURCE_BLOCK, SpanCoordinateSpace.CONTEXT_AFTER},
    "certainty": {SpanCoordinateSpace.CANDIDATE_TEXT, SpanCoordinateSpace.SOURCE_BLOCK, SpanCoordinateSpace.CONTEXT_AFTER},
    "population": {
        SpanCoordinateSpace.CANDIDATE_TEXT,
        SpanCoordinateSpace.SOURCE_BLOCK,
        SpanCoordinateSpace.CONTEXT_BEFORE,
        SpanCoordinateSpace.SECTION_HEADING,
        SpanCoordinateSpace.CLINICAL_QUESTION,
    },
    "interventions": {SpanCoordinateSpace.CANDIDATE_TEXT, SpanCoordinateSpace.SOURCE_BLOCK},
    "dosage": {SpanCoordinateSpace.CANDIDATE_TEXT, SpanCoordinateSpace.SOURCE_BLOCK},
    "duration": {SpanCoordinateSpace.CANDIDATE_TEXT, SpanCoordinateSpace.SOURCE_BLOCK},
    "conditions": {SpanCoordinateSpace.CANDIDATE_TEXT, SpanCoordinateSpace.SOURCE_BLOCK},
}


class EvidenceSpanValidator:
    def validate(self, candidate: RecommendationCandidate, extraction: ExtractionResult) -> ValidationResult:
        field_statuses: dict[str, list[SpanSupportStatus]] = {}
        field_issue_codes: dict[str, list[SpanIssueCode]] = {}
        corrected_spans: dict[str, list[dict]] = {}
        unsupported: list[str] = []
        invalid: list[str] = []
        for field_name, evidence_items in extraction.field_evidence.items():
            statuses: list[SpanSupportStatus] = []
            issues: list[SpanIssueCode] = []
            corrections: list[dict] = []
            for item in evidence_items:
                status, issue, correction = self._validate_item(candidate, field_name, item)
                statuses.append(status)
                issues.append(issue)
                if correction:
                    corrections.append(correction)
            field_statuses[field_name] = statuses
            field_issue_codes[field_name] = issues
            if corrections:
                corrected_spans[field_name] = corrections
            if any(status == SpanSupportStatus.UNSUPPORTED for status in statuses):
                unsupported.append(field_name)
            if any(status in {SpanSupportStatus.INVALID_SPAN, SpanSupportStatus.SOURCE_REVISION_MISMATCH} for status in statuses):
                invalid.append(field_name)
        flat = [status for statuses in field_statuses.values() for status in statuses]
        overall = _overall_status(flat)
        return ValidationResult(
            validation_id=make_record_id("validation", candidate.candidate_id, extraction.extraction_id, candidate.source_revision_id),
            candidate_id=candidate.candidate_id,
            extraction_id=extraction.extraction_id,
            source_revision_id=candidate.source_revision_id,
            field_statuses=field_statuses,
            field_issue_codes=field_issue_codes,
            corrected_spans=corrected_spans,
            unsupported_fields=unsupported,
            invalid_fields=invalid,
            overall_status=overall,
            recommended_route=RecommendedRoute.HUMAN_REVIEW if any(status in CRITICAL_STATUSES for status in flat) else RecommendedRoute.AUTO_ACCEPT,
        )

    def _validate_item(self, candidate: RecommendationCandidate, field_name: str, item) -> tuple[SpanSupportStatus, SpanIssueCode, dict]:
        if item.source_type == SourceType.MODEL_INFERRED:
            return SpanSupportStatus.UNSUPPORTED, SpanIssueCode.OVER_INFERRED, {}
        if item.source_type == SourceType.NOT_STATED:
            return SpanSupportStatus.SUPPORTED, SpanIssueCode.EXACT_MATCH, {}
        if item.source_revision_id and item.source_revision_id != candidate.source_revision_id:
            return SpanSupportStatus.SOURCE_REVISION_MISMATCH, SpanIssueCode.SOURCE_REVISION_MISMATCH, {}
        if item.source_block_key and item.source_block_key != candidate.source_block_key:
            return SpanSupportStatus.SOURCE_REVISION_MISMATCH, SpanIssueCode.WRONG_SOURCE_BLOCK, {}
        if item.span_coordinate_space not in ALLOWED_COORDINATES_BY_FIELD.get(field_name, {SpanCoordinateSpace.CANDIDATE_TEXT}):
            return SpanSupportStatus.UNSUPPORTED, SpanIssueCode.WRONG_CONTEXT_SOURCE, {}
        source_text = _source_text(candidate, item.span_coordinate_space)
        if item.span_start is not None and item.span_end is not None:
            if item.span_start < 0 or item.span_end > len(source_text) or item.span_start >= item.span_end:
                return SpanSupportStatus.INVALID_SPAN, SpanIssueCode.SPAN_OUT_OF_RANGE, {}
            if source_text[item.span_start : item.span_end] == item.quote:
                return SpanSupportStatus.SUPPORTED, SpanIssueCode.EXACT_MATCH, {}
        unique = find_unique(source_text, item.quote)
        if unique:
            return SpanSupportStatus.PARTIALLY_SUPPORTED, SpanIssueCode.PARTIAL_EVIDENCE, {"corrected_start": unique[0], "corrected_end": unique[1], "level": "unique_quote_search"}
        normalized = normalized_unique_match(source_text, item.quote)
        if normalized:
            start, end, steps = normalized
            return SpanSupportStatus.PARTIALLY_SUPPORTED, SpanIssueCode.NORMALIZATION_ONLY, {"corrected_start": start, "corrected_end": end, "normalization_steps": steps, "level": "controlled_normalization"}
        return SpanSupportStatus.UNSUPPORTED, SpanIssueCode.QUOTE_NOT_FOUND, {}


def _source_text(candidate: RecommendationCandidate, coordinate: SpanCoordinateSpace) -> str:
    if coordinate in {SpanCoordinateSpace.SOURCE_BLOCK, SpanCoordinateSpace.CANDIDATE_TEXT}:
        return candidate.candidate_text
    if coordinate == SpanCoordinateSpace.CONTEXT_BEFORE:
        return candidate.context_before
    if coordinate == SpanCoordinateSpace.CONTEXT_AFTER:
        return candidate.context_after
    if coordinate == SpanCoordinateSpace.SECTION_HEADING:
        return " > ".join(candidate.section_path)
    return candidate.candidate_text


def _overall_status(statuses: list[SpanSupportStatus]) -> SpanSupportStatus:
    if any(status == SpanSupportStatus.SOURCE_REVISION_MISMATCH for status in statuses):
        return SpanSupportStatus.SOURCE_REVISION_MISMATCH
    if any(status == SpanSupportStatus.INVALID_SPAN for status in statuses):
        return SpanSupportStatus.INVALID_SPAN
    if any(status == SpanSupportStatus.UNSUPPORTED for status in statuses):
        return SpanSupportStatus.UNSUPPORTED
    if any(status == SpanSupportStatus.PARTIALLY_SUPPORTED for status in statuses):
        return SpanSupportStatus.PARTIALLY_SUPPORTED
    return SpanSupportStatus.SUPPORTED
