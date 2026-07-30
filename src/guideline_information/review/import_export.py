"""CSV exchange for human review."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.guideline_information.enums import IssueCode, ReviewDecisionType, ReviewOrigin, ReviewStatus
from src.guideline_information.ids import make_record_id
from src.guideline_information.models import ExtractionResult, RecommendationCandidate, ReviewDecision, ReviewSample, VerificationResult

READONLY_FIELDS = [
    "sample_id",
    "candidate_id",
    "doc_id",
    "title",
    "section_path",
    "source_revision_id",
    "context_before",
    "candidate_text",
    "context_after",
    "extractor_is_formal",
    "extractor_direction",
    "extractor_strength",
    "extractor_certainty",
    "extractor_population",
    "extractor_interventions",
    "verifier_route",
    "selection_reasons",
]
REVIEW_FIELDS = [
    "review_decision",
    "correct_is_formal",
    "correct_recommendation_type",
    "correct_recommendation_text",
    "correct_direction",
    "correct_strength",
    "correct_certainty",
    "correct_population",
    "correct_interventions",
    "issue_codes",
    "review_comment",
    "review_origin",
    "reviewer_id",
]
CSV_FIELDS = READONLY_FIELDS + REVIEW_FIELDS


def export_review_samples_csv(
    samples: list[ReviewSample],
    path: str | Path,
    *,
    candidates_by_id: dict[str, RecommendationCandidate] | None = None,
    extractions_by_id: dict[str, ExtractionResult] | None = None,
    verifications_by_id: dict[str, VerificationResult] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for sample in samples:
            writer.writerow(_readonly_row(sample, candidates_by_id or {}, extractions_by_id or {}, verifications_by_id or {}))


def import_review_decisions_csv(
    path: str | Path,
    samples: list[ReviewSample],
    *,
    candidates_by_id: dict[str, RecommendationCandidate] | None = None,
    extractions_by_id: dict[str, ExtractionResult] | None = None,
    verifications_by_id: dict[str, VerificationResult] | None = None,
    existing_review_ids: set[str] | None = None,
    review_origin: ReviewOrigin = ReviewOrigin.SYNTHETIC_TEST,
) -> list[ReviewDecision]:
    sample_by_id = {sample.sample_id: sample for sample in samples}
    seen = set(existing_review_ids or set())
    decisions: list[ReviewDecision] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for line_no, row in enumerate(csv.DictReader(handle), start=2):
            sample = sample_by_id.get(row.get("sample_id") or "")
            if sample is None:
                raise ValueError(f"Unknown sample_id at CSV line {line_no}")
            _validate_readonly(row, sample, candidates_by_id or {}, extractions_by_id or {}, verifications_by_id or {}, line_no)
            if sample.review_status not in {ReviewStatus.PENDING, ReviewStatus.ASSIGNED, ReviewStatus.IN_REVIEW, ReviewStatus.SUBMITTED}:
                raise ValueError(f"Sample status does not allow submission at CSV line {line_no}")
            reviewer_id = (row.get("reviewer_id") or "").strip()
            if not reviewer_id:
                raise ValueError(f"reviewer_id is required at CSV line {line_no}")
            try:
                decision = ReviewDecisionType((row.get("review_decision") or row.get("decision") or "").strip())
            except ValueError as exc:
                raise ValueError(f"Invalid review_decision at CSV line {line_no}") from exc
            corrected = _corrected_annotation(row)
            issue_codes = _issue_codes(row.get("issue_codes") or "", line_no)
            if decision == ReviewDecisionType.EDIT and not corrected:
                raise ValueError(f"EDIT requires at least one corrected field at CSV line {line_no}")
            if decision == ReviewDecisionType.REJECT and not issue_codes:
                raise ValueError(f"REJECT requires issue_codes at CSV line {line_no}")
            if decision == ReviewDecisionType.UNRESOLVED and not (row.get("review_comment") or "").strip():
                raise ValueError(f"UNRESOLVED requires review_comment at CSV line {line_no}")
            origin = _review_origin(row.get("review_origin") or "", review_origin, line_no)
            review_id = make_record_id("review", sample.sample_id, reviewer_id, row.get("review_decision"), corrected, issue_codes, origin.value)
            if review_id in seen:
                raise ValueError(f"Duplicate review submission at CSV line {line_no}")
            seen.add(review_id)
            decisions.append(
                ReviewDecision(
                    review_id=review_id,
                    sample_id=sample.sample_id,
                    reviewer_id=reviewer_id,
                    review_round=1,
                    decision=decision,
                    review_origin=origin,
                    corrected_annotation=corrected,
                    issue_codes=issue_codes,
                    comment=row.get("review_comment") or "",
                )
            )
    return decisions


def _readonly_row(
    sample: ReviewSample,
    candidates_by_id: dict[str, RecommendationCandidate],
    extractions_by_id: dict[str, ExtractionResult],
    verifications_by_id: dict[str, VerificationResult],
) -> dict[str, str]:
    candidate = candidates_by_id.get(sample.candidate_id)
    extraction = extractions_by_id.get(sample.extraction_id or "")
    verification = verifications_by_id.get(sample.verification_id or "")
    return {
        "sample_id": sample.sample_id,
        "candidate_id": sample.candidate_id,
        "doc_id": candidate.doc_id if candidate else "",
        "title": candidate.title if candidate else "",
        "section_path": " > ".join(candidate.section_path) if candidate else "",
        "source_revision_id": sample.source_revision_id,
        "context_before": sample.context_before or (candidate.context_before if candidate else ""),
        "candidate_text": sample.source_text_snapshot,
        "context_after": sample.context_after or (candidate.context_after if candidate else ""),
        "extractor_is_formal": str(extraction.is_formal_recommendation) if extraction else "",
        "extractor_direction": extraction.direction or "" if extraction else "",
        "extractor_strength": extraction.strength or "" if extraction else "",
        "extractor_certainty": extraction.certainty or "" if extraction else "",
        "extractor_population": extraction.population or "" if extraction else "",
        "extractor_interventions": "|".join(extraction.interventions) if extraction else "",
        "verifier_route": verification.recommended_route.value if verification else "",
        "selection_reasons": "|".join(sample.selection_reasons),
        "review_decision": "",
        "correct_is_formal": "",
        "correct_recommendation_type": "",
        "correct_recommendation_text": "",
        "correct_direction": "",
        "correct_strength": "",
        "correct_certainty": "",
        "correct_population": "",
        "correct_interventions": "",
        "issue_codes": "",
        "review_comment": "",
        "review_origin": "",
        "reviewer_id": "",
    }


def _validate_readonly(
    row: dict[str, str],
    sample: ReviewSample,
    candidates_by_id: dict[str, RecommendationCandidate],
    extractions_by_id: dict[str, ExtractionResult],
    verifications_by_id: dict[str, VerificationResult],
    line_no: int,
) -> None:
    expected = _readonly_row(sample, candidates_by_id, extractions_by_id, verifications_by_id)
    for field in READONLY_FIELDS:
        if (row.get(field) or "") != (expected.get(field) or ""):
            if field == "source_revision_id":
                raise ValueError(f"source_revision_id mismatch at CSV line {line_no}")
            raise ValueError(f"Readonly field {field} changed at CSV line {line_no}")


def _corrected_annotation(row: dict[str, str]) -> dict[str, Any]:
    mapping = {
        "correct_is_formal": "is_formal_recommendation",
        "correct_recommendation_type": "recommendation_type",
        "correct_recommendation_text": "recommendation_text",
        "correct_direction": "direction",
        "correct_strength": "strength",
        "correct_certainty": "certainty",
        "correct_population": "population",
        "correct_interventions": "interventions",
    }
    result: dict[str, Any] = {}
    for csv_field, target in mapping.items():
        value = (row.get(csv_field) or "").strip()
        if not value:
            continue
        result[target] = [item.strip() for item in value.split("|") if item.strip()] if target == "interventions" else value
    return result


def _review_origin(raw: str, default: ReviewOrigin, line_no: int) -> ReviewOrigin:
    value = (raw or "").strip()
    if not value:
        return default
    try:
        origin = ReviewOrigin(value)
    except ValueError as exc:
        raise ValueError(f"Invalid review_origin at CSV line {line_no}") from exc
    if origin != default:
        raise ValueError(f"review_origin does not match import origin at CSV line {line_no}")
    return origin


def _issue_codes(raw: str, line_no: int) -> list[str]:
    values = [item.strip() for item in raw.split("|") if item.strip()]
    valid = {item.value for item in IssueCode}
    unknown = [item for item in values if item not in valid]
    if unknown:
        raise ValueError(f"Unknown issue_codes at CSV line {line_no}: {unknown}")
    return values


FIELD_CSV_FIELDS = [
    "sample_id",
    "field_annotation_id",
    "candidate_id",
    "source_revision_id",
    "field_name",
    "value_original",
    "value_normalized",
    "source_type",
    "span_coordinate_space",
    "quote",
    "span_start",
    "span_end",
    "validation_status",
    "span_issue_code",
    "corrected_start_suggestion",
    "corrected_end_suggestion",
    "correct_span_start",
    "correct_span_end",
    "correct_quote",
    "review_comment",
]


def export_review_fields_csv(
    samples: list[ReviewSample],
    path: str | Path,
    *,
    extractions_by_id: dict[str, ExtractionResult],
    validations_by_extraction_id: dict[str, Any],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELD_CSV_FIELDS)
        writer.writeheader()
        for sample in samples:
            extraction = extractions_by_id.get(sample.extraction_id or "")
            if not extraction:
                continue
            validation = validations_by_extraction_id.get(extraction.extraction_id)
            for field_name, items in extraction.field_evidence.items():
                statuses = (validation.field_statuses.get(field_name) if validation else []) or []
                issues = (validation.field_issue_codes.get(field_name) if validation else []) or []
                corrections = (validation.corrected_spans.get(field_name) if validation else []) or []
                for index, item in enumerate(items):
                    correction = corrections[index] if index < len(corrections) else {}
                    writer.writerow(
                        {
                            "sample_id": sample.sample_id,
                            "field_annotation_id": make_record_id("field", sample.sample_id, field_name, index),
                            "candidate_id": sample.candidate_id,
                            "source_revision_id": sample.source_revision_id,
                            "field_name": field_name,
                            "value_original": item.value_original,
                            "value_normalized": item.value_normalized,
                            "source_type": item.source_type.value,
                            "span_coordinate_space": item.span_coordinate_space.value,
                            "quote": item.quote,
                            "span_start": item.span_start if item.span_start is not None else "",
                            "span_end": item.span_end if item.span_end is not None else "",
                            "validation_status": statuses[index].value if index < len(statuses) else "",
                            "span_issue_code": issues[index].value if index < len(issues) else "",
                            "corrected_start_suggestion": correction.get("corrected_start", ""),
                            "corrected_end_suggestion": correction.get("corrected_end", ""),
                            "correct_span_start": "",
                            "correct_span_end": "",
                            "correct_quote": "",
                            "review_comment": "",
                        }
                    )
