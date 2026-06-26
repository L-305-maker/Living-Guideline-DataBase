from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.pipeline.extraction.common.enums import (
    CERTAINTIES,
    EFFECT_DIRECTIONS,
    EVIDENCE_TYPES,
    GRADE_DOMAIN_JUDGEMENTS,
    PUBLICATION_BIAS_JUDGEMENTS,
    RECOMMENDATION_DIRECTIONS,
    RECOMMENDATION_STRENGTHS,
    normalize_certainty,
    normalize_direction,
    normalize_evidence_type,
    normalize_grade_domain,
    normalize_publication_bias,
    normalize_strength,
)


JsonDict = Dict[str, Any]
BAD_SOURCE_SECTIONS = {"references", "bibliography", "affiliations", "funding", "conflict of interest", "coi"}


@dataclass
class ValidationResult:
    is_valid: bool
    quality_score: int
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        return {"is_valid": self.is_valid, "quality_score": self.quality_score, "warnings": self.warnings}


def _has_text(row: JsonDict, field: str) -> bool:
    return bool(str(row.get(field) or "").strip())


def _source_section(row: JsonDict) -> str:
    return str(row.get("source_section") or "").strip().lower()


def _text_in_source(text: Any, source_span: Any) -> bool:
    needle = " ".join(str(text or "").lower().split())
    haystack = " ".join(str(source_span or "").lower().split())
    return bool(needle and haystack and (needle in haystack or haystack in needle))


def validate_recommendation(row: JsonDict) -> ValidationResult:
    warnings: List[str] = []
    row["direction"] = normalize_direction(row.get("direction"))
    row["strength"] = normalize_strength(row.get("strength"))
    row["certainty"] = normalize_certainty(row.get("certainty"))

    score = 0
    if _has_text(row, "source_span"):
        score += 1
    else:
        warnings.append("missing_source_span")
    if _has_text(row, "recommendation_text"):
        score += 1
    else:
        warnings.append("missing_recommendation_text")
    if row["direction"] in RECOMMENDATION_DIRECTIONS:
        score += 1
    else:
        warnings.append("invalid_direction")
    if row["strength"] in RECOMMENDATION_STRENGTHS:
        score += 1
    else:
        warnings.append("invalid_strength")
    if row["certainty"] in CERTAINTIES:
        score += 1
    else:
        warnings.append("invalid_certainty")
    if _text_in_source(row.get("recommendation_text"), row.get("source_span")):
        score += 1
    else:
        warnings.append("recommendation_text_not_in_source_span")
    if not (row.get("recommendation_id") or row.get("recommendation_version_id")):
        warnings.append("missing_recommendation_id")
    if _source_section(row) in BAD_SOURCE_SECTIONS:
        warnings.append("bad_source_section")
    return ValidationResult(score >= 5 and "bad_source_section" not in warnings, score, warnings)


def validate_pico(row: JsonDict) -> ValidationResult:
    warnings: List[str] = []
    score = 0
    if _has_text(row, "population"):
        score += 1
    else:
        warnings.append("missing_population")
    if _has_text(row, "intervention"):
        score += 1
    else:
        warnings.append("missing_intervention")
    if "comparator" in row:
        score += 1
    else:
        warnings.append("missing_comparator_marker")
    if isinstance(row.get("outcomes"), list):
        score += 1
    else:
        warnings.append("outcomes_not_list")
    if _has_text(row, "source_span"):
        score += 1
    else:
        warnings.append("missing_source_span")
    if row.get("recommendation_id") or row.get("source_record_id"):
        score += 1
    else:
        warnings.append("missing_link")
    if _source_section(row) in BAD_SOURCE_SECTIONS:
        warnings.append("bad_source_section")
    return ValidationResult(score >= 4 and "bad_source_section" not in warnings, score, warnings)


def validate_evidence_item(row: JsonDict) -> ValidationResult:
    warnings: List[str] = []
    row["evidence_type"] = normalize_evidence_type(row.get("evidence_type"))
    if row.get("effect_direction") not in EFFECT_DIRECTIONS:
        row["effect_direction"] = "uncertain"

    score = 0
    if _has_text(row, "source_span"):
        score += 1
    else:
        warnings.append("missing_source_span")
    if row["evidence_type"] in EVIDENCE_TYPES:
        score += 1
    else:
        warnings.append("invalid_evidence_type")
    if row.get("recommendation_id") or row.get("pico_id"):
        score += 1
    else:
        warnings.append("missing_link")
    if row.get("benefits") or row.get("harms") or row.get("outcomes_extracted"):
        score += 1
    else:
        warnings.append("missing_evidence_content")
    if not row.get("fabricated_fields"):
        score += 1
    else:
        warnings.append("fabricated_quantitative_fields")
    if len(str(row.get("source_span") or "")) > 2500:
        warnings.append("source_span_too_long")
    if _source_section(row) in BAD_SOURCE_SECTIONS:
        warnings.append("bad_source_section")
    return ValidationResult(score >= 4 and not {"bad_source_section", "source_span_too_long"} & set(warnings), score, warnings)


def validate_grade_assessment(row: JsonDict) -> ValidationResult:
    warnings: List[str] = []
    row["final_certainty"] = normalize_certainty(row.get("final_certainty"))
    for field_name in ["risk_of_bias", "inconsistency", "indirectness", "imprecision"]:
        row[field_name] = normalize_grade_domain(row.get(field_name))
    row["publication_bias"] = normalize_publication_bias(row.get("publication_bias"))

    score = 0
    if row["final_certainty"] in CERTAINTIES:
        score += 1
    else:
        warnings.append("invalid_final_certainty")
    if _has_text(row, "judgement_rationale"):
        score += 1
    else:
        warnings.append("missing_judgement_rationale")
    if _has_text(row, "source_span"):
        score += 1
    else:
        warnings.append("missing_source_span")
    domain_values = [row[field] for field in ["risk_of_bias", "inconsistency", "indirectness", "imprecision", "publication_bias"]]
    if any(value not in {"not_reported", "unclear"} for value in domain_values):
        score += 1
    else:
        warnings.append("no_explicit_grade_domain")
    if all(value in GRADE_DOMAIN_JUDGEMENTS for value in domain_values[:4]) and row["publication_bias"] in PUBLICATION_BIAS_JUDGEMENTS:
        score += 1
    else:
        warnings.append("invalid_grade_domain")
    if any(value == "no_concern" for value in domain_values) and not str(row.get("source_span") or "").lower().count("no serious"):
        warnings.append("unjustified_no_concern")
    return ValidationResult(score >= 4 and "unjustified_no_concern" not in warnings, score, warnings)
