"""Controlled provider response normalization before strict domain validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from src.guideline_information.enums import FailureCode

NORMALIZER_VERSION = "response_normalizer_v2"
TEXT_KEYS = ("value", "text", "value_original", "original_text")
POPULATION_KEYS = ("value", "text", "value_original", "population_text", "original_text")
SCALAR_FIELD_KEYS = {
    "recommendation_text": ("value", "text", "value_original", "recommendation_text", "original_text"),
    "direction": ("value", "text", "value_normalized"),
    "strength": ("value", "text", "value_normalized"),
    "certainty": ("value", "text", "value_normalized"),
    "population": POPULATION_KEYS,
    "dosage": ("value", "text", "value_original", "dosage", "original_text"),
    "duration": ("value", "text", "value_original", "duration", "original_text"),
}


@dataclass(frozen=True)
class NormalizationEvent:
    field_path: str
    original_type: str
    canonical_type: str
    rule_id: str
    status: str
    lossy: bool = False
    message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "field_path": self.field_path,
            "original_type": self.original_type,
            "canonical_type": self.canonical_type,
            "rule_id": self.rule_id,
            "status": self.status,
            "lossy": self.lossy,
            "message": self.message,
        }


@dataclass
class NormalizationResult:
    canonical_payload: dict[str, Any] | None = None
    events: list[NormalizationEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    error_code: FailureCode | None = None
    error_stage: str = ""
    error_field_path: str = ""
    is_lossless: bool = True
    is_ambiguous: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "canonical_payload": self.canonical_payload,
            "events": [event.as_dict() for event in self.events],
            "warnings": self.warnings,
            "errors": self.errors,
            "error_code": self.error_code.value if self.error_code else "",
            "error_stage": self.error_stage,
            "error_field_path": self.error_field_path,
            "is_lossless": self.is_lossless,
            "is_ambiguous": self.is_ambiguous,
        }


class ResponseNormalizer(Protocol):
    def normalize_extraction(self, payload: dict[str, object]) -> NormalizationResult:
        ...

    def normalize_verification(self, payload: dict[str, object]) -> NormalizationResult:
        ...


class ControlledResponseNormalizer:
    def normalize_extraction(self, payload: dict[str, object]) -> NormalizationResult:
        result = NormalizationResult(canonical_payload={})
        if not isinstance(payload, dict):
            return _error("", "wire_response_validation", FailureCode.WIRE_RESPONSE_INVALID, "Top-level extraction payload must be an object")
        out: dict[str, Any] = {}
        out["is_formal_recommendation"] = bool(payload.get("is_formal_recommendation"))
        out["recommendation_type"] = payload.get("recommendation_type") or "UNRESOLVED"
        for field_name in ["recommendation_text", "direction", "strength", "certainty", "population", "dosage", "duration"]:
            normalized = normalize_optional_text_field(payload.get(field_name), field_name=field_name, allowed_object_keys=SCALAR_FIELD_KEYS[field_name])
            _merge(result, normalized)
            if normalized.errors:
                return normalized
            out[field_name] = normalized.canonical_payload[field_name]
        if out["recommendation_text"] is None:
            out["recommendation_text"] = ""
            result.events.append(NormalizationEvent("recommendation_text", "null", "str", "null_recommendation_text_to_empty_string", "CONTROLLED_NORMALIZATION"))
        interventions = normalize_text_list_field(payload.get("interventions"), field_name="interventions")
        _merge(result, interventions)
        if interventions.errors:
            return interventions
        conditions = normalize_text_list_field(payload.get("conditions"), field_name="conditions")
        _merge(result, conditions)
        if conditions.errors:
            return conditions
        evidence = normalize_field_evidence(payload.get("field_evidence"))
        _merge(result, evidence)
        if evidence.errors:
            return evidence
        out["interventions"] = interventions.canonical_payload["interventions"]
        out["conditions"] = conditions.canonical_payload["conditions"]
        out["field_evidence"] = evidence.canonical_payload["field_evidence"]
        result.canonical_payload = out
        return result

    def normalize_verification(self, payload: dict[str, object]) -> NormalizationResult:
        if not isinstance(payload, dict):
            return _error("", "wire_response_validation", FailureCode.WIRE_RESPONSE_INVALID, "Top-level verification payload must be an object")
        result = NormalizationResult(canonical_payload={
            "agrees_is_formal_recommendation": payload.get("agrees_is_formal_recommendation"),
            "field_agreements": _dict_or_empty(payload.get("field_agreements")),
            "field_conflicts": _dict_or_empty(payload.get("field_conflicts")),
            "unsupported_fields": _strings_or_empty(payload.get("unsupported_fields"), "unsupported_fields"),
            "logic_errors": _strings_or_empty(payload.get("logic_errors"), "logic_errors"),
            "recommended_route": payload.get("recommended_route") or "HUMAN_REVIEW",
        })
        return result


def normalize_optional_text_field(value: Any, *, field_name: str, allowed_object_keys: tuple[str, ...]) -> NormalizationResult:
    field_path = field_name
    if value is None:
        return NormalizationResult(canonical_payload={field_name: None})
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return NormalizationResult(
                canonical_payload={field_name: None},
                events=[NormalizationEvent(field_path, "str", "null", "empty_string_to_null", "LOSSLESS_NORMALIZATION")],
            )
        if stripped != value:
            return NormalizationResult(
                canonical_payload={field_name: stripped},
                events=[NormalizationEvent(field_path, "str", "str", "strip_text", "LOSSLESS_NORMALIZATION")],
            )
        return NormalizationResult(canonical_payload={field_name: stripped})
    if isinstance(value, dict):
        texts = []
        for key in allowed_object_keys:
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                texts.append((key, item.strip()))
            elif item not in (None, "") and key in value and not isinstance(item, str):
                return _error(field_path, "text_field_normalization", FailureCode.UNSUPPORTED_RESPONSE_SHAPE, f"{field_name}.{key} must be text")
        if not texts:
            return _error(field_path, "text_field_normalization", FailureCode.RESPONSE_NORMALIZATION_ERROR, f"{field_name} object has no allowed text key")
        normalized_values = {_squash_ws(text) for _, text in texts}
        if len(normalized_values) > 1:
            return _error(field_path, "text_field_normalization", FailureCode.AMBIGUOUS_FIELD_SHAPE, f"{field_name} object has conflicting text values", ambiguous=True)
        selected = texts[0][1]
        return NormalizationResult(
            canonical_payload={field_name: selected},
            events=[NormalizationEvent(field_path, "object", "str", "object_text_key_to_text", "CONTROLLED_NORMALIZATION", message=','.join(key for key, _ in texts))],
            is_lossless=False,
        )
    if isinstance(value, list):
        strings = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        if len(strings) != len([item for item in value if item not in (None, "")]):
            return _error(field_path, "text_field_normalization", FailureCode.RESPONSE_NORMALIZATION_ERROR, f"{field_name} list must contain only strings")
        if not strings:
            return NormalizationResult(canonical_payload={field_name: None})
        if len({_squash_ws(item) for item in strings}) > 1:
            return _error(field_path, "text_field_normalization", FailureCode.AMBIGUOUS_FIELD_SHAPE, f"{field_name} list has multiple different text values", ambiguous=True)
        return NormalizationResult(
            canonical_payload={field_name: strings[0]},
            events=[NormalizationEvent(field_path, "list", "str", "single_or_identical_text_list_to_text", "CONTROLLED_NORMALIZATION")],
            is_lossless=False,
        )
    return _error(field_path, "text_field_normalization", FailureCode.UNSUPPORTED_RESPONSE_SHAPE, f"Unsupported {field_name} shape: {type(value).__name__}")


def normalize_text_list_field(value: Any, *, field_name: str) -> NormalizationResult:
    if value is None:
        return NormalizationResult(canonical_payload={field_name: []})
    if isinstance(value, str):
        stripped = value.strip()
        return NormalizationResult(
            canonical_payload={field_name: [stripped] if stripped else []},
            events=[NormalizationEvent(field_name, "str", "list", "string_to_single_item_list", "LOSSLESS_NORMALIZATION")],
        )
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return NormalizationResult(canonical_payload={field_name: [item.strip() for item in value if item.strip()]})
    return _error(field_name, "list_field_normalization", FailureCode.UNSUPPORTED_RESPONSE_SHAPE, f"{field_name} must be a string list or string")


def normalize_field_evidence(value: Any) -> NormalizationResult:
    if value is None:
        return NormalizationResult(canonical_payload={"field_evidence": {}})
    records: list[tuple[str, dict[str, Any]]] = []
    result = NormalizationResult(canonical_payload={"field_evidence": {}})
    if isinstance(value, dict) and _looks_like_evidence(value):
        field_name = value.get("field_name")
        if not isinstance(field_name, str) or not field_name.strip():
            return _error("field_evidence", "field_evidence_normalization", FailureCode.RESPONSE_NORMALIZATION_ERROR, "Evidence object missing field_name")
        records.append((field_name.strip(), _normalize_evidence_item(_without_field_name(value), result)))
        result.events.append(NormalizationEvent("field_evidence", "object", "dict", "single_evidence_object_to_mapping", "LOSSLESS_NORMALIZATION"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                return _error(f"field_evidence[{index}]", "field_evidence_normalization", FailureCode.RESPONSE_NORMALIZATION_ERROR, "Evidence list item must be object")
            field_name = item.get("field_name")
            if not isinstance(field_name, str) or not field_name.strip():
                return _error(f"field_evidence[{index}]", "field_evidence_normalization", FailureCode.RESPONSE_NORMALIZATION_ERROR, "Evidence list item missing field_name")
            records.append((field_name.strip(), _normalize_evidence_item(_without_field_name(item), result)))
        result.events.append(NormalizationEvent("field_evidence", "list", "dict", "evidence_list_to_mapping", "CONTROLLED_NORMALIZATION"))
        result.is_lossless = False
    elif isinstance(value, dict):
        for field_name, raw_items in value.items():
            if not isinstance(field_name, str) or not field_name.strip():
                return _error("field_evidence", "field_evidence_normalization", FailureCode.RESPONSE_NORMALIZATION_ERROR, "Field evidence mapping key must be text")
            items = raw_items if isinstance(raw_items, list) else [raw_items]
            for index, item in enumerate(items):
                if not isinstance(item, dict):
                    return _error(f"field_evidence.{field_name}[{index}]", "field_evidence_normalization", FailureCode.RESPONSE_NORMALIZATION_ERROR, "Evidence mapping value must be object or list of objects")
                inner = item.get("field_name")
                if isinstance(inner, str) and inner.strip() and inner.strip() != field_name:
                    return _error(f"field_evidence.{field_name}[{index}].field_name", "field_evidence_normalization", FailureCode.RESPONSE_NORMALIZATION_ERROR, "CONFLICTING_FIELD_IDENTITY")
                records.append((field_name, _normalize_evidence_item(_without_field_name(item), result)))
        result.events.append(NormalizationEvent("field_evidence", "object", "dict", "field_keyed_mapping_to_canonical", "LOSSLESS_NORMALIZATION"))
    else:
        return _error("field_evidence", "field_evidence_normalization", FailureCode.UNSUPPORTED_RESPONSE_SHAPE, f"Unsupported field_evidence shape: {type(value).__name__}")
    grouped: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for field_name, item in records:
        key = repr(sorted(item.items()))
        dedupe_key = f"{field_name}:{key}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        grouped.setdefault(field_name, []).append(item)
    result.canonical_payload = {"field_evidence": grouped}
    return result



def _normalize_evidence_item(item: dict[str, Any], result: NormalizationResult) -> dict[str, Any]:
    normalized = dict(item)
    source_type = normalized.get("source_type")
    if isinstance(source_type, str) and source_type.strip().lower() in {"candidate_text", "source_text", "candidate"}:
        normalized["source_type"] = "EXPLICIT_IN_RECOMMENDATION"
        result.events.append(NormalizationEvent("field_evidence.source_type", "str", "str", "candidate_text_source_type_to_explicit_recommendation", "CONTROLLED_NORMALIZATION"))
        result.is_lossless = False
    confidence = normalized.get("confidence_signal")
    if confidence is None:
        normalized["confidence_signal"] = ""
    elif not isinstance(confidence, str):
        normalized["confidence_signal"] = str(confidence)
        result.events.append(NormalizationEvent("field_evidence.confidence_signal", type(confidence).__name__, "str", "confidence_signal_to_string", "CONTROLLED_NORMALIZATION"))
        result.is_lossless = False
    return normalized
def _looks_like_evidence(value: dict[str, Any]) -> bool:
    return any(key in value for key in ["field_name", "quote", "source_type", "span_start", "span_end"])


def _without_field_name(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key != "field_name"}


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _strings_or_empty(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    if isinstance(value, str):
        return [value]
    return [f"invalid_{field_name}_shape"]


def _merge(target: NormalizationResult, source: NormalizationResult) -> None:
    target.events.extend(source.events)
    target.warnings.extend(source.warnings)
    target.errors.extend(source.errors)
    target.is_lossless = target.is_lossless and source.is_lossless
    target.is_ambiguous = target.is_ambiguous or source.is_ambiguous
    if source.error_code:
        target.error_code = source.error_code
        target.error_stage = source.error_stage
        target.error_field_path = source.error_field_path


def _error(field_path: str, stage: str, code: FailureCode, message: str, *, ambiguous: bool = False) -> NormalizationResult:
    return NormalizationResult(
        canonical_payload=None,
        errors=[message],
        error_code=code,
        error_stage=stage,
        error_field_path=field_path,
        is_lossless=False,
        is_ambiguous=ambiguous,
    )


def _squash_ws(text: str) -> str:
    return " ".join(text.split())




