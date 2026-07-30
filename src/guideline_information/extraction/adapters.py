"""Recommendation extractor and verifier adapters."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import ValidationError

from src.guideline_information.enums import FailureCode, RecommendedRoute
from src.guideline_information.extraction.clients.base import ModelResponse, StructuredModelClient
from src.guideline_information.extraction.failures import AdapterFailure
from src.guideline_information.extraction.prompt_loader import load_prompt
from src.guideline_information.extraction.response_normalizer import ControlledResponseNormalizer, NormalizationResult
from src.guideline_information.extraction.schemas import EXTRACTION_SCHEMA, VERIFICATION_SCHEMA
from src.guideline_information.ids import make_extraction_id, make_verification_id
from src.guideline_information.models import ExtractionResult, RecommendationCandidate, SCHEMA_VERSION, VerificationResult

ADAPTER_VERSION = "adapter_v2"


class RecommendationExtractorAdapter:
    def __init__(self, client: StructuredModelClient, *, prompt_version: str = "extraction_v1") -> None:
        self.client = client
        self.prompt_version = prompt_version
        self.extractor_name = prompt_version
        self.normalizer = ControlledResponseNormalizer()
        self.last_response_record: dict[str, Any] = {}
        self.last_normalization_record: dict[str, Any] = {}
        self.last_canonical_validation_record: dict[str, Any] = {}

    def extract(self, candidate: RecommendationCandidate) -> ExtractionResult:
        request_id = make_extraction_id(candidate.candidate_id, self.extractor_name, _client_model(self.client), self.prompt_version, SCHEMA_VERSION)
        response = self.client.generate_json(
            system_prompt=load_prompt(self.prompt_version),
            user_prompt=_candidate_prompt(candidate),
            schema=EXTRACTION_SCHEMA,
            request_id=request_id,
        )
        self.last_response_record = _response_record(response, candidate_id=candidate.candidate_id, operation="extractor", prompt_version=self.prompt_version, schema=EXTRACTION_SCHEMA)
        _ensure_extraction_wire_anchor(response.parsed_json, response_record=self.last_response_record)
        normalized = self.normalizer.normalize_extraction(response.parsed_json)
        _inject_candidate_evidence_defaults(normalized, candidate)
        self.last_normalization_record = _normalization_record(request_id, candidate.candidate_id, "extractor", normalized)
        if normalized.errors or normalized.canonical_payload is None:
            raise AdapterFailure(
                "; ".join(normalized.errors) or "response normalization failed",
                failure_code=normalized.error_code or FailureCode.RESPONSE_NORMALIZATION_ERROR,
                stage=normalized.error_stage or "response_normalization",
                field_path=normalized.error_field_path,
                exception_type="ResponseNormalizationError",
                response_record=self.last_response_record,
                normalization_record=self.last_normalization_record,
            )
        try:
            result = ExtractionResult(
                extraction_id=request_id,
                candidate_id=candidate.candidate_id,
                extractor_name=self.extractor_name,
                model_name=response.model_name,
                model_version=response.model_name,
                prompt_version=self.prompt_version,
                model_request_id=response.request_id,
                raw_response_text=response.raw_text,
                parsed_with_repair=response.parsed_with_repair,
                usage=response.usage,
                latency_ms=response.latency_ms,
                **normalized.canonical_payload,
            )
        except ValidationError as exc:
            self.last_canonical_validation_record = _validation_record(request_id, candidate.candidate_id, "extractor", False, exc)
            raise AdapterFailure(
                "canonical extraction validation failed",
                failure_code=FailureCode.CANONICAL_SCHEMA_VALIDATION_ERROR,
                stage="canonical_pydantic_validation",
                field_path=_first_error_path(exc),
                exception_type="ValidationError",
                response_record=self.last_response_record,
                normalization_record=self.last_normalization_record,
                canonical_validation_record=self.last_canonical_validation_record,
            ) from exc
        self.last_canonical_validation_record = _validation_record(request_id, candidate.candidate_id, "extractor", True, None)
        return result


class RecommendationVerifierAdapter:
    def __init__(self, client: StructuredModelClient, *, prompt_version: str = "verification_v1") -> None:
        self.client = client
        self.prompt_version = prompt_version
        self.verifier_name = prompt_version
        self.normalizer = ControlledResponseNormalizer()
        self.last_response_record: dict[str, Any] = {}
        self.last_normalization_record: dict[str, Any] = {}
        self.last_canonical_validation_record: dict[str, Any] = {}

    def verify(self, candidate: RecommendationCandidate, extraction: ExtractionResult) -> VerificationResult:
        request_id = make_verification_id(extraction.extraction_id, self.verifier_name, _client_model(self.client), self.prompt_version, SCHEMA_VERSION)
        response = self.client.generate_json(
            system_prompt=load_prompt(self.prompt_version),
            user_prompt=_verification_prompt(candidate, extraction),
            schema=VERIFICATION_SCHEMA,
            request_id=request_id,
        )
        self.last_response_record = _response_record(response, candidate_id=candidate.candidate_id, operation="verifier", prompt_version=self.prompt_version, schema=VERIFICATION_SCHEMA)
        normalized = self.normalizer.normalize_verification(response.parsed_json)
        self.last_normalization_record = _normalization_record(request_id, candidate.candidate_id, "verifier", normalized)
        if normalized.errors or normalized.canonical_payload is None:
            raise AdapterFailure(
                "; ".join(normalized.errors) or "response normalization failed",
                failure_code=normalized.error_code or FailureCode.RESPONSE_NORMALIZATION_ERROR,
                stage=normalized.error_stage or "response_normalization",
                field_path=normalized.error_field_path,
                exception_type="ResponseNormalizationError",
                response_record=self.last_response_record,
                normalization_record=self.last_normalization_record,
            )
        parsed = normalized.canonical_payload
        try:
            result = VerificationResult(
                verification_id=request_id,
                extraction_id=extraction.extraction_id,
                candidate_id=candidate.candidate_id,
                verifier_name=self.verifier_name,
                verifier_version=self.prompt_version,
                model_name=response.model_name,
                prompt_version=self.prompt_version,
                agrees_is_formal_recommendation=parsed.get("agrees_is_formal_recommendation"),
                field_agreements=dict(parsed.get("field_agreements") or {}),
                field_conflicts=dict(parsed.get("field_conflicts") or {}),
                unsupported_fields=[str(item) for item in parsed.get("unsupported_fields") or []],
                logic_errors=[str(item) for item in parsed.get("logic_errors") or []],
                recommended_route=RecommendedRoute(parsed.get("recommended_route") or RecommendedRoute.HUMAN_REVIEW.value),
                model_request_id=response.request_id,
                raw_response_text=response.raw_text,
                parsed_with_repair=response.parsed_with_repair,
                usage=response.usage,
                latency_ms=response.latency_ms,
            )
        except ValidationError as exc:
            self.last_canonical_validation_record = _validation_record(request_id, candidate.candidate_id, "verifier", False, exc)
            raise AdapterFailure(
                "canonical verification validation failed",
                failure_code=FailureCode.CANONICAL_SCHEMA_VALIDATION_ERROR,
                stage="canonical_pydantic_validation",
                field_path=_first_error_path(exc),
                exception_type="ValidationError",
                response_record=self.last_response_record,
                normalization_record=self.last_normalization_record,
                canonical_validation_record=self.last_canonical_validation_record,
            ) from exc
        self.last_canonical_validation_record = _validation_record(request_id, candidate.candidate_id, "verifier", True, None)
        return result


def _candidate_prompt(candidate: RecommendationCandidate) -> str:
    return "\n".join(
        [
            f"source_block_key: {candidate.source_block_key}",
            f"source_revision_id: {candidate.source_revision_id}",
            f"section_path: {' > '.join(candidate.section_path)}",
            f"Context before: {candidate.context_before}",
            f"Candidate text: {candidate.candidate_text}",
            f"Context after: {candidate.context_after}",
        ]
    )


def _verification_prompt(candidate: RecommendationCandidate, extraction: ExtractionResult) -> str:
    return "\n".join(
        [
            "Verification task",
            f"Candidate text: {candidate.candidate_text}",
            "Extraction JSON:",
            json.dumps(extraction.model_dump(mode="json"), ensure_ascii=False),
        ]
    )


def _ensure_extraction_wire_anchor(payload: dict[str, object], *, response_record: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        return
    if "is_formal_recommendation" not in payload:
        keys = ", ".join(sorted(str(key) for key in payload.keys())[:12]) or "<none>"
        raise AdapterFailure(
            f"model extraction response missing required key is_formal_recommendation; observed keys: {keys}",
            failure_code=FailureCode.WIRE_RESPONSE_INVALID,
            stage="wire_response_validation",
            field_path="is_formal_recommendation",
            retryable=True,
            exception_type="WireResponseValidationError",
            response_record=response_record,
        )
    if not isinstance(payload["is_formal_recommendation"], bool):
        raise AdapterFailure(
            "model extraction response field is_formal_recommendation must be boolean",
            failure_code=FailureCode.WIRE_RESPONSE_INVALID,
            stage="wire_response_validation",
            field_path="is_formal_recommendation",
            retryable=True,
            exception_type="WireResponseValidationError",
            response_record=response_record,
        )

def _inject_candidate_evidence_defaults(normalized: NormalizationResult, candidate: RecommendationCandidate) -> None:
    if not normalized.canonical_payload:
        return
    field_evidence = normalized.canonical_payload.get("field_evidence") or {}
    if not isinstance(field_evidence, dict):
        return
    for items in field_evidence.values():
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                item.setdefault("source_block_key", candidate.source_block_key)
                item.setdefault("source_revision_id", candidate.source_revision_id)


def _response_record(response: ModelResponse, *, candidate_id: str, operation: str, prompt_version: str, schema: dict[str, object]) -> dict[str, Any]:
    return {
        "request_id": response.request_id,
        "candidate_id": candidate_id,
        "operation": operation,
        "provider": response.provider,
        "model_name": response.model_name,
        "prompt_version": prompt_version,
        "prompt_hash": _hash_text(load_prompt(prompt_version)),
        "schema_version": SCHEMA_VERSION,
        "schema_hash": _hash_json(schema),
        "raw_response_hash": _hash_text(response.raw_text),
        "parsed_payload": _truncate(response.parsed_json),
        "parsed_with_repair": response.parsed_with_repair,
        "latency_ms": response.latency_ms,
        "input_tokens": int(response.usage.get("input_tokens") or response.usage.get("prompt_tokens") or 0),
        "output_tokens": int(response.usage.get("output_tokens") or response.usage.get("completion_tokens") or 0),
        "finish_reason": response.finish_reason,
        "retry_count": response.response_metadata.get("retry_count", 0),
        "network_called": response.provider != "fake",
    }


def _normalization_record(request_id: str, candidate_id: str, operation: str, result: NormalizationResult) -> dict[str, Any]:
    payload = result.as_dict()
    payload.update({"request_id": request_id, "candidate_id": candidate_id, "operation": operation})
    payload["canonical_payload"] = _truncate(payload.get("canonical_payload"))
    return payload


def _validation_record(request_id: str, candidate_id: str, operation: str, ok: bool, exc: ValidationError | None) -> dict[str, Any]:
    errors = []
    if exc is not None:
        for error in exc.errors():
            errors.append({"loc": [str(item) for item in error.get("loc", [])], "type": error.get("type"), "msg": error.get("msg")})
    return {"request_id": request_id, "candidate_id": candidate_id, "operation": operation, "canonical_validation_status": "PASS" if ok else "FAIL", "errors": errors}


def _first_error_path(exc: ValidationError) -> str:
    first = exc.errors()[0] if exc.errors() else {}
    return ".".join(str(item) for item in first.get("loc", []))


def _client_model(client: StructuredModelClient) -> str:
    return str(getattr(client, "model_name", "structured-model") or "structured-model")


def _hash_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _hash_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _truncate(value: Any, *, max_text: int = 800) -> Any:
    if isinstance(value, str):
        if len(value) <= max_text:
            return value
        return {"text": value[:max_text], "truncated": True, "original_length": len(value)}
    if isinstance(value, list):
        return [_truncate(item, max_text=max_text) for item in value[:20]] + ([{"truncated": True, "remaining_items": len(value) - 20}] if len(value) > 20 else [])
    if isinstance(value, dict):
        return {str(key): _truncate(item, max_text=max_text) for key, item in list(value.items())[:50]}
    return value
