"""End-to-end pilot pipeline using existing evidence sections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.guideline_information.enums import FailureCode, RecommendedRoute, RecommendationType
from src.guideline_information.extraction.adapters import RecommendationExtractorAdapter, RecommendationVerifierAdapter
from src.guideline_information.extraction.clients.base import (
    ModelConfigurationError,
    ModelFormatError,
    ModelNetworkError,
    ModelRateLimitError,
    StructuredModelClient,
)
from src.guideline_information.extraction.clients.fake import FakeStructuredModelClient
from src.guideline_information.extraction.clients.openai_compatible import OpenAICompatibleStructuredModelClient
from src.guideline_information.extraction.candidate_builder import CandidateBuilder
from src.guideline_information.extraction.failures import AdapterFailure
from src.guideline_information.extraction.router import route_candidate
from src.guideline_information.extraction.span_validator import EvidenceSpanValidator
from src.guideline_information.ids import make_extraction_id, make_verification_id
from src.guideline_information.models import ExtractionResult, RecommendationCandidate, RouteResult, SCHEMA_VERSION, ValidationResult, VerificationResult
from src.guideline_information.paths import DEFAULT_INFORMATION_DIR, InformationRunPaths, PilotPaths
from src.guideline_information.repository import append_model, read_models, write_models
from src.guideline_information.review.import_export import export_review_fields_csv, export_review_samples_csv
from src.guideline_information.review.selectors import (
    ExtractorVerifierDisagreementSelector,
    HardNegativeSelector,
    RandomAuditSelector,
    UncertaintySelector,
    build_review_samples,
)
from src.utils.io import append_jsonl, read_jsonl


def make_client(client_name: str) -> StructuredModelClient:
    if client_name == "fake":
        return FakeStructuredModelClient()
    if client_name == "openai-compatible":
        return OpenAICompatibleStructuredModelClient()
    raise ValueError(f"Unknown client: {client_name}")


def run_pilot(
    *,
    pilot_id: str,
    run_id: str,
    output_root: str | Path = DEFAULT_INFORMATION_DIR,
    client_name: str = "fake",
    max_items: int | None = None,
    max_model_calls: int | None = None,
    input_candidates_path: str | Path | None = None,
    experiment_id: str = "",
    resume: bool = False,
    force: bool = False,
    dry_run: bool = False,
    fail_fast: bool = False,
    random_audit_rate: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    pilot_paths = PilotPaths.create(output_root, pilot_id)
    run_paths = InformationRunPaths.ensure(output_root, run_id)
    if input_candidates_path:
        candidates = read_models(input_candidates_path, RecommendationCandidate)
    else:
        sections = list(read_jsonl(pilot_paths.selected_sections))
        candidates = CandidateBuilder().build(sections, run_id)
    if max_items is not None:
        candidates = candidates[:max_items]
    if dry_run:
        return {"dry_run": True, "candidate_count": len(candidates), "run_dir": str(run_paths.run_dir)}
    write_models(run_paths.candidates, candidates)
    client = make_client(client_name)
    extractor = RecommendationExtractorAdapter(client)
    verifier = RecommendationVerifierAdapter(client)
    validator = EvidenceSpanValidator()
    existing_extractions = _by_id(run_paths.extraction_results, ExtractionResult, "extraction_id") if resume and not force else {}
    existing_verifications = _by_id(run_paths.verification_results, VerificationResult, "verification_id") if resume and not force else {}
    extraction_results: list[ExtractionResult] = list(existing_extractions.values()) if resume and not force else []
    verification_results: list[VerificationResult] = list(existing_verifications.values()) if resume and not force else []
    validation_results: list[ValidationResult] = []
    route_results: list[RouteResult] = []
    extraction_by_candidate: dict[str, ExtractionResult] = {item.candidate_id: item for item in extraction_results}
    verification_by_extraction: dict[str, VerificationResult] = {item.extraction_id: item for item in verification_results}
    counts = {
        "candidates": len(candidates),
        "extracted": 0,
        "extraction_skipped": 0,
        "extraction_failed": 0,
        "verified": 0,
        "verification_skipped": 0,
        "verification_failed": 0,
        "verifier_not_run_due_to_extraction_failure": 0,
        "model_calls_limited": 0,
        "pre_routed_reference": 0,
    }
    model_calls = 0
    for index, candidate in enumerate(candidates, start=1):
        short_id = candidate.candidate_id[-8:]
        extraction = extraction_by_candidate.get(candidate.candidate_id)
        if extraction:
            counts["extraction_skipped"] += 1
        else:
            if _is_reference_candidate(candidate):
                extraction, verification = _reference_skip_records(candidate)
                append_model(run_paths.extraction_results, extraction)
                append_model(run_paths.verification_results, verification)
                extraction_results.append(extraction)
                verification_results.append(verification)
                extraction_by_candidate[candidate.candidate_id] = extraction
                verification_by_extraction[extraction.extraction_id] = verification
                counts["extracted"] += 1
                counts["verified"] += 1
                counts["pre_routed_reference"] += 1
                print(f"[{index}/{len(candidates)}] reference pre-route reject candidate={short_id}", flush=True)
            elif max_model_calls is not None and model_calls >= max_model_calls:
                counts["model_calls_limited"] += 1
                continue
            else:
                print(f"[{index}/{len(candidates)}] extractor start candidate={short_id}", flush=True)
                try:
                    model_calls += 1
                    extraction = extractor.extract(candidate)
                    _append_adapter_artifacts(run_paths, extractor)
                    append_model(run_paths.extraction_results, extraction)
                    extraction_results.append(extraction)
                    extraction_by_candidate[candidate.candidate_id] = extraction
                    counts["extracted"] += 1
                    print(f"[{index}/{len(candidates)}] extractor success latency_ms={extraction.latency_ms}", flush=True)
                except Exception as exc:
                    counts["extraction_failed"] += 1
                    counts["verifier_not_run_due_to_extraction_failure"] += 1
                    _append_failure(run_paths, candidate.candidate_id, exc, operation="extractor")
                    print(f"[{index}/{len(candidates)}] extractor failed code={_failure_code(exc).value}", flush=True)
                    if fail_fast:
                        raise
                    continue
        verification = verification_by_extraction.get(extraction.extraction_id)
        if verification:
            counts["verification_skipped"] += 1
        else:
            if max_model_calls is not None and model_calls >= max_model_calls:
                counts["model_calls_limited"] += 1
                continue
            print(f"[{index}/{len(candidates)}] verifier start candidate={short_id}", flush=True)
            try:
                model_calls += 1
                verification = verifier.verify(candidate, extraction)
                _append_adapter_artifacts(run_paths, verifier)
                append_model(run_paths.verification_results, verification)
                verification_results.append(verification)
                verification_by_extraction[extraction.extraction_id] = verification
                counts["verified"] += 1
                print(f"[{index}/{len(candidates)}] verifier success latency_ms={verification.latency_ms}", flush=True)
            except Exception as exc:
                counts["verification_failed"] += 1
                _append_failure(run_paths, candidate.candidate_id, exc, operation="verifier", extraction_id=extraction.extraction_id)
                print(f"[{index}/{len(candidates)}] verifier failed code={_failure_code(exc).value}", flush=True)
                if fail_fast:
                    raise
                continue
        try:
            validation = validator.validate(candidate, extraction)
        except Exception as exc:
            _append_failure(run_paths, candidate.candidate_id, exc, operation="span_validator", extraction_id=extraction.extraction_id)
            if fail_fast:
                raise
            continue
        route = route_candidate(candidate, extraction, verification, validation, random_audit_rate=random_audit_rate, seed=seed)
        append_model(run_paths.validation_results, validation)
        append_model(run_paths.route_results, route)
        validation_results.append(validation)
        route_results.append(route)
    # Re-write compact current state at normal completion; append calls above preserve partial progress on interruption.
    write_models(run_paths.extraction_results, extraction_results)
    write_models(run_paths.verification_results, verification_results)
    write_models(run_paths.validation_results, validation_results)
    write_models(run_paths.route_results, route_results)
    review_candidates = [candidate for candidate in candidates if _route_for(candidate, route_results) in {"HUMAN_REVIEW", "RANDOM_AUDIT"}]
    samples, sample_manifest = build_review_samples(
        candidates=review_candidates,
        selectors=[ExtractorVerifierDisagreementSelector(), UncertaintySelector(), HardNegativeSelector(), RandomAuditSelector()],
        budgets={
            "ExtractorVerifierDisagreementSelector": len(review_candidates),
            "UncertaintySelector": len(review_candidates),
            "HardNegativeSelector": len(review_candidates),
            "RandomAuditSelector": len(review_candidates),
        },
        sample_batch_id=f"{run_id}-review",
        seed=seed,
        verification_by_candidate={item.candidate_id: item for item in verification_results},
    )
    _attach_stage_ids(samples, extraction_by_candidate, verification_by_extraction)
    write_models(run_paths.review_samples, samples)
    candidate_map = {item.candidate_id: item for item in candidates}
    extraction_map = {item.extraction_id: item for item in extraction_results}
    verification_map = {item.verification_id: item for item in verification_results}
    validation_map = {item.extraction_id: item for item in validation_results}
    export_review_samples_csv(samples, run_paths.review_csv, candidates_by_id=candidate_map, extractions_by_id=extraction_map, verifications_by_id=verification_map)
    export_review_fields_csv(samples, run_paths.review_fields_csv, extractions_by_id=extraction_map, validations_by_extraction_id=validation_map)
    run_manifest = {
        "run_id": run_id,
        "pilot_id": pilot_id,
        "client": client_name,
        "experiment_id": experiment_id,
        "input_candidates_path": str(input_candidates_path) if input_candidates_path else "",
        "network_called": client_name != "fake",
        "counts": counts,
        "model_calls": model_calls,
        "routes": _route_counts(route_results),
        "artifacts": {"run_dir": str(run_paths.run_dir), "review_samples_csv": str(run_paths.review_csv), "review_fields_csv": str(run_paths.review_fields_csv)},
    }
    run_paths.manifest.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    run_paths.sample_manifest.write_text(json.dumps(sample_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return run_manifest


def _is_reference_candidate(candidate: RecommendationCandidate) -> bool:
    profile = candidate.candidate_profile or {}
    signals = {str(item).lower() for item in candidate.candidate_signals or []}
    section = " > ".join(candidate.section_path).lower()
    return profile.get("block_type") == "reference" or "reference" in signals or section.endswith("references")


def _reference_skip_records(candidate: RecommendationCandidate) -> tuple[ExtractionResult, VerificationResult]:
    extractor_name = "deterministic_reference_skip"
    verifier_name = "deterministic_reference_skip"
    prompt_version = "reference_skip_v1"
    model_name = "deterministic"
    extraction_id = make_extraction_id(candidate.candidate_id, extractor_name, model_name, prompt_version, SCHEMA_VERSION)
    verification_id = make_verification_id(extraction_id, verifier_name, model_name, prompt_version, SCHEMA_VERSION)
    extraction = ExtractionResult(
        extraction_id=extraction_id,
        candidate_id=candidate.candidate_id,
        extractor_name=extractor_name,
        model_name=model_name,
        model_version=model_name,
        prompt_version=prompt_version,
        is_formal_recommendation=False,
        recommendation_type=RecommendationType.NOT_RECOMMENDATION,
        recommendation_text="",
        field_evidence={},
        model_request_id=extraction_id,
        raw_response_text='{"reason":"reference_candidate_pre_route"}',
    )
    verification = VerificationResult(
        verification_id=verification_id,
        extraction_id=extraction_id,
        candidate_id=candidate.candidate_id,
        verifier_name=verifier_name,
        verifier_version=prompt_version,
        model_name=model_name,
        prompt_version=prompt_version,
        agrees_is_formal_recommendation=True,
        recommended_route=RecommendedRoute.REJECT,
        model_request_id=verification_id,
        raw_response_text='{"reason":"reference_candidate_pre_route"}',
    )
    return extraction, verification

def _by_id(path: Path, model, key: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    return {getattr(item, key): item for item in read_models(path, model)}


def _append_adapter_artifacts(run_paths: InformationRunPaths, adapter) -> None:
    if getattr(adapter, "last_response_record", None):
        append_jsonl(run_paths.model_responses, adapter.last_response_record)
    if getattr(adapter, "last_normalization_record", None):
        append_jsonl(run_paths.normalization_results, adapter.last_normalization_record)
    if getattr(adapter, "last_canonical_validation_record", None):
        append_jsonl(run_paths.canonical_validation_results, adapter.last_canonical_validation_record)


def _append_failure(run_paths: InformationRunPaths, candidate_id: str, exc: Exception, *, operation: str, extraction_id: str = "") -> None:
    if isinstance(exc, AdapterFailure):
        if exc.response_record:
            append_jsonl(run_paths.model_responses, exc.response_record)
        if exc.normalization_record:
            append_jsonl(run_paths.normalization_results, exc.normalization_record)
        if exc.canonical_validation_record:
            append_jsonl(run_paths.canonical_validation_results, exc.canonical_validation_record)
        record = exc.failure_record(candidate_id=candidate_id, extraction_id=extraction_id)
    else:
        record = {
            "candidate_id": candidate_id,
            "extraction_id": extraction_id,
            "failure_code": _failure_code(exc).value,
            "exception_type": type(exc).__name__,
            "stage": operation,
            "field_path": "",
            "retryable": _failure_code(exc) in {FailureCode.NETWORK_TIMEOUT, FailureCode.NETWORK_ERROR, FailureCode.RATE_LIMIT},
            "error": _safe_error_message(exc),
        }
    record["operation"] = operation
    target = run_paths.verification_failures if operation == "verifier" else run_paths.extraction_failures
    append_jsonl(target, record)


def _failure_code(exc: Exception) -> FailureCode:
    if isinstance(exc, AdapterFailure):
        return exc.failure_code
    if isinstance(exc, ModelRateLimitError):
        return FailureCode.RATE_LIMIT
    if isinstance(exc, ModelConfigurationError):
        return FailureCode.MODEL_UNAVAILABLE
    if isinstance(exc, ModelFormatError):
        return FailureCode.JSON_PARSE_ERROR
    if isinstance(exc, ModelNetworkError):
        message = str(exc).lower()
        return FailureCode.NETWORK_TIMEOUT if "timeout" in message or "timed out" in message else FailureCode.NETWORK_ERROR
    return FailureCode.UNEXPECTED_ERROR


def _safe_error_message(exc: Exception) -> str:
    text = str(exc)
    for marker in ["Authorization", "Bearer", "API_KEY"]:
        text = text.replace(marker, "[REDACTED]")
    return text[:1000]


def _route_for(candidate: RecommendationCandidate, routes: list[RouteResult]) -> str:
    for route in routes:
        if route.candidate_id == candidate.candidate_id:
            return route.route.value
    return ""


def _route_counts(routes: list[RouteResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for route in routes:
        counts[route.route.value] = counts.get(route.route.value, 0) + 1
    return counts


def _attach_stage_ids(samples, extraction_by_candidate, verification_by_extraction) -> None:
    for index, sample in enumerate(samples):
        extraction = extraction_by_candidate.get(sample.candidate_id)
        verification = verification_by_extraction.get(extraction.extraction_id) if extraction else None
        samples[index] = sample.model_copy(update={"extraction_id": extraction.extraction_id if extraction else None, "verification_id": verification.verification_id if verification else None})
