"""V0.1 human review loop for real guideline information extraction runs."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.guideline_information.enums import AnnotationTier, IssueCode, ReviewDecisionType, ReviewOrigin, SpanIssueCode, SpanSupportStatus
from src.guideline_information.extraction.response_normalizer import NORMALIZER_VERSION
from src.guideline_information.ids import make_record_id
from src.guideline_information.models import ExtractionResult, RecommendationCandidate, RouteResult, ValidationResult, VerificationResult
from src.guideline_information.repository import read_models
from src.utils.ids import sha256_file
from src.utils.io import DATA_DIR, read_jsonl, write_jsonl

DEFAULT_EXPERIMENT_ROOT = DATA_DIR.parent / "guideline_information" / "experiments"
DEFAULT_REVIEW_ROOT = DATA_DIR.parent / "guideline_information" / "review_batches"

SAMPLE_READONLY_FIELDS = [
    "sample_id", "candidate_id", "doc_id", "title", "section_path", "source_revision_id",
    "candidate_text", "context_before", "context_after", "model_is_formal",
    "model_recommendation_type", "model_recommendation_text", "model_direction",
    "model_strength", "model_certainty", "model_population", "model_interventions",
    "model_dosage", "model_duration", "verifier_agreement", "validation_summary",
]
SAMPLE_REVIEW_FIELDS = [
    "reviewer_id", "review_decision", "correct_is_formal", "correct_recommendation_type",
    "correct_recommendation_text", "correct_direction", "correct_strength", "correct_certainty",
    "correct_population", "correct_interventions", "correct_dosage", "correct_duration",
    "issue_codes", "review_comment",
]
SAMPLE_FIELDS = SAMPLE_READONLY_FIELDS + SAMPLE_REVIEW_FIELDS
FIELD_READONLY_FIELDS = [
    "sample_id", "field_annotation_id", "candidate_id", "field_name", "model_value_original",
    "model_value_normalized", "source_type", "span_coordinate_space", "quote", "span_start",
    "span_end", "validation_status", "source_text_excerpt",
]
FIELD_REVIEW_FIELDS = [
    "reviewer_id", "field_decision", "correct_value_original", "correct_value_normalized",
    "correct_source_type", "correct_quote", "correct_span_start", "correct_span_end",
    "span_issue_code", "review_comment",
]
FIELD_FIELDS = FIELD_READONLY_FIELDS + FIELD_REVIEW_FIELDS
SILVER_FIELDS = ["recommendation_text", "recommendation_type", "direction", "strength", "certainty", "population", "interventions", "dosage", "duration"]
CORRECTED_SAMPLE_MAP = {
    "correct_is_formal": "is_formal_recommendation",
    "correct_recommendation_type": "recommendation_type",
    "correct_recommendation_text": "recommendation_text",
    "correct_direction": "direction",
    "correct_strength": "strength",
    "correct_certainty": "certainty",
    "correct_population": "population",
    "correct_interventions": "interventions",
    "correct_dosage": "dosage",
    "correct_duration": "duration",
}


class IncompleteStageAArtifacts(ValueError):
    status = "BLOCKED_INCOMPLETE_STAGE_A_ARTIFACTS"


def load_stage_a(*, experiment_id: str, experiment_root: str | Path = DEFAULT_EXPERIMENT_ROOT) -> dict[str, Any]:
    exp_dir = Path(experiment_root) / experiment_id
    manifest = _read_json(exp_dir / "experiment_manifest.json")
    gate = _read_json(exp_dir / "stage_a_technical_gate.json")
    artifacts = gate.get("artifacts", {})
    run_dirs = [Path(item) for item in [artifacts.get("canary_run_dir"), artifacts.get("full_run_dir")] if item]
    if not run_dirs:
        raise IncompleteStageAArtifacts("Missing Stage A run directories in technical gate")
    stage = {
        "experiment_id": experiment_id,
        "experiment_dir": exp_dir,
        "experiment_manifest": manifest,
        "gate": gate,
        "run_dirs": run_dirs,
        "run_ids": [path.name for path in run_dirs],
        "candidates": _read_model_many(run_dirs, "candidates.jsonl", RecommendationCandidate),
        "extractions": _read_model_many(run_dirs, "extraction_results.jsonl", ExtractionResult),
        "verifications": _read_model_many(run_dirs, "verification_results.jsonl", VerificationResult),
        "validations": _read_model_many(run_dirs, "validation_results.jsonl", ValidationResult),
        "routes": _read_model_many(run_dirs, "route_results.jsonl", RouteResult),
        "model_responses": _read_jsonl_many(run_dirs, "model_responses.jsonl"),
        "normalizations": _read_jsonl_many(run_dirs, "normalization_results.jsonl"),
        "provider": "openai-compatible",
    }
    stage["model_name"] = manifest.get("model_name") or _first_attr(stage["extractions"], "model_name")
    stage["prompt_versions"] = {"extractor": _first_attr(stage["extractions"], "prompt_version"), "verifier": _first_attr(stage["verifications"], "prompt_version")}
    stage["schema_versions"] = {"candidate": _first_attr(stage["candidates"], "schema_version"), "extraction": _first_attr(stage["extractions"], "schema_version"), "verification": _first_attr(stage["verifications"], "schema_version")}
    stage["input_hashes"] = _input_hashes(exp_dir, run_dirs)
    return stage


def stage_a_status(experiment_id: str) -> dict[str, Any]:
    stage = load_stage_a(experiment_id=experiment_id)
    _require_complete_stage_a(stage)
    return {"status": "STAGE_A_ARTIFACTS_COMPLETE", "candidates": len(stage["candidates"]), "extractions": len(stage["extractions"]), "verifications": len(stage["verifications"]), "validations": len(stage["validations"]), "routes": len(stage["routes"])}


def prepare_review_batch(*, experiment_id: str, review_batch_id: str, experiment_root: str | Path = DEFAULT_EXPERIMENT_ROOT, review_root: str | Path = DEFAULT_REVIEW_ROOT, force: bool = False) -> dict[str, Any]:
    stage = load_stage_a(experiment_id=experiment_id, experiment_root=experiment_root)
    _require_complete_stage_a(stage)
    batch_dir = Path(review_root) / review_batch_id
    if batch_dir.exists():
        if not force:
            raise FileExistsError(f"Review batch already exists: {batch_dir}")
        backup = batch_dir.with_name(f"{batch_dir.name}.backup_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}")
        batch_dir.rename(backup)
    batch_dir.mkdir(parents=True, exist_ok=False)
    review_candidates, excluded = _filter_review_candidates(stage["candidates"])
    sample_rows = [_sample_row(review_batch_id, item, stage) for item in review_candidates]
    field_rows = _field_rows(sample_rows, stage)
    _write_csv(batch_dir / "review_samples.csv", SAMPLE_FIELDS, sample_rows)
    _write_csv(batch_dir / "review_fields.csv", FIELD_FIELDS, field_rows)
    write_jsonl(batch_dir / "source_snapshots.jsonl", [_source_snapshot(row, stage) for row in sample_rows])
    (batch_dir / "review_instructions.md").write_text(_review_instructions(), encoding="utf-8")
    manifest = {
        "review_batch_id": review_batch_id,
        "experiment_id": experiment_id,
        "run_id": "+".join(stage["run_ids"]),
        "run_ids": stage["run_ids"],
        "provider": stage["provider"],
        "model_name": stage["model_name"],
        "candidate_count": len(sample_rows),
        "stage_a_candidate_count": len(stage["candidates"]),
        "excluded_candidate_count": len(excluded),
        "excluded_candidates": excluded,
        "allowed_languages": ["zh", "en"],
        "language_filter": "exclude_japanese_kana",
        "field_annotation_count": len(field_rows),
        "prompt_versions": stage["prompt_versions"],
        "schema_versions": stage["schema_versions"],
        "input_hashes": stage["input_hashes"],
        "created_at": datetime.now(UTC).isoformat(),
        "review_status": "READY_FOR_REVIEW",
        "artifacts": {"review_samples_csv": str(batch_dir / "review_samples.csv"), "review_fields_csv": str(batch_dir / "review_fields.csv"), "review_instructions_md": str(batch_dir / "review_instructions.md"), "source_snapshots_jsonl": str(batch_dir / "source_snapshots.jsonl")},
    }
    _write_json(batch_dir / "review_batch_manifest.json", manifest)
    return manifest


def import_review_batch(*, review_batch_id: str, samples_csv: str | Path, fields_csv: str | Path, review_origin: str = "human", review_root: str | Path = DEFAULT_REVIEW_ROOT) -> dict[str, Any]:
    batch_dir = Path(review_root) / review_batch_id
    manifest = _read_json(batch_dir / "review_batch_manifest.json")
    if manifest.get("review_batch_id") != review_batch_id:
        raise ValueError("review_batch_id mismatch")
    origin = _parse_origin(review_origin)
    if origin != ReviewOrigin.HUMAN:
        raise ValueError("V0.1 import-review requires HUMAN review_origin")
    if _has_jsonl_records(batch_dir / "review_decisions.jsonl") or _has_jsonl_records(batch_dir / "field_review_decisions.jsonl"):
        raise FileExistsError("Refusing to overwrite existing review decisions")
    original_samples = _read_csv(batch_dir / "review_samples.csv")
    original_fields = _read_csv(batch_dir / "review_fields.csv") if (batch_dir / "review_fields.csv").exists() else []
    submitted_samples = _read_csv(samples_csv)
    submitted_fields = _read_csv(fields_csv) if Path(fields_csv).exists() else []
    if len(submitted_samples) != len(original_samples):
        raise ValueError("Sample CSV row count changed")
    original_by_sample = {row["sample_id"]: row for row in original_samples}
    sample_decisions = []
    for line_no, row in enumerate(submitted_samples, start=2):
        original = original_by_sample.get(row.get("sample_id", ""))
        if original is None:
            raise ValueError(f"Unknown sample_id at CSV line {line_no}")
        _validate_readonly_row(row, original, SAMPLE_READONLY_FIELDS, line_no)
        reviewer_id = row.get("reviewer_id", "").strip()
        if not reviewer_id:
            raise ValueError(f"reviewer_id is required at CSV line {line_no}")
        decision = _parse_decision(row.get("review_decision", ""), line_no)
        corrected = _corrected_annotation(row)
        issue_codes = _parse_issue_codes(row.get("issue_codes", ""), line_no)
        if decision == ReviewDecisionType.EDIT and not _has_meaningful_sample_edit(corrected, original):
            raise ValueError(f"EDIT requires at least one corrected field at CSV line {line_no}")
        if decision == ReviewDecisionType.REJECT and not issue_codes:
            raise ValueError(f"REJECT requires issue_codes at CSV line {line_no}")
        if decision == ReviewDecisionType.UNRESOLVED and not row.get("review_comment", "").strip():
            raise ValueError(f"UNRESOLVED requires review_comment at CSV line {line_no}")
        review_id = make_record_id("review", review_batch_id, row["sample_id"], reviewer_id, decision.value, corrected, issue_codes, origin.value)
        sample_decisions.append({"review_id": review_id, "review_batch_id": review_batch_id, "sample_id": row["sample_id"], "candidate_id": row["candidate_id"], "source_revision_id": row["source_revision_id"], "reviewer_id": reviewer_id, "review_round": 1, "decision": decision.value, "review_origin": origin.value, "corrected_annotation": corrected, "issue_codes": issue_codes, "comment": row.get("review_comment", ""), "created_at": datetime.now(UTC).isoformat()})
    original_field_by_id = {row["field_annotation_id"]: row for row in original_fields}
    sample_decision_by_id = {row["sample_id"]: row for row in sample_decisions}
    field_decisions = []
    for line_no, row in enumerate(submitted_fields, start=2):
        field_id = row.get("field_annotation_id", "")
        original = original_field_by_id.get(field_id)
        if original is None:
            raise ValueError(f"Unknown field_annotation_id at CSV line {line_no}")
        _validate_readonly_row(row, original, FIELD_READONLY_FIELDS, line_no)
        decision_raw = row.get("field_decision", "").strip()
        if not decision_raw:
            continue
        decision = _parse_field_decision(decision_raw, line_no)
        sample_decision = sample_decision_by_id.get(row["sample_id"])
        if sample_decision and sample_decision["decision"] == ReviewDecisionType.REJECT.value and decision != "NOT_APPLICABLE":
            raise ValueError(f"Rejected sample fields must be NOT_APPLICABLE at CSV line {line_no}")
        reviewer_id = row.get("reviewer_id", "").strip() or (sample_decision or {}).get("reviewer_id", "")
        if not reviewer_id:
            raise ValueError(f"field reviewer_id is required at CSV line {line_no}")
        _validate_span_correction(row, line_no)
        field_decisions.append({"field_review_id": make_record_id("field_review", review_batch_id, field_id, reviewer_id, decision), "review_batch_id": review_batch_id, "sample_id": row["sample_id"], "field_annotation_id": field_id, "candidate_id": row["candidate_id"], "field_name": row["field_name"], "reviewer_id": reviewer_id, "field_decision": decision, "correct_value_original": row.get("correct_value_original", ""), "correct_value_normalized": row.get("correct_value_normalized", ""), "correct_source_type": row.get("correct_source_type", ""), "correct_quote": row.get("correct_quote", ""), "correct_span_start": row.get("correct_span_start", ""), "correct_span_end": row.get("correct_span_end", ""), "span_issue_code": row.get("span_issue_code", ""), "review_comment": row.get("review_comment", ""), "created_at": datetime.now(UTC).isoformat()})
    write_jsonl(batch_dir / "review_decisions.jsonl", sample_decisions)
    write_jsonl(batch_dir / "field_review_decisions.jsonl", field_decisions)
    report = {"review_batch_id": review_batch_id, "review_origin": origin.value, "sample_decisions_imported": len(sample_decisions), "field_decisions_imported": len(field_decisions), "decision_counts": dict(Counter(item["decision"] for item in sample_decisions)), "created_at": datetime.now(UTC).isoformat()}
    _write_json(batch_dir / "review_import_report.json", report)
    return report


def build_silver_dataset(*, review_batch_id: str, review_root: str | Path = DEFAULT_REVIEW_ROOT) -> dict[str, Any]:
    batch_dir = Path(review_root) / review_batch_id
    manifest = _read_json(batch_dir / "review_batch_manifest.json")
    stage = load_stage_a(experiment_id=manifest["experiment_id"])
    decisions = list(read_jsonl(batch_dir / "review_decisions.jsonl")) if (batch_dir / "review_decisions.jsonl").exists() else []
    sample_rows = {row["sample_id"]: row for row in _read_csv(batch_dir / "review_samples.csv")}
    by_candidate = _stage_maps(stage)
    silver = []
    skipped = Counter()
    for decision in decisions:
        if decision.get("review_origin") != ReviewOrigin.HUMAN.value:
            skipped["non_human_review"] += 1
            continue
        if decision.get("decision") not in {ReviewDecisionType.ACCEPT.value, ReviewDecisionType.EDIT.value}:
            skipped[decision.get("decision", "skipped").lower()] += 1
            continue
        sample = sample_rows.get(decision["sample_id"])
        candidate = by_candidate["candidates"].get(decision["candidate_id"])
        extraction = by_candidate["extractions"].get(decision["candidate_id"])
        verification = by_candidate["verifications"].get(decision["candidate_id"])
        validation = by_candidate["validations"].get(decision["candidate_id"])
        if not sample or not candidate or not extraction or not verification or not validation:
            skipped["missing_stage_a_record"] += 1
            continue
        if _is_fake_extraction(extraction):
            skipped["fake_extraction"] += 1
            continue
        if sample["source_revision_id"] != candidate.source_revision_id or validation.overall_status == SpanSupportStatus.SOURCE_REVISION_MISMATCH:
            skipped["source_revision_mismatch"] += 1
            continue
        final = _final_annotation(extraction, decision)
        if not final.get("is_formal_recommendation"):
            skipped["not_formal_recommendation"] += 1
            continue
        if not _silver_evidence_ok(final, extraction, validation):
            skipped["invalid_or_missing_evidence"] += 1
            continue
        silver.append(_silver_record(review_batch_id, decision, candidate, extraction, verification, final))
    write_jsonl(batch_dir / "silver_recommendations.jsonl", silver)
    silver_manifest = {"review_batch_id": review_batch_id, "annotation_tier": AnnotationTier.SILVER_SINGLE_REVIEW.value, "built": len(silver), "skipped": sum(skipped.values()), "skip_reasons": dict(skipped), "created_at": datetime.now(UTC).isoformat()}
    _write_json(batch_dir / "silver_manifest.json", silver_manifest)
    return silver_manifest


def evaluate_review_batch(*, review_batch_id: str, review_root: str | Path = DEFAULT_REVIEW_ROOT, real_only: bool = True) -> dict[str, Any]:
    batch_dir = Path(review_root) / review_batch_id
    manifest = _read_json(batch_dir / "review_batch_manifest.json")
    stage = load_stage_a(experiment_id=manifest["experiment_id"])
    sample_rows = {row["sample_id"]: row for row in _read_csv(batch_dir / "review_samples.csv")}
    decisions = list(read_jsonl(batch_dir / "review_decisions.jsonl")) if (batch_dir / "review_decisions.jsonl").exists() else []
    silver = list(read_jsonl(batch_dir / "silver_recommendations.jsonl")) if (batch_dir / "silver_recommendations.jsonl").exists() else []
    human = [item for item in decisions if item.get("review_origin") == ReviewOrigin.HUMAN.value]
    counts = Counter(item.get("decision", "") for item in human)
    report = {
        "evaluation_scope": "REAL_MODEL_HUMAN_REVIEWED" if real_only else "ALL_REVIEWED",
        "data_scale": {"candidates": int(manifest.get("candidate_count", len(stage["candidates"]))), "stage_a_candidates": len(stage["candidates"]), "excluded_candidates": int(manifest.get("excluded_candidate_count", 0)), "human_reviewed_samples": len(human), "accepted": counts.get("ACCEPT", 0), "edited": counts.get("EDIT", 0), "rejected": counts.get("REJECT", 0), "unresolved": counts.get("UNRESOLVED", 0), "silver_examples": len(silver)},
        "recommendation_classification": _classification_metrics(human, sample_rows),
        "field_correctness": _field_metrics_v01(human, sample_rows),
        "evidence_span": _span_metrics(stage, batch_dir),
        "review_cost": _review_cost(human),
        "model_stability": _model_stability(stage),
    }
    _write_json(batch_dir / "evaluation_v0_1.json", report)
    (batch_dir / "evaluation_v0_1.md").write_text(_render_eval_md(report), encoding="utf-8")
    write_jsonl(batch_dir / "error_examples_v0_1.jsonl", _error_examples(human, sample_rows, batch_dir))
    return report


def complete_v0(*, experiment_id: str, review_batch_id: str, review_root: str | Path = DEFAULT_REVIEW_ROOT, force_prepare: bool = False) -> dict[str, Any]:
    batch_dir = Path(review_root) / review_batch_id
    if not (batch_dir / "review_batch_manifest.json").exists():
        manifest = prepare_review_batch(experiment_id=experiment_id, review_batch_id=review_batch_id, review_root=review_root, force=force_prepare)
    else:
        manifest = _read_json(batch_dir / "review_batch_manifest.json")
    completed_samples = batch_dir / "completed_review_samples.csv"
    completed_fields = batch_dir / "completed_review_fields.csv"
    if not completed_samples.exists() or not completed_fields.exists() or not _looks_review_complete(completed_samples):
        write_loop_report(review_batch_id=review_batch_id, review_root=review_root)
        return {"status": "READY_FOR_HUMAN_REVIEW", "review_batch_id": review_batch_id, "review_samples_csv": str(batch_dir / "review_samples.csv"), "review_fields_csv": str(batch_dir / "review_fields.csv"), "completed_samples_expected": str(completed_samples), "completed_fields_expected": str(completed_fields), "manifest": str(batch_dir / "review_batch_manifest.json"), "sample_count": manifest.get("candidate_count", 0)}
    if not (batch_dir / "review_decisions.jsonl").exists():
        import_review_batch(review_batch_id=review_batch_id, samples_csv=completed_samples, fields_csv=completed_fields, review_origin="human", review_root=review_root)
    silver_manifest = build_silver_dataset(review_batch_id=review_batch_id, review_root=review_root)
    evaluation = evaluate_review_batch(review_batch_id=review_batch_id, review_root=review_root, real_only=True)
    final_report = write_loop_report(review_batch_id=review_batch_id, review_root=review_root)
    return {"status": "INFORMATION_EXTRACTION_LOOP_V0_1_COMPLETE", "review_batch_id": review_batch_id, "silver": silver_manifest, "evaluation": evaluation, "report": final_report}


def write_loop_report(*, review_batch_id: str, review_root: str | Path = DEFAULT_REVIEW_ROOT) -> str:
    batch_dir = Path(review_root) / review_batch_id
    manifest = _read_json(batch_dir / "review_batch_manifest.json")
    gate = _read_json(Path(DEFAULT_EXPERIMENT_ROOT) / manifest["experiment_id"] / "stage_a_technical_gate.json")
    decisions = list(read_jsonl(batch_dir / "review_decisions.jsonl")) if (batch_dir / "review_decisions.jsonl").exists() else []
    silver_manifest = _read_json(batch_dir / "silver_manifest.json") if (batch_dir / "silver_manifest.json").exists() else {"built": 0, "skipped": 0, "skip_reasons": {}}
    eval_report = _read_json(batch_dir / "evaluation_v0_1.json") if (batch_dir / "evaluation_v0_1.json").exists() else {}
    counts = Counter(item.get("decision", "") for item in decisions if item.get("review_origin") == ReviewOrigin.HUMAN.value)
    data_scale = eval_report.get("data_scale", {})
    status = "INFORMATION_EXTRACTION_LOOP_V0_1_COMPLETE" if eval_report else "READY_FOR_HUMAN_REVIEW"
    next_steps = ["- Promote to Gold only after double review or adjudication is implemented."] if eval_report else ["- Complete human review.", "- Then run build-silver and evaluate-review."]
    lines = [
        "# Guideline Information Extraction Loop V0.1", "",
        "V0.1 uses single-reviewer human review, so it produces a Silver Dataset, not a formal Gold Dataset.", "",
        "## Status", f"- status: {status}", "",
        "## Data Source", f"- experiment_id: {manifest['experiment_id']}", "- source: real IDSA Stage A sections and DeepSeek extraction artifacts", f"- model: {manifest.get('model_name', '')}", f"- prompt_versions: {manifest.get('prompt_versions', {})}", "",
        "## Stage A", f"- candidates: {data_scale.get('stage_a_candidates', gate.get('stage_a_candidates', manifest.get('stage_a_candidate_count', manifest.get('candidate_count', 0))))}", f"- active_review_candidates: {data_scale.get('candidates', manifest.get('candidate_count', 0))}", f"- excluded_candidates: {data_scale.get('excluded_candidates', manifest.get('excluded_candidate_count', 0))}", f"- extraction_success: {gate.get('extraction_success', 0)}", f"- verification_success: {gate.get('verification_success', 0)}", "",
        "## Human Review", f"- reviewed: {data_scale.get('human_reviewed_samples', sum(counts.values()))}", f"- ACCEPT: {data_scale.get('accepted', counts.get('ACCEPT', 0))}", f"- EDIT: {data_scale.get('edited', counts.get('EDIT', 0))}", f"- REJECT: {data_scale.get('rejected', counts.get('REJECT', 0))}", f"- UNRESOLVED: {data_scale.get('unresolved', counts.get('UNRESOLVED', 0))}", "",
        "## Silver", f"- silver_examples: {silver_manifest.get('built', 0)}", f"- skipped: {silver_manifest.get('skipped', 0)}", f"- skip_reasons: {silver_manifest.get('skip_reasons', {})}", "",
        "## Evaluation", f"- field_correctness: {eval_report.get('field_correctness', {})}", f"- evidence_span: {eval_report.get('evidence_span', {})}", "",
        "## Next Steps", *next_steps,
    ]
    target = Path("reports") / "guideline_information_loop_v0_1.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(target)



def _filter_review_candidates(candidates: list[RecommendationCandidate]) -> tuple[list[RecommendationCandidate], list[dict[str, str]]]:
    kept: list[RecommendationCandidate] = []
    excluded: list[dict[str, str]] = []
    for candidate in candidates:
        text = "\n".join([candidate.title, candidate.candidate_text, candidate.context_before, candidate.context_after])
        if _contains_japanese_kana(text):
            excluded.append({"candidate_id": candidate.candidate_id, "doc_id": candidate.doc_id, "title": candidate.title, "reason": "JAPANESE_GUIDELINE_EXCLUDED"})
        else:
            kept.append(candidate)
    return kept, excluded


def _contains_japanese_kana(text: str) -> bool:
    return any("\u3040" <= ch <= "\u30ff" for ch in text or "")
def _require_complete_stage_a(stage: dict[str, Any]) -> None:
    expected = {item.candidate_id for item in stage["candidates"]}
    maps = _stage_maps(stage)
    missing: dict[str, list[str]] = {}
    for name in ["candidates", "extractions", "verifications", "validations", "routes"]:
        ids = set(maps[name].keys())
        if ids != expected:
            missing[name] = sorted(expected - ids)
    counts = {"candidates": len(stage["candidates"]), "extractions": len(stage["extractions"]), "verifications": len(stage["verifications"]), "validations": len(stage["validations"]), "routes": len(stage["routes"])}
    if counts != {"candidates": 5, "extractions": 5, "verifications": 5, "validations": 5, "routes": 5} or missing:
        raise IncompleteStageAArtifacts(json.dumps({"missing": missing, "counts": counts}, ensure_ascii=False))
    for extraction in stage["extractions"]:
        if not extraction.model_name or not extraction.prompt_version or not extraction.schema_version:
            raise IncompleteStageAArtifacts(f"Extraction missing metadata: {extraction.candidate_id}")
    for verification in stage["verifications"]:
        if not verification.model_name or not verification.prompt_version or not verification.schema_version:
            raise IncompleteStageAArtifacts(f"Verification missing metadata: {verification.candidate_id}")


def _stage_maps(stage: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        "candidates": {item.candidate_id: item for item in stage["candidates"]},
        "extractions": {item.candidate_id: item for item in stage["extractions"]},
        "verifications": {item.candidate_id: item for item in stage["verifications"]},
        "validations": {item.candidate_id: item for item in stage["validations"]},
        "routes": {item.candidate_id: item for item in stage["routes"]},
    }


def _sample_row(review_batch_id: str, candidate: RecommendationCandidate, stage: dict[str, Any]) -> dict[str, str]:
    maps = _stage_maps(stage)
    extraction = maps["extractions"][candidate.candidate_id]
    verification = maps["verifications"][candidate.candidate_id]
    validation = maps["validations"][candidate.candidate_id]
    return {
        "sample_id": make_record_id("sample", review_batch_id, candidate.candidate_id), "candidate_id": candidate.candidate_id, "doc_id": candidate.doc_id, "title": candidate.title, "section_path": " > ".join(candidate.section_path), "source_revision_id": candidate.source_revision_id,
        "candidate_text": candidate.candidate_text, "context_before": candidate.context_before, "context_after": candidate.context_after,
        "model_is_formal": str(extraction.is_formal_recommendation), "model_recommendation_type": extraction.recommendation_type.value if hasattr(extraction.recommendation_type, "value") else str(extraction.recommendation_type), "model_recommendation_text": extraction.recommendation_text,
        "model_direction": extraction.direction or "", "model_strength": extraction.strength or "", "model_certainty": extraction.certainty or "", "model_population": extraction.population or "", "model_interventions": _join_list(extraction.interventions), "model_dosage": extraction.dosage or "", "model_duration": extraction.duration or "",
        "verifier_agreement": str(verification.agrees_is_formal_recommendation), "validation_summary": _validation_summary(validation), **{field: "" for field in SAMPLE_REVIEW_FIELDS},
    }


def _field_rows(sample_rows: list[dict[str, str]], stage: dict[str, Any]) -> list[dict[str, str]]:
    maps = _stage_maps(stage)
    rows: list[dict[str, str]] = []
    sample_by_candidate = {row["candidate_id"]: row for row in sample_rows}
    for candidate_id, sample in sample_by_candidate.items():
        candidate = maps["candidates"][candidate_id]
        extraction = maps["extractions"][candidate_id]
        validation = maps["validations"][candidate_id]
        for field_name, evidence_items in extraction.field_evidence.items():
            statuses = validation.field_statuses.get(field_name, [])
            for index, item in enumerate(evidence_items):
                rows.append({"sample_id": sample["sample_id"], "field_annotation_id": make_record_id("field_annotation", sample["sample_id"], field_name, index), "candidate_id": candidate.candidate_id, "field_name": field_name, "model_value_original": item.value_original, "model_value_normalized": item.value_normalized, "source_type": item.source_type.value, "span_coordinate_space": item.span_coordinate_space.value, "quote": item.quote, "span_start": "" if item.span_start is None else str(item.span_start), "span_end": "" if item.span_end is None else str(item.span_end), "validation_status": statuses[index].value if index < len(statuses) else "", "source_text_excerpt": _source_excerpt(candidate, item.quote), **{field: "" for field in FIELD_REVIEW_FIELDS}})
    return rows


def _source_snapshot(row: dict[str, str], stage: dict[str, Any]) -> dict[str, Any]:
    candidate = _stage_maps(stage)["candidates"][row["candidate_id"]]
    return {"sample_id": row["sample_id"], "candidate_id": candidate.candidate_id, "doc_id": candidate.doc_id, "source_block_key": candidate.source_block_key, "source_revision_id": candidate.source_revision_id, "section_path": candidate.section_path, "candidate_text": candidate.candidate_text, "context_before": candidate.context_before, "context_after": candidate.context_after}


def _review_instructions() -> str:
    return """# IDSA Stage A Review Instructions

1. A formal Recommendation is a guideline statement that tells clinicians or patients what should, should not, may, or is recommended to do. Background, rationale, evidence summaries, and methods text are not formal recommendations unless they contain a directive recommendation.
2. Direction is action polarity such as FOR or AGAINST. Strength is how forceful the recommendation is. Certainty is the evidence confidence level.
3. Population may be inherited from the recommendation sentence, clinical question, section heading, or immediate context only when the text explicitly supports it.
4. Intervention, Dosage, and Duration must come from source text. Do not fill them from medical knowledge.
5. Evidence Span review checks whether quote and span support the field value and come from expected source text.
6. ACCEPT means the model core annotation is usable as-is. EDIT means at least one field is corrected. REJECT means this is not a target recommendation. UNRESOLVED means you cannot judge reliably and must explain why.
7. Use issue_codes for concrete problems such as false positive, direction error, strength error, unsupported field, or invalid evidence span.
8. Do not edit readonly columns. Only fill reviewer_id, decision, correction, issue, and comment columns.
9. Submit completed_review_samples.csv and completed_review_fields.csv in this review batch directory.
"""


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _validate_readonly_row(row: dict[str, str], original: dict[str, str], fields: list[str], line_no: int) -> None:
    for field in fields:
        if (row.get(field) or "") != (original.get(field) or ""):
            if field == "source_revision_id":
                raise ValueError(f"source_revision_id mismatch at CSV line {line_no}")
            raise ValueError(f"Readonly field {field} changed at CSV line {line_no}")


def _parse_decision(raw: str, line_no: int) -> ReviewDecisionType:
    value = raw.strip()
    if not value:
        raise ValueError(f"review_decision is required at CSV line {line_no}")
    try:
        return ReviewDecisionType(value)
    except ValueError as exc:
        raise ValueError(f"Invalid review_decision at CSV line {line_no}") from exc


def _parse_field_decision(raw: str, line_no: int) -> str:
    if raw not in {"ACCEPT", "EDIT", "REJECT", "NOT_APPLICABLE"}:
        raise ValueError(f"Invalid field_decision at CSV line {line_no}")
    return raw


def _parse_origin(raw: str) -> ReviewOrigin:
    mapping = {"human": ReviewOrigin.HUMAN, "synthetic-test": ReviewOrigin.SYNTHETIC_TEST, "migrated": ReviewOrigin.MIGRATED}
    if raw not in mapping:
        raise ValueError(f"Invalid review_origin: {raw}")
    return mapping[raw]


def _parse_issue_codes(raw: str, line_no: int) -> list[str]:
    values = [item.strip() for item in raw.split("|") if item.strip()]
    unknown = [item for item in values if item not in {item.value for item in IssueCode}]
    if unknown:
        raise ValueError(f"Unknown issue_codes at CSV line {line_no}: {unknown}")
    return values


def _corrected_annotation(row: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for csv_field, target in CORRECTED_SAMPLE_MAP.items():
        value = row.get(csv_field, "").strip()
        if not value:
            continue
        result[target] = [item.strip() for item in value.split("|") if item.strip()] if target == "interventions" else (value.lower() in {"true", "1", "yes"} if target == "is_formal_recommendation" else value)
    return result


def _has_meaningful_sample_edit(corrected: dict[str, Any], original: dict[str, str]) -> bool:
    if not corrected:
        return False
    original_map = {"is_formal_recommendation": original.get("model_is_formal", ""), "recommendation_type": original.get("model_recommendation_type", ""), "recommendation_text": original.get("model_recommendation_text", ""), "direction": original.get("model_direction", ""), "strength": original.get("model_strength", ""), "certainty": original.get("model_certainty", ""), "population": original.get("model_population", ""), "interventions": original.get("model_interventions", ""), "dosage": original.get("model_dosage", ""), "duration": original.get("model_duration", "")}
    return any((_join_list(value) if key == "interventions" else str(value)) != original_map.get(key, "") for key, value in corrected.items())


def _validate_span_correction(row: dict[str, str], line_no: int) -> None:
    start = row.get("correct_span_start", "").strip(); end = row.get("correct_span_end", "").strip()
    if not start and not end:
        return
    if not start or not end:
        raise ValueError(f"Both correct_span_start and correct_span_end are required at CSV line {line_no}")
    start_i = int(start); end_i = int(end)
    if start_i < 0 or end_i < start_i:
        raise ValueError(f"Corrected span is invalid at CSV line {line_no}")


def _final_annotation(extraction: ExtractionResult, decision: dict[str, Any]) -> dict[str, Any]:
    model = _model_annotation(extraction)
    if decision.get("decision") == ReviewDecisionType.ACCEPT.value:
        return model
    final = dict(model); final.update(decision.get("corrected_annotation") or {})
    return final


def _model_annotation(extraction: ExtractionResult) -> dict[str, Any]:
    return {"is_formal_recommendation": extraction.is_formal_recommendation, "recommendation_type": extraction.recommendation_type.value if hasattr(extraction.recommendation_type, "value") else str(extraction.recommendation_type), "recommendation_text": extraction.recommendation_text, "direction": extraction.direction, "strength": extraction.strength, "certainty": extraction.certainty, "population": extraction.population, "interventions": extraction.interventions, "dosage": extraction.dosage, "duration": extraction.duration}


def _silver_evidence_ok(final: dict[str, Any], extraction: ExtractionResult, validation: ValidationResult) -> bool:
    if validation.overall_status in {SpanSupportStatus.INVALID_SPAN, SpanSupportStatus.SOURCE_REVISION_MISMATCH}:
        return False
    if final.get("recommendation_text") and not extraction.field_evidence.get("recommendation_text"):
        return False
    invalid = {SpanIssueCode.SPAN_OUT_OF_RANGE.value, SpanIssueCode.WRONG_SOURCE_BLOCK.value, SpanIssueCode.WRONG_CONTEXT_SOURCE.value, SpanIssueCode.SOURCE_REVISION_MISMATCH.value}
    return not any((issue.value if hasattr(issue, "value") else str(issue)) in invalid for issues in validation.field_issue_codes.values() for issue in issues)


def _silver_record(review_batch_id: str, decision: dict[str, Any], candidate: RecommendationCandidate, extraction: ExtractionResult, verification: VerificationResult, final: dict[str, Any]) -> dict[str, Any]:
    return {"silver_example_id": make_record_id("silver", review_batch_id, decision["review_id"]), "annotation_tier": AnnotationTier.SILVER_SINGLE_REVIEW.value, "candidate_id": candidate.candidate_id, "doc_id": candidate.doc_id, "source_block_key": candidate.source_block_key, "source_revision_id": candidate.source_revision_id, "section_path": candidate.section_path, "final_recommendation_text": final.get("recommendation_text") or "", "final_recommendation_type": final.get("recommendation_type") or "", "final_direction": final.get("direction"), "final_strength": final.get("strength"), "final_certainty": final.get("certainty"), "final_population": final.get("population"), "final_interventions": final.get("interventions") or [], "final_dosage": final.get("dosage"), "final_duration": final.get("duration"), "field_evidence": extraction.model_dump(mode="json")["field_evidence"], "extraction_id": extraction.extraction_id, "verification_id": verification.verification_id, "review_id": decision["review_id"], "reviewer_id": decision["reviewer_id"], "reviewed_at": decision["created_at"], "provider": "openai-compatible", "model_name": extraction.model_name, "prompt_version": extraction.prompt_version, "schema_version": extraction.schema_version, "normalizer_version": NORMALIZER_VERSION, "model_original_annotation": _model_annotation(extraction), "human_final_annotation": final}


def _classification_metrics(decisions: list[dict[str, Any]], sample_rows: dict[str, dict[str, str]]) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for decision in decisions:
        if decision.get("decision") == ReviewDecisionType.UNRESOLVED.value:
            continue
        row = sample_rows[decision["sample_id"]]
        model_formal = row.get("model_is_formal") == "True"
        human_formal = False if decision.get("decision") == ReviewDecisionType.REJECT.value else _final_formal_from_decision(decision, row)
        if model_formal and human_formal:
            tp += 1
        elif model_formal and not human_formal:
            fp += 1
        elif not model_formal and human_formal:
            fn += 1
        else:
            tn += 1
    precision = _safe_div(tp, tp + fp)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": None, "f1": None, "recall_note": "Recall unavailable because the full source section set was not exhaustively annotated."}


def _final_formal_from_decision(decision: dict[str, Any], row: dict[str, str]) -> bool:
    corrected = decision.get("corrected_annotation") or {}
    if "is_formal_recommendation" in corrected:
        return bool(corrected["is_formal_recommendation"])
    return row.get("model_is_formal") == "True"


def _field_metrics_v01(decisions: list[dict[str, Any]], sample_rows: dict[str, dict[str, str]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in SILVER_FIELDS:
        correct = edited = rejected = missing = total = 0
        for decision in decisions:
            if decision.get("decision") == ReviewDecisionType.UNRESOLVED.value:
                continue
            if decision.get("decision") == ReviewDecisionType.REJECT.value:
                rejected += 1; total += 1; continue
            row = sample_rows[decision["sample_id"]]
            corrected = decision.get("corrected_annotation") or {}
            model_value = _sample_model_value(row, field)
            total += 1
            if field in corrected and _normalize_value(corrected[field]) != _normalize_value(model_value):
                edited += 1
                if model_value in (None, "", []):
                    missing += 1
            else:
                correct += 1
        result[field] = {"correct": correct, "edited": edited, "rejected": rejected, "missing": missing, "total": total, "accuracy": _safe_div(correct, total)}
    return result


def _span_metrics(stage: dict[str, Any], batch_dir: Path) -> dict[str, Any]:
    issues = Counter(); statuses = Counter()
    for validation in stage["validations"]:
        for values in validation.field_statuses.values():
            for status in values:
                statuses[status.value if hasattr(status, "value") else str(status)] += 1
        for values in validation.field_issue_codes.values():
            for issue in values:
                issues[issue.value if hasattr(issue, "value") else str(issue)] += 1
    field_decisions = list(read_jsonl(batch_dir / "field_review_decisions.jsonl")) if (batch_dir / "field_review_decisions.jsonl").exists() else []
    return {"exact": issues.get(SpanIssueCode.EXACT_MATCH.value, 0), "normalized": issues.get(SpanIssueCode.NORMALIZATION_ONLY.value, 0), "partial": issues.get(SpanIssueCode.PARTIAL_EVIDENCE.value, 0), "invalid": issues.get(SpanIssueCode.SPAN_OUT_OF_RANGE.value, 0), "wrong_source": issues.get(SpanIssueCode.WRONG_SOURCE_BLOCK.value, 0) + issues.get(SpanIssueCode.WRONG_CONTEXT_SOURCE.value, 0), "unsupported": issues.get(SpanIssueCode.UNSUPPORTED_FIELD.value, 0) + issues.get(SpanIssueCode.QUOTE_NOT_FOUND.value, 0), "human_corrected": sum(1 for item in field_decisions if item.get("field_decision") == "EDIT" or item.get("correct_span_start")), "status_distribution": dict(statuses), "issue_distribution": dict(issues)}


def _review_cost(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(item.get("decision", "") for item in decisions); total = len(decisions)
    return {"accept_rate": _safe_div(counts.get("ACCEPT", 0), total), "edit_rate": _safe_div(counts.get("EDIT", 0), total), "reject_rate": _safe_div(counts.get("REJECT", 0), total), "average_corrected_fields": _safe_div(sum(len(item.get("corrected_annotation") or {}) for item in decisions), total), "average_corrected_spans": None}


def _model_stability(stage: dict[str, Any]) -> dict[str, Any]:
    responses = stage["model_responses"]; normalizations = stage["normalizations"]
    return {"extractor_success": len(stage["extractions"]), "verifier_success": len(stage["verifications"]), "json_parse_success": len(responses), "normalization_events": sum(len(item.get("events") or []) for item in normalizations), "retry_count": sum(int((item.get("response_metadata") or {}).get("retry_count") or item.get("retry_count") or 0) for item in responses), "average_latency_ms": _safe_div(sum(int(item.get("latency_ms") or 0) for item in responses), len(responses)), "input_tokens": sum(int((item.get("usage") or {}).get("input_tokens") or (item.get("usage") or {}).get("prompt_tokens") or 0) for item in responses), "output_tokens": sum(int((item.get("usage") or {}).get("output_tokens") or (item.get("usage") or {}).get("completion_tokens") or 0) for item in responses)}


def _error_examples(decisions: list[dict[str, Any]], sample_rows: dict[str, dict[str, str]], batch_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for decision in decisions:
        sample = sample_rows.get(decision["sample_id"], {})
        for code in decision.get("issue_codes") or []:
            rows.append({"sample_id": decision["sample_id"], "candidate_id": decision["candidate_id"], "issue_code": code, "field_name": "sample", "model_value": sample.get("model_recommendation_text", "")[:240], "human_value": json.dumps(decision.get("corrected_annotation") or {}, ensure_ascii=False)[:240], "short_source_excerpt": sample.get("candidate_text", "")[:240], "likely_component": _likely_component(code)})
    field_decisions = list(read_jsonl(batch_dir / "field_review_decisions.jsonl")) if (batch_dir / "field_review_decisions.jsonl").exists() else []
    for item in field_decisions:
        if item.get("field_decision") in {"EDIT", "REJECT"} or item.get("span_issue_code"):
            rows.append({"sample_id": item["sample_id"], "candidate_id": item["candidate_id"], "issue_code": item.get("span_issue_code") or item.get("field_decision"), "field_name": item.get("field_name", ""), "model_value": "", "human_value": item.get("correct_value_normalized") or item.get("correct_quote") or "", "short_source_excerpt": "", "likely_component": "span_validator" if item.get("span_issue_code") else "extractor"})
    return rows


def _render_eval_md(report: dict[str, Any]) -> str:
    lines = ["# Guideline Information V0.1 Evaluation", ""]
    for key, value in report.items():
        lines.append(f"## {key}")
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                lines.append(f"- {sub_key}: {sub_value}")
        else:
            lines.append(f"- value: {value}")
        lines.append("")
    return "\n".join(lines)


def _looks_review_complete(path: Path) -> bool:
    rows = _read_csv(path)
    return bool(rows) and all(row.get("review_decision", "").strip() for row in rows)


def _sample_model_value(row: dict[str, str], field: str) -> Any:
    key = "model_recommendation_text" if field == "recommendation_text" else f"model_{field}"
    return row.get(key, "")


def _normalize_value(value: Any) -> str:
    if isinstance(value, list):
        return "|".join(str(item).strip().lower() for item in value)
    if value is None:
        return ""
    return str(value).strip().lower()


def _join_list(value: Any) -> str:
    return "|".join(str(item) for item in value or [])


def _safe_div(num: int | float, den: int | float) -> float | None:
    return None if not den else num / den


def _is_fake_extraction(extraction: ExtractionResult) -> bool:
    model_name = (extraction.model_name or extraction.model_version or "").lower()
    return model_name.startswith("fake") or "fake" in model_name


def _validation_summary(validation: ValidationResult) -> str:
    statuses = Counter(status.value if hasattr(status, "value") else str(status) for values in validation.field_statuses.values() for status in values)
    return f"overall={validation.overall_status.value}; statuses={dict(statuses)}"


def _source_excerpt(candidate: RecommendationCandidate, quote: str, limit: int = 240) -> str:
    text = candidate.candidate_text or ""
    if quote and quote in text:
        pos = text.find(quote); start = max(0, pos - 60)
        return text[start:start + limit]
    return text[:limit]


def _likely_component(issue_code: str) -> str:
    if "SPAN" in issue_code or "SOURCE" in issue_code or "EVIDENCE" in issue_code:
        return "span_validator"
    if "FALSE_POSITIVE" in issue_code or "FALSE_NEGATIVE" in issue_code:
        return "candidate_or_extractor"
    return "extractor"


def _read_model_many(run_dirs: list[Path], name: str, model) -> list[Any]:
    items: list[Any] = []
    for run_dir in run_dirs:
        path = run_dir / name
        if path.exists():
            items.extend(read_models(path, model))
    return items


def _read_jsonl_many(run_dirs: list[Path], name: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        path = run_dir / name
        if path.exists():
            items.extend(read_jsonl(path))
    return items


def _input_hashes(exp_dir: Path, run_dirs: list[Path]) -> dict[str, str]:
    paths = [exp_dir / "experiment_manifest.json", exp_dir / "stage_a_technical_gate.json", exp_dir / "stage_a_candidates.jsonl"]
    for run_dir in run_dirs:
        paths.extend([run_dir / "candidates.jsonl", run_dir / "extraction_results.jsonl", run_dir / "verification_results.jsonl", run_dir / "validation_results.jsonl", run_dir / "route_results.jsonl"])
    return {str(path): sha256_file(path) for path in paths if path.exists()}


def _first_attr(items: list[Any], attr: str) -> str:
    for item in items:
        value = getattr(item, attr, "")
        if value:
            return str(value)
    return ""


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    resolved = Path(path); resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _has_jsonl_records(path: Path) -> bool:
    return path.exists() and any(line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines())





