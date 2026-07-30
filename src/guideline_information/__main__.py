"""CLI for guideline information processing."""

from __future__ import annotations

import argparse
import json

from src.guideline_information.enums import ReviewOrigin
from src.guideline_information.evaluation.metrics import build_error_examples, build_evaluation_report, build_real_model_evaluation_report, render_markdown_report
from src.guideline_information.experiment import missing_model_configuration, model_configuration_status, prepare_experiment_batch
from src.guideline_information.extraction.model_config import model_config_audit
from src.guideline_information.models import (
    ExtractionResult,
    GoldRecommendationExample,
    RecommendationCandidate,
    ReviewDecision,
    ReviewSample,
    RouteResult,
    ValidationResult,
    VerificationResult,
)
from src.guideline_information.orchestration.pilot_pipeline import run_pilot
from src.guideline_information.orchestration.run_pipeline import run_information_pipeline
from src.guideline_information.paths import DEFAULT_INFORMATION_DIR, InformationRunPaths
from src.guideline_information.pilot import prepare_pilot
from src.guideline_information.repository import read_models, write_models
from src.guideline_information.review.gold_builder import build_gold_from_review
from src.guideline_information.review.import_export import import_review_decisions_csv
from src.guideline_information.review.selectors import RandomAuditSelector, build_review_samples
from src.guideline_information.review_loop import (
    build_silver_dataset,
    complete_v0,
    evaluate_review_batch,
    import_review_batch,
    prepare_review_batch,
    stage_a_status,
)
from src.utils.io import DATA_DIR, ensure_parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Guideline information processing")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_prepare_pilot(sub)
    _add_prepare_experiment(sub)
    _add_model_config_audit(sub)
    _add_run_pilot(sub)
    _add_review_commands(sub)
    _add_legacy_commands(sub)
    args = parser.parse_args()
    try:
        payload = _dispatch(args)
    except Exception as exc:
        payload = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        raise SystemExit(1) from exc
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _add_prepare_pilot(sub) -> None:
    command = sub.add_parser("prepare-pilot")
    command.add_argument("--evidence-data-dir", default=str(DATA_DIR))
    command.add_argument("--output-root", default=str(DEFAULT_INFORMATION_DIR))
    command.add_argument("--organization", default="IDSA")
    command.add_argument("--pilot-id", default="idsa_pilot_v1")
    command.add_argument("--documents", type=int, default=3)
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--max-sections", type=int, default=150)


def _add_prepare_experiment(sub) -> None:
    command = sub.add_parser("prepare-experiment")
    command.add_argument("--pilot-id", required=True)
    command.add_argument("--pilot-root", default=str(DEFAULT_INFORMATION_DIR))
    command.add_argument("--experiment-root", default="data/guideline_information/experiments")
    command.add_argument("--experiment-id", default="idsa_real_model_experiment_v1")
    command.add_argument("--source-run-id", default="idsa_real_model_candidate_batch")
    command.add_argument("--stage-a-size", type=int, default=5)
    command.add_argument("--stage-b-size", type=int, default=15)
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--parent-experiment-id", default="")
    command.add_argument("--restart-reason", default="")

def _add_model_config_audit(sub) -> None:
    command = sub.add_parser("model-config-audit")
    command.add_argument("--probe", action="store_true")
    command.add_argument("--show-models", action="store_true")
    command.add_argument("--experiment-root", default="data/guideline_information/experiments")
    command.add_argument("--experiment-id", default="idsa_real_model_experiment_v1")


def _add_run_pilot(sub) -> None:
    command = sub.add_parser("run-pilot")
    command.add_argument("--pilot-id", required=True)
    command.add_argument("--run-id", required=True)
    command.add_argument("--output-root", default=str(DEFAULT_INFORMATION_DIR))
    command.add_argument("--client", choices=["fake", "openai-compatible"], default="fake")
    command.add_argument("--experiment-id", default="")
    command.add_argument("--input", default=None)
    command.add_argument("--max-items", type=int, default=None)
    command.add_argument("--max-model-calls", type=int, default=None)
    command.add_argument("--resume", action="store_true")
    command.add_argument("--force", action="store_true")
    command.add_argument("--dry-run", action="store_true")
    command.add_argument("--fail-fast", action="store_true")
    command.add_argument("--random-audit-rate", type=float, default=0.1)
    command.add_argument("--seed", type=int, default=42)


def _add_review_commands(sub) -> None:
    command = sub.add_parser("prepare-review")
    command.add_argument("--experiment-id", required=True)
    command.add_argument("--review-batch-id", required=True)
    command.add_argument("--experiment-root", default="data/guideline_information/experiments")
    command.add_argument("--review-root", default="data/guideline_information/review_batches")
    command.add_argument("--force", action="store_true")

    command = sub.add_parser("import-review")
    command.add_argument("--run-id", default="")
    command.add_argument("--output-root", default=str(DEFAULT_INFORMATION_DIR))
    command.add_argument("--csv", default=None)
    command.add_argument("--review-batch-id", default="")
    command.add_argument("--samples-csv", default=None)
    command.add_argument("--fields-csv", default=None)
    command.add_argument("--review-root", default="data/guideline_information/review_batches")
    command.add_argument("--review-origin", choices=["human", "synthetic-test", "migrated"], default="synthetic-test")

    command = sub.add_parser("build-silver")
    command.add_argument("--review-batch-id", required=True)
    command.add_argument("--review-root", default="data/guideline_information/review_batches")

    command = sub.add_parser("evaluate-review")
    command.add_argument("--review-batch-id", required=True)
    command.add_argument("--review-root", default="data/guideline_information/review_batches")
    command.add_argument("--real-only", action="store_true")

    command = sub.add_parser("complete-v0")
    command.add_argument("--experiment-id", required=True)
    command.add_argument("--review-batch-id", required=True)
    command.add_argument("--review-root", default="data/guideline_information/review_batches")
    command.add_argument("--force-prepare", action="store_true")

    command = sub.add_parser("check-stage-a")
    command.add_argument("--experiment-id", required=True)

    command = sub.add_parser("build-gold")
    command.add_argument("--run-id", required=True)
    command.add_argument("--output-root", default=str(DEFAULT_INFORMATION_DIR))
    command.add_argument("--allow-single-review", action="store_true")

    command = sub.add_parser("evaluate")
    command.add_argument("--run-id", required=True)
    command.add_argument("--output-root", default=str(DEFAULT_INFORMATION_DIR))
    command.add_argument("--real-only", action="store_true")

def _add_legacy_commands(sub) -> None:
    build = sub.add_parser("build-candidates")
    build.add_argument("--evidence-data-dir", default=str(DATA_DIR))
    build.add_argument("--output-root", default=str(DEFAULT_INFORMATION_DIR))
    build.add_argument("--run-id", default=None)

    for name in ["extract", "verify"]:
        command = sub.add_parser(name)
        command.add_argument("--run-id", required=True)
        command.add_argument("--output-root", default=str(DEFAULT_INFORMATION_DIR))

    sample = sub.add_parser("build-samples")
    sample.add_argument("--run-id", required=True)
    sample.add_argument("--output-root", default=str(DEFAULT_INFORMATION_DIR))
    sample.add_argument("--budget", type=int, default=20)
    sample.add_argument("--seed", type=int, default=0)

    sub.add_parser("adjudicate")


def _dispatch(args) -> dict:
    if args.command == "prepare-pilot":
        return prepare_pilot(
            evidence_data_dir=args.evidence_data_dir,
            output_root=args.output_root,
            organization=args.organization,
            pilot_id=args.pilot_id,
            documents=args.documents,
            seed=args.seed,
            max_sections=args.max_sections,
        )
    if args.command == "prepare-experiment":
        return prepare_experiment_batch(
            pilot_id=args.pilot_id,
            pilot_root=args.pilot_root,
            experiment_root=args.experiment_root,
            experiment_id=args.experiment_id,
            source_run_id=args.source_run_id,
            stage_a_size=args.stage_a_size,
            stage_b_size=args.stage_b_size,
            seed=args.seed,
            parent_experiment_id=args.parent_experiment_id,
            restart_reason=args.restart_reason,
        )
    if args.command == "model-config-audit":
        audit = model_config_audit(probe=args.probe, show_models=args.show_models)
        if args.probe:
            from pathlib import Path
            audit_path = Path(args.experiment_root) / args.experiment_id / "model_configuration_audit.json"
            ensure_parent(audit_path).write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            audit["artifact"] = str(audit_path)
        return audit
    if args.command == "run-pilot":
        return run_pilot(
            pilot_id=args.pilot_id,
            run_id=args.run_id,
            output_root=args.output_root,
            client_name=args.client,
            max_items=args.max_items,
            max_model_calls=args.max_model_calls,
            input_candidates_path=args.input,
            experiment_id=args.experiment_id,
            resume=args.resume,
            force=args.force,
            dry_run=args.dry_run,
            fail_fast=args.fail_fast,
            random_audit_rate=args.random_audit_rate,
            seed=args.seed,
        )
    if args.command == "prepare-review":
        return prepare_review_batch(
            experiment_id=args.experiment_id,
            review_batch_id=args.review_batch_id,
            experiment_root=args.experiment_root,
            review_root=args.review_root,
            force=args.force,
        )
    if args.command == "import-review":
        if getattr(args, "review_batch_id", ""):
            return import_review_batch(
                review_batch_id=args.review_batch_id,
                samples_csv=args.samples_csv,
                fields_csv=args.fields_csv,
                review_origin=args.review_origin,
                review_root=args.review_root,
            )
        return _import_review(args)
    if args.command == "build-silver":
        return build_silver_dataset(review_batch_id=args.review_batch_id, review_root=args.review_root)
    if args.command == "evaluate-review":
        return evaluate_review_batch(review_batch_id=args.review_batch_id, review_root=args.review_root, real_only=args.real_only)
    if args.command == "complete-v0":
        return complete_v0(experiment_id=args.experiment_id, review_batch_id=args.review_batch_id, review_root=args.review_root, force_prepare=args.force_prepare)
    if args.command == "check-stage-a":
        return stage_a_status(args.experiment_id)
    if args.command == "build-gold":        return _build_gold(args)
    if args.command == "evaluate":
        return _evaluate(args)
    if args.command == "build-candidates":
        return run_information_pipeline(evidence_data_dir=args.evidence_data_dir, output_root=args.output_root, run_id=args.run_id)
    if args.command == "build-samples":
        return _build_samples(args)
    if args.command in {"extract", "verify", "adjudicate"}:
        return {"ok": True, "command": args.command, "implemented": "use run-pilot for connected pilot flow", "model_sdk_called": False}
    raise SystemExit(f"Unsupported command: {args.command}")


def _load_run(paths: InformationRunPaths):
    candidates = read_models(paths.candidates, RecommendationCandidate) if paths.candidates.exists() else []
    extractions = read_models(paths.extraction_results, ExtractionResult) if paths.extraction_results.exists() else []
    verifications = read_models(paths.verification_results, VerificationResult) if paths.verification_results.exists() else []
    samples = read_models(paths.review_samples, ReviewSample) if paths.review_samples.exists() else []
    return candidates, extractions, verifications, samples


def _import_review(args) -> dict:
    paths = InformationRunPaths.existing(args.output_root, args.run_id)
    candidates, extractions, verifications, samples = _load_run(paths)
    existing = read_models(paths.review_decisions, ReviewDecision) if paths.review_decisions.exists() else []
    decisions = import_review_decisions_csv(
        args.csv or paths.review_csv,
        samples,
        candidates_by_id={item.candidate_id: item for item in candidates},
        extractions_by_id={item.extraction_id: item for item in extractions},
        verifications_by_id={item.verification_id: item for item in verifications},
        existing_review_ids={item.review_id for item in existing},
        review_origin=_review_origin_arg(getattr(args, "review_origin", "synthetic-test")),
    )
    write_models(paths.review_decisions, existing + decisions)
    return {"imported_review_decisions": len(decisions), "total_review_decisions": len(existing) + len(decisions)}


def _review_origin_arg(raw: str) -> ReviewOrigin:
    return {"human": ReviewOrigin.HUMAN, "synthetic-test": ReviewOrigin.SYNTHETIC_TEST, "migrated": ReviewOrigin.MIGRATED}[raw]


def _build_gold(args) -> dict:
    paths = InformationRunPaths.existing(args.output_root, args.run_id)
    candidates, extractions, verifications, samples = _load_run(paths)
    decisions = read_models(paths.review_decisions, ReviewDecision) if paths.review_decisions.exists() else []
    candidate_map = {item.candidate_id: item for item in candidates}
    extraction_map = {item.extraction_id: item for item in extractions}
    verification_map = {item.verification_id: item for item in verifications}
    decisions_by_sample: dict[str, list[ReviewDecision]] = {}
    for decision in decisions:
        decisions_by_sample.setdefault(decision.sample_id, []).append(decision)
    gold: list[GoldRecommendationExample] = []
    for sample in samples:
        if sample.sample_id not in decisions_by_sample or sample.candidate_id not in candidate_map:
            continue
        sample_decisions = decisions_by_sample[sample.sample_id]
        if len(sample_decisions) == 1 and args.allow_single_review:
            approved = sample.model_copy(update={"review_status": "APPROVED"})
            extraction = extraction_map.get(sample.extraction_id or "")
            verification = verification_map.get(sample.verification_id or "")
            gold.append(build_gold_from_review(approved, candidate_map[sample.candidate_id], sample_decisions[0], extraction=extraction, verification=verification, allow_single_reviewer_gold=True))
    write_models(paths.gold_examples, gold)
    return {"gold_examples": len(gold), "allow_single_review": args.allow_single_review}


def _evaluate(args) -> dict:
    paths = InformationRunPaths.existing(args.output_root, args.run_id)
    candidates = read_models(paths.candidates, RecommendationCandidate) if paths.candidates.exists() else []
    extractions = read_models(paths.extraction_results, ExtractionResult) if paths.extraction_results.exists() else []
    validations = read_models(paths.validation_results, ValidationResult) if paths.validation_results.exists() else []
    routes = read_models(paths.route_results, RouteResult) if paths.route_results.exists() else []
    decisions = read_models(paths.review_decisions, ReviewDecision) if paths.review_decisions.exists() else []
    gold = read_models(paths.gold_examples, GoldRecommendationExample) if paths.gold_examples.exists() else []
    if args.real_only:
        report = build_real_model_evaluation_report(candidates_count=len(candidates), extractions=extractions, validations=validations, routes=routes, decisions=decisions, gold_examples=gold)
        report_json = paths.run_dir / "real_model_evaluation_report.json"
        report_md = paths.run_dir / "real_model_evaluation_report.md"
        error_examples = paths.run_dir / "real_model_error_examples.jsonl"
    else:
        report = build_evaluation_report(candidates_count=len(candidates), extractions=extractions, validations=validations, routes=routes, decisions=decisions, gold_examples=gold)
        report_json = paths.evaluation_report_json
        report_md = paths.evaluation_report_md
        error_examples = paths.error_examples
    ensure_parent(report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ensure_parent(report_md).write_text(render_markdown_report(report), encoding="utf-8")
    from src.utils.io import write_jsonl
    write_jsonl(error_examples, build_error_examples(validations=validations, extractions=extractions))
    return {"evaluation_report_json": str(report_json), "evaluation_report_md": str(report_md), "error_examples": str(error_examples), "gold_examples": len(gold)}


def _build_samples(args) -> dict:
    paths = InformationRunPaths.existing(args.output_root, args.run_id)
    candidates = read_models(paths.candidates, RecommendationCandidate)
    samples, manifest = build_review_samples(
        candidates=candidates,
        selectors=[RandomAuditSelector()],
        budgets={"RandomAuditSelector": args.budget},
        sample_batch_id=f"{args.run_id}-sample",
        seed=args.seed,
    )
    write_models(paths.review_samples, samples)
    ensure_parent(paths.sample_manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    main()
