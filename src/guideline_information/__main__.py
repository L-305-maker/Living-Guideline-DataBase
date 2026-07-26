"""CLI for guideline information processing."""

from __future__ import annotations

import argparse
import json

from src.guideline_information.models import RecommendationCandidate
from src.guideline_information.orchestration.run_pipeline import run_information_pipeline
from src.guideline_information.paths import DEFAULT_INFORMATION_DIR, InformationRunPaths
from src.guideline_information.repository import read_models, write_models
from src.guideline_information.review.selectors import RandomAuditSelector, build_review_samples
from src.utils.io import DATA_DIR, ensure_parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Guideline information processing")
    sub = parser.add_subparsers(dest="command", required=True)
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
    sub.add_parser("evaluate")
    args = parser.parse_args()

    if args.command == "build-candidates":
        payload = run_information_pipeline(
            evidence_data_dir=args.evidence_data_dir,
            output_root=args.output_root,
            run_id=args.run_id,
        )
    elif args.command == "build-samples":
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
        payload = manifest
    elif args.command in {"extract", "verify", "adjudicate", "evaluate"}:
        payload = {"ok": True, "command": args.command, "implemented": "protocol_or_placeholder", "model_sdk_called": False}
    else:  # pragma: no cover
        raise SystemExit(f"Unsupported command: {args.command}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
