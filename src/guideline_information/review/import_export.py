"""CSV exchange for human review."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.guideline_information.enums import ReviewDecisionType, ReviewStatus
from src.guideline_information.ids import make_record_id
from src.guideline_information.models import ReviewDecision, ReviewSample


CSV_FIELDS = [
    "sample_id",
    "candidate_id",
    "source_revision_id",
    "source_text_snapshot",
    "reviewer_id",
    "decision",
    "corrected_annotation",
    "issue_codes",
    "comment",
]


def export_review_samples_csv(samples: list[ReviewSample], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "sample_id": sample.sample_id,
                    "candidate_id": sample.candidate_id,
                    "source_revision_id": sample.source_revision_id,
                    "source_text_snapshot": sample.source_text_snapshot,
                    "reviewer_id": "",
                    "decision": "",
                    "corrected_annotation": "{}",
                    "issue_codes": "",
                    "comment": "",
                }
            )


def import_review_decisions_csv(
    path: str | Path,
    samples: list[ReviewSample],
    *,
    existing_review_ids: set[str] | None = None,
) -> list[ReviewDecision]:
    sample_by_id = {sample.sample_id: sample for sample in samples}
    seen = set(existing_review_ids or set())
    decisions: list[ReviewDecision] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for line_no, row in enumerate(csv.DictReader(handle), start=2):
            sample = sample_by_id.get(row.get("sample_id") or "")
            if sample is None:
                raise ValueError(f"Unknown sample_id at CSV line {line_no}")
            if row.get("candidate_id") != sample.candidate_id:
                raise ValueError(f"Immutable candidate_id changed at CSV line {line_no}")
            if row.get("source_revision_id") != sample.source_revision_id:
                raise ValueError(f"source_revision_id mismatch at CSV line {line_no}")
            if row.get("source_text_snapshot") != sample.source_text_snapshot:
                raise ValueError(f"source text snapshot changed at CSV line {line_no}")
            if sample.review_status not in {ReviewStatus.ASSIGNED, ReviewStatus.IN_REVIEW, ReviewStatus.SUBMITTED}:
                raise ValueError(f"Sample status does not allow submission at CSV line {line_no}")
            reviewer_id = (row.get("reviewer_id") or "").strip()
            if not reviewer_id:
                raise ValueError(f"reviewer_id is required at CSV line {line_no}")
            try:
                decision = ReviewDecisionType((row.get("decision") or "").strip())
            except ValueError as exc:
                raise ValueError(f"Invalid decision at CSV line {line_no}") from exc
            try:
                corrected = json.loads(row.get("corrected_annotation") or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid corrected_annotation JSON at CSV line {line_no}") from exc
            if decision == ReviewDecisionType.EDIT and not corrected:
                raise ValueError(f"EDIT requires corrected_annotation at CSV line {line_no}")
            review_id = make_record_id("review", sample.sample_id, reviewer_id, row.get("decision"), row.get("corrected_annotation"))
            if review_id in seen:
                raise ValueError(f"Duplicate review submission at CSV line {line_no}")
            seen.add(review_id)
            issue_codes = [item.strip() for item in (row.get("issue_codes") or "").split("|") if item.strip()]
            decisions.append(
                ReviewDecision(
                    review_id=review_id,
                    sample_id=sample.sample_id,
                    reviewer_id=reviewer_id,
                    review_round=1,
                    decision=decision,
                    corrected_annotation=corrected,
                    issue_codes=issue_codes,
                    comment=row.get("comment") or "",
                )
            )
    return decisions
