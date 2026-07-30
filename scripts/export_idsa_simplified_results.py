from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_INPUT = Path("information/IDSA/integrated/idsa_information_integrated_v1.jsonl")
DEFAULT_OUTPUT_DIR = Path("information/IDSA/final")
CORE_FIELDS = [
    "recommendation_text",
    "direction",
    "strength",
    "certainty",
    "population",
    "interventions",
    "dosage",
    "duration",
    "conditions",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export simplified IDSA information extraction results.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    rows = _read_jsonl(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    minimal_path = args.output_dir / "idsa_recommendations_minimal.jsonl"
    review_path = args.output_dir / "idsa_recommendations_review.jsonl"
    manifest_path = args.output_dir / "idsa_recommendations_simplified_manifest.json"

    minimal = [_minimal_record(row) for row in rows]
    review = [_review_record(row, min_row) for row, min_row in zip(rows, minimal, strict=True)]
    _write_jsonl(minimal_path, minimal)
    _write_jsonl(review_path, review)

    status_counts = Counter(row["status"] for row in minimal)
    route_counts = Counter(row["route"] for row in minimal)
    formal_count = sum(1 for row in minimal if row["is_formal_recommendation"])
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "input": str(args.input),
        "outputs": {
            "minimal": str(minimal_path),
            "review": str(review_path),
        },
        "counts": {
            "rows": len(minimal),
            "status": dict(status_counts),
            "route": dict(route_counts),
            "formal_recommendations": formal_count,
        },
        "notes": [
            "minimal keeps only analysis-ready labels and core extraction fields",
            "review adds source text and context for human review",
            "full audit data remains in the integrated source file",
            "when extraction and extraction_failure both exist, extraction takes precedence in simplified status",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _minimal_record(row: dict[str, Any]) -> dict[str, Any]:
    extraction = _dict(row.get("extraction"))
    route = _dict(row.get("route"))
    validation = _dict(row.get("validation"))
    extraction_failure = _dict(row.get("extraction_failure"))
    verification_failure = _dict(row.get("verification_failure"))

    has_extraction = bool(extraction)
    status = "EXTRACTED" if has_extraction else row.get("integration_status") or "NOT_PROCESSED"
    failure_code = None if has_extraction else extraction_failure.get("failure_code")
    verification_status = _verification_status(row)

    record = {
        "candidate_id": row.get("candidate_id", ""),
        "doc_id": row.get("doc_id", ""),
        "title": row.get("title", ""),
        "section_path": row.get("section_path") or [],
        "source_block_key": row.get("source_block_key", ""),
        "source_revision_id": row.get("source_revision_id", ""),
        "status": status,
        "route": route.get("route") or "NO_ROUTE",
        "route_reasons": route.get("reasons") or [],
        "validation_status": validation.get("overall_status") or "NO_VALIDATION",
        "verification_status": verification_status,
        "failure_code": failure_code,
        "verification_failure_code": verification_failure.get("failure_code") if verification_failure else None,
        "is_formal_recommendation": bool(extraction.get("is_formal_recommendation")) if has_extraction else None,
        "recommendation_type": extraction.get("recommendation_type") if has_extraction else None,
    }
    for field in CORE_FIELDS:
        record[field] = extraction.get(field) if has_extraction else _empty_value(field)
    record["evidence_quotes"] = _evidence_quotes(extraction.get("field_evidence") if has_extraction else {})
    return record


def _review_record(row: dict[str, Any], minimal: dict[str, Any]) -> dict[str, Any]:
    return {
        **minimal,
        "candidate_text": row.get("candidate_text", ""),
        "context_before": row.get("context_before", ""),
        "context_after": row.get("context_after", ""),
        "candidate_block_type": _dict(row.get("candidate_profile")).get("block_type", ""),
        "hard_negative": bool(_dict(row.get("candidate_profile")).get("hard_negative")),
    }


def _verification_status(row: dict[str, Any]) -> str:
    if row.get("verification_failure"):
        return "FAILED"
    if row.get("verification"):
        return "VERIFIED"
    if row.get("extraction"):
        return "NOT_RUN"
    return "NO_EXTRACTION"


def _evidence_quotes(field_evidence: Any) -> dict[str, str]:
    if not isinstance(field_evidence, dict):
        return {}
    quotes: dict[str, str] = {}
    for field, items in field_evidence.items():
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict) and item.get("quote"):
                quotes[str(field)] = str(item["quote"])
                break
    return quotes


def _empty_value(field: str) -> Any:
    return [] if field in {"interventions", "conditions"} else None


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
