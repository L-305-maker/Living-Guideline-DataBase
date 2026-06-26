from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.common.process_jsonl import iter_jsonl, write_jsonl
from src.pipeline.quality.common import report_common


JsonDict = Dict[str, Any]


def _payload(row: JsonDict) -> JsonDict:
    value = row.get("normalized_payload")
    return value if isinstance(value, dict) else {}


def _text(row: JsonDict, field: str) -> str:
    return str(row.get(field) or "").strip()


def _confidence(row: JsonDict) -> float:
    try:
        return float(row.get("extraction_confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _outcomes(row: JsonDict) -> List[JsonDict]:
    value = row.get("outcomes")
    return value if isinstance(value, list) else []


def summarize_picos(rows: Iterable[JsonDict]) -> JsonDict:
    total = 0
    status_counts: Counter[str] = Counter()
    priority_counts: Counter[str] = Counter()
    section_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    confidence_values: List[float] = []
    outcome_counts: List[int] = []
    block_counter: Counter[str] = Counter()

    for row in rows:
        total += 1
        status_counts[str(row.get("status") or "unknown")] += 1
        priority_counts[str(row.get("priority") or "unknown")] += 1
        section_counts[str(row.get("source_section") or "unknown")] += 1
        confidence_values.append(_confidence(row))
        outcome_counts.append(len(_outcomes(row)))

        block_id = str(row.get("source_block_id") or _payload(row).get("block_id") or "")
        if block_id:
            block_counter[block_id] += 1
        else:
            missing_counts["source_block_id"] += 1

        for field in ("pico_id", "guideline_id", "clinical_question", "source_record_id", "source_text"):
            if not row.get(field):
                missing_counts[field] += 1
        if _text(row, "population").lower() == "unclear":
            missing_counts["population_unclear"] += 1
        if _text(row, "intervention").lower() == "unclear":
            missing_counts["intervention_unclear"] += 1
        if not _outcomes(row):
            missing_counts["outcomes_empty"] += 1
        if _confidence(row) < 0.55:
            missing_counts["low_confidence"] += 1

    avg_confidence = round(sum(confidence_values) / total, 4) if total else 0.0
    avg_outcomes = round(sum(outcome_counts) / total, 2) if total else 0.0
    multi_pico_blocks = sum(1 for count in block_counter.values() if count > 1)

    return {
        "pico_questions": total,
        "status_counts": dict(status_counts),
        "priority_counts": dict(priority_counts),
        "missing_or_unclear_counts": dict(missing_counts),
        "avg_extraction_confidence": avg_confidence,
        "avg_outcomes_per_pico": avg_outcomes,
        "multi_pico_blocks": multi_pico_blocks,
        "top_source_sections": dict(section_counts.most_common(20)),
    }


def sample_picos(rows: Iterable[JsonDict], per_group: int = 5, max_text_chars: int = 600) -> List[JsonDict]:
    groups: Dict[str, List[JsonDict]] = defaultdict(list)
    for row in rows:
        labels = [f"status:{row.get('status') or 'unknown'}"]
        if _text(row, "population").lower() == "unclear":
            labels.append("population_unclear")
        if _text(row, "intervention").lower() == "unclear":
            labels.append("intervention_unclear")
        if not _outcomes(row):
            labels.append("outcomes_empty")
        if _confidence(row) < 0.55:
            labels.append("low_confidence")

        for label in labels:
            report_common.append_limited(
                groups,
                label,
                {
                    "sample_group": label,
                    "pico_id": row.get("pico_id", ""),
                    "guideline_id": row.get("guideline_id", ""),
                    "source_record_id": row.get("source_record_id", ""),
                    "source_block_id": row.get("source_block_id", ""),
                    "status": row.get("status", ""),
                    "extraction_confidence": row.get("extraction_confidence", 0),
                    "source_section": row.get("source_section", ""),
                    "clinical_question": report_common.truncate(row.get("clinical_question"), max_text_chars),
                    "population": report_common.truncate(row.get("population"), 240),
                    "intervention": report_common.truncate(row.get("intervention"), 240),
                    "comparator": report_common.truncate(row.get("comparator"), 160),
                    "outcomes": row.get("outcomes", []),
                    "source_text": report_common.truncate(row.get("source_text"), max_text_chars),
                },
                per_group,
            )
    return report_common.flatten_groups(groups)


def build_report(input_path: str | Path, sample_size: int = 5, max_text_chars: int = 600) -> JsonDict:
    rows = list(iter_jsonl(input_path))
    return {
        "input_path": str(input_path),
        "summary": summarize_picos(rows),
        "samples": sample_picos(rows, per_group=sample_size, max_text_chars=max_text_chars),
    }


def write_report(report: JsonDict, summary_output: str | Path, samples_output: str | Path) -> None:
    report_common.write_report(summary_output, report["summary"])
    write_jsonl(samples_output, report["samples"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a quality report for PicoQuestion JSONL output.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--samples-output", required=True)
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--max-text-chars", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.input, sample_size=args.sample_size, max_text_chars=args.max_text_chars)
    write_report(report, args.summary_output, args.samples_output)
    summary = report["summary"]
    print("pico_questions={pico_questions} status={status_counts} missing={missing_or_unclear_counts}".format(**summary))


if __name__ == "__main__":
    main()
