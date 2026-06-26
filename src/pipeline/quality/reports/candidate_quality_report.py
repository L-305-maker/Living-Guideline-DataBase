from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.pipeline.quality.common import report_common
from src.common.process_jsonl import iter_jsonl, write_jsonl


JsonDict = Dict[str, Any]


@dataclass
class RecommendationCandidateStats:
    total: int = 0
    status_counts: Counter[str] = field(default_factory=Counter)
    direction_counts: Counter[str] = field(default_factory=Counter)
    strength_counts: Counter[str] = field(default_factory=Counter)
    certainty_counts: Counter[str] = field(default_factory=Counter)
    source_counts: Counter[str] = field(default_factory=Counter)
    missing_counts: Counter[str] = field(default_factory=Counter)
    quality_note_counts: Counter[str] = field(default_factory=Counter)
    text_counter: Counter[str] = field(default_factory=Counter)
    block_counter: Counter[str] = field(default_factory=Counter)
    text_lengths: List[int] = field(default_factory=list)
    confidence_values: List[float] = field(default_factory=list)


def _text(row: JsonDict) -> str:
    return str(row.get("recommendation_text") or "").strip()


def _block_id(row: JsonDict) -> str:
    payload = row.get("normalized_payload")
    if isinstance(payload, dict):
        return str(payload.get("block_id") or "")
    return ""


def _quality_notes(row: JsonDict) -> List[str]:
    payload = row.get("normalized_payload")
    notes = payload.get("quality_notes") if isinstance(payload, dict) else []
    return [str(note) for note in notes] if isinstance(notes, list) else []


def _confidence(row: JsonDict) -> float:
    try:
        return float(row.get("extraction_confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _update_candidate_stats(stats: RecommendationCandidateStats, row: JsonDict) -> None:
    stats.total += 1
    stats.status_counts[str(row.get("status") or "unknown")] += 1
    stats.direction_counts[str(row.get("direction") or "unknown")] += 1
    stats.strength_counts[str(row.get("strength") or "unknown")] += 1
    stats.certainty_counts[str(row.get("certainty") or "unknown")] += 1
    stats.source_counts[str(row.get("source") or row.get("source_section") or "unknown")] += 1

    text = _text(row)
    stats.text_lengths.append(len(text))
    if text:
        stats.text_counter[text.lower()] += 1
    else:
        stats.missing_counts["recommendation_text"] += 1

    block_id = _block_id(row)
    if block_id:
        stats.block_counter[block_id] += 1
    else:
        stats.missing_counts["block_id"] += 1

    for field in ("candidate_id", "record_id", "guideline_id", "model_trace_id", "source_text"):
        if not row.get(field):
            stats.missing_counts[field] += 1

    for field in ("strength", "certainty", "direction"):
        if row.get(field) == "unclear":
            stats.missing_counts[f"{field}_unclear"] += 1

    for note in _quality_notes(row):
        stats.quality_note_counts[note] += 1

    stats.confidence_values.append(_confidence(row))


def _candidate_summary(stats: RecommendationCandidateStats) -> JsonDict:
    duplicate_texts = sum(1 for count in stats.text_counter.values() if count > 1)
    multi_candidate_blocks = sum(1 for count in stats.block_counter.values() if count > 1)
    avg_text_length = round(sum(stats.text_lengths) / stats.total, 2) if stats.total else 0
    max_text_length = max(stats.text_lengths) if stats.text_lengths else 0
    avg_confidence = round(sum(stats.confidence_values) / stats.total, 4) if stats.total else 0
    return {
        "candidates_count": stats.total,
        "status_counts": dict(stats.status_counts),
        "direction_counts": dict(stats.direction_counts),
        "strength_counts": dict(stats.strength_counts),
        "certainty_counts": dict(stats.certainty_counts),
        "missing_or_unclear_counts": dict(stats.missing_counts),
        "quality_note_counts": dict(stats.quality_note_counts),
        "duplicate_recommendation_texts": duplicate_texts,
        "multi_candidate_blocks": multi_candidate_blocks,
        "avg_text_length": avg_text_length,
        "max_text_length": max_text_length,
        "avg_extraction_confidence": avg_confidence,
    }


def summarize_recommendation_candidates(rows: Iterable[JsonDict]) -> JsonDict:
    stats = RecommendationCandidateStats()
    for row in rows:
        _update_candidate_stats(stats, row)
    return _candidate_summary(stats)


def sample_recommendation_candidates(rows: Iterable[JsonDict], per_group: int = 5, max_text_chars: int = 600) -> List[JsonDict]:
    groups: Dict[str, List[JsonDict]] = defaultdict(list)
    for row in rows:
        labels = [
            f"status:{row.get('status') or 'unknown'}",
            f"direction:{row.get('direction') or 'unknown'}",
            f"strength:{row.get('strength') or 'unknown'}",
        ]
        if row.get("certainty") == "unclear":
            labels.append("certainty_unclear")
        labels.extend(f"note:{note}" for note in _quality_notes(row))

        for label in labels:
            if len(groups[label]) >= per_group:
                continue
            groups[label].append(
                {
                    "sample_group": label,
                    "candidate_id": row.get("candidate_id", ""),
                    "record_id": row.get("record_id", ""),
                    "guideline_id": row.get("guideline_id", ""),
                    "model_trace_id": row.get("model_trace_id", ""),
                    "block_id": _block_id(row),
                    "status": row.get("status", ""),
                    "direction": row.get("direction", ""),
                    "strength": row.get("strength", ""),
                    "certainty": row.get("certainty", ""),
                    "extraction_confidence": row.get("extraction_confidence", 0),
                    "quality_notes": _quality_notes(row),
                    "source_section": row.get("source_section", ""),
                    "recommendation_text": _text(row)[:max_text_chars],
                }
            )

    samples: List[JsonDict] = []
    for label in sorted(groups):
        samples.extend(groups[label])
    return samples


def build_report(input_path: str | Path, sample_size: int = 5, max_text_chars: int = 600) -> JsonDict:
    rows = list(iter_jsonl(input_path))
    return {
        "input_path": str(input_path),
        "summary": summarize_recommendation_candidates(rows),
        "samples": sample_recommendation_candidates(rows, per_group=sample_size, max_text_chars=max_text_chars),
    }


def write_report(report: JsonDict, summary_output: str | Path, samples_output: str | Path) -> None:
    report_common.write_report(summary_output, report["summary"])
    write_jsonl(samples_output, report["samples"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a quality report for RecommendationCandidate JSONL output.")
    parser.add_argument("--input", required=True, help="Input RecommendationCandidate JSONL path.")
    parser.add_argument("--summary-output", required=True, help="Output one-line JSONL summary path.")
    parser.add_argument("--samples-output", required=True, help="Output sampled candidate JSONL path.")
    parser.add_argument("--sample-size", type=int, default=5, help="Samples per group.")
    parser.add_argument("--max-text-chars", type=int, default=600, help="Maximum recommendation text chars per sample.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.input, sample_size=args.sample_size, max_text_chars=args.max_text_chars)
    write_report(report, args.summary_output, args.samples_output)
    summary = report["summary"]
    print(
        "candidates={candidates_count} status={status_counts} unclear={missing_or_unclear_counts}".format(
            **summary
        )
    )


if __name__ == "__main__":
    main()
