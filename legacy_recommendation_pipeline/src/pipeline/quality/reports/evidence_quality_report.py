"""质量报告文件：读取流水线产物并生成摘要、计数和样本，帮助定位抽取或路由质量问题。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.common.process_jsonl import iter_jsonl, write_jsonl
from src.pipeline.quality.common import report_common


JsonDict = Dict[str, Any]


@dataclass
class EvidenceSummaryStats:
    total: int = 0
    status_counts: Counter[str] = field(default_factory=Counter)
    study_design_counts: Counter[str] = field(default_factory=Counter)
    direction_counts: Counter[str] = field(default_factory=Counter)
    missing_counts: Counter[str] = field(default_factory=Counter)
    pico_reason_counts: Counter[str] = field(default_factory=Counter)
    rec_reason_counts: Counter[str] = field(default_factory=Counter)
    confidence_values: List[float] = field(default_factory=list)
    trace_status: Counter[str] = field(default_factory=Counter)
    skipped: Counter[str] = field(default_factory=Counter)


def _payload(row: JsonDict) -> JsonDict:
    value = row.get("normalized_payload")
    return value if isinstance(value, dict) else {}


def _confidence(row: JsonDict) -> float:
    try:
        return float(row.get("extraction_confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _outcomes(row: JsonDict) -> List[JsonDict]:
    value = row.get("outcomes_extracted")
    return value if isinstance(value, list) else []


def _update_evidence_stats(stats: EvidenceSummaryStats, row: JsonDict) -> None:
    stats.total += 1
    stats.status_counts[str(row.get("screening_status") or "unknown")] += 1
    stats.study_design_counts[str(row.get("study_design") or "unknown")] += 1
    stats.direction_counts[str(row.get("effect_direction") or "unknown")] += 1
    confidence = _confidence(row)
    stats.confidence_values.append(confidence)
    payload = _payload(row)
    stats.pico_reason_counts[str(payload.get("pico_association_reason") or "unknown")] += 1
    stats.rec_reason_counts[str(payload.get("recommendation_association_reason") or "unknown")] += 1

    for field in ("evidence_id", "paper_id", "pico_id", "source_record_id", "source_block_id", "source_text"):
        if not row.get(field):
            stats.missing_counts[field] += 1
    if not row.get("recommendation_candidate_id"):
        stats.missing_counts["recommendation_candidate_id"] += 1
    if row.get("study_design") == "unclear":
        stats.missing_counts["study_design_unclear"] += 1
    if row.get("effect_direction") == "uncertain":
        stats.missing_counts["effect_direction_uncertain"] += 1
    if not _outcomes(row):
        stats.missing_counts["outcomes_empty"] += 1
    if not row.get("effect_size"):
        stats.missing_counts["effect_size_empty"] += 1
    if confidence < 0.55:
        stats.missing_counts["low_confidence"] += 1


def _update_trace_stats(stats: EvidenceSummaryStats, traces: Iterable[JsonDict]) -> None:
    for trace in traces:
        stats.trace_status["success" if trace.get("success") else "failed"] += 1
        parsed = trace.get("parsed_output")
        if isinstance(parsed, dict) and parsed.get("skipped_reason"):
            stats.skipped[str(parsed["skipped_reason"])] += 1


def _evidence_summary(stats: EvidenceSummaryStats) -> JsonDict:
    avg_confidence = round(sum(stats.confidence_values) / stats.total, 4) if stats.total else 0.0
    return {
        "evidence_items": stats.total,
        "screening_status_counts": dict(stats.status_counts),
        "study_design_counts": dict(stats.study_design_counts),
        "effect_direction_counts": dict(stats.direction_counts),
        "missing_or_unclear_counts": dict(stats.missing_counts),
        "pico_association_counts": dict(stats.pico_reason_counts),
        "recommendation_association_counts": dict(stats.rec_reason_counts),
        "trace_success_counts": dict(stats.trace_status),
        "skipped_trace_counts": dict(stats.skipped),
        "avg_extraction_confidence": avg_confidence,
    }


def summarize_evidence(rows: Iterable[JsonDict], traces: Iterable[JsonDict]) -> JsonDict:
    stats = EvidenceSummaryStats()
    for row in rows:
        _update_evidence_stats(stats, row)
    _update_trace_stats(stats, traces)
    return _evidence_summary(stats)


def sample_evidence(rows: Iterable[JsonDict], per_group: int = 5, max_text_chars: int = 600) -> List[JsonDict]:
    groups: Dict[str, List[JsonDict]] = defaultdict(list)
    for row in rows:
        labels = [
            f"status:{row.get('screening_status') or 'unknown'}",
            f"study:{row.get('study_design') or 'unknown'}",
            f"direction:{row.get('effect_direction') or 'unknown'}",
        ]
        if not row.get("recommendation_candidate_id"):
            labels.append("missing_recommendation_link")
        if not _outcomes(row):
            labels.append("outcomes_empty")
        if _confidence(row) < 0.55:
            labels.append("low_confidence")

        payload = _payload(row)
        for label in labels:
            report_common.append_limited(
                groups,
                label,
                {
                    "sample_group": label,
                    "evidence_id": row.get("evidence_id", ""),
                    "paper_id": row.get("paper_id", ""),
                    "pico_id": row.get("pico_id", ""),
                    "recommendation_candidate_id": row.get("recommendation_candidate_id", ""),
                    "source_record_id": row.get("source_record_id", ""),
                    "source_block_id": row.get("source_block_id", ""),
                    "screening_status": row.get("screening_status", ""),
                    "study_design": row.get("study_design", ""),
                    "effect_direction": row.get("effect_direction", ""),
                    "extraction_confidence": row.get("extraction_confidence", 0),
                    "pico_association_reason": payload.get("pico_association_reason", ""),
                    "recommendation_association_reason": payload.get("recommendation_association_reason", ""),
                    "outcomes_extracted": row.get("outcomes_extracted", []),
                    "effect_size": row.get("effect_size", {}),
                    "confidence_interval": row.get("confidence_interval"),
                    "source_section": row.get("source_section", ""),
                    "source_text": report_common.truncate(row.get("source_text"), max_text_chars),
                },
                per_group,
            )
    return report_common.flatten_groups(groups)


def build_report(
    input_path: str | Path,
    traces_input: str | Path,
    sample_size: int = 5,
    max_text_chars: int = 600,
) -> JsonDict:
    rows = list(iter_jsonl(input_path))
    traces = list(iter_jsonl(traces_input))
    return {
        "input_path": str(input_path),
        "traces_input": str(traces_input),
        "summary": summarize_evidence(rows, traces),
        "samples": sample_evidence(rows, per_group=sample_size, max_text_chars=max_text_chars),
    }


def write_report(report: JsonDict, summary_output: str | Path, samples_output: str | Path) -> None:
    report_common.write_report(summary_output, report["summary"])
    write_jsonl(samples_output, report["samples"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a quality report for EvidenceItem JSONL output.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--traces-input", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--samples-output", required=True)
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--max-text-chars", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.input, args.traces_input, sample_size=args.sample_size, max_text_chars=args.max_text_chars)
    write_report(report, args.summary_output, args.samples_output)
    summary = report["summary"]
    print("evidence_items={evidence_items} status={screening_status_counts} missing={missing_or_unclear_counts}".format(**summary))


if __name__ == "__main__":
    main()

