"""质量报告文件：读取流水线产物并生成摘要、计数和样本，帮助定位抽取或路由质量问题。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

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
class BlockSummaryStats:
    total: int = 0
    block_type_counts: Counter[str] = field(default_factory=Counter)
    hint_counts: Counter[str] = field(default_factory=Counter)
    source_counts: Counter[str] = field(default_factory=Counter)
    record_counts: Counter[str] = field(default_factory=Counter)
    missing_counts: Counter[str] = field(default_factory=Counter)
    quality_counts: Counter[str] = field(default_factory=Counter)
    token_values: List[int] = field(default_factory=list)
    text_lengths: List[int] = field(default_factory=list)
    section_depths: List[int] = field(default_factory=list)


def _hint_values(row: JsonDict) -> List[str]:
    hints = row.get("candidate_hints")
    return [str(hint) for hint in hints] if isinstance(hints, list) else []


def _text_len(row: JsonDict) -> int:
    return len(str(row.get("text") or ""))


def _token_estimate(row: JsonDict) -> int:
    try:
        return int(row.get("token_estimate") or 0)
    except (TypeError, ValueError):
        return 0


def _update_block_stats(stats: BlockSummaryStats, row: JsonDict) -> None:
    stats.total += 1
    stats.block_type_counts[str(row.get("block_type") or "unknown")] += 1
    stats.source_counts[str(row.get("source") or "unknown")] += 1
    record_id = str(row.get("record_id") or "")
    if record_id:
        stats.record_counts[record_id] += 1
    else:
        stats.missing_counts["record_id"] += 1
    for field in ("block_id", "guideline_id"):
        if not row.get(field):
            stats.missing_counts[field] += 1
    if not str(row.get("text") or "").strip():
        stats.missing_counts["text"] += 1
    hints = _hint_values(row)
    for hint in hints:
        stats.hint_counts[hint] += 1
    if len(hints) > 1:
        stats.quality_counts["multi_hint_blocks"] += 1
    _update_quality_stats(stats, row)
    stats.token_values.append(_token_estimate(row))
    stats.text_lengths.append(_text_len(row))
    section_path = row.get("section_path")
    stats.section_depths.append(len(section_path) if isinstance(section_path, list) else 0)


def _update_quality_stats(stats: BlockSummaryStats, row: JsonDict) -> None:
    quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    if quality.get("is_low_value"):
        stats.quality_counts["low_value_blocks"] += 1
    if quality.get("skip_candidate_extraction"):
        stats.quality_counts["skip_candidate_extraction_blocks"] += 1
    if metadata.get("reference_like"):
        stats.quality_counts["reference_like_blocks"] += 1
    if metadata.get("forced_heading"):
        stats.quality_counts["inline_or_forced_heading_blocks"] += 1


def _block_summary(stats: BlockSummaryStats) -> JsonDict:
    avg_blocks = round(stats.total / len(stats.record_counts), 2) if stats.record_counts else 0
    avg_tokens = round(sum(stats.token_values) / stats.total, 2) if stats.total else 0
    max_tokens = max(stats.token_values) if stats.token_values else 0
    avg_text_len = round(sum(stats.text_lengths) / stats.total, 2) if stats.total else 0
    max_text_len = max(stats.text_lengths) if stats.text_lengths else 0
    avg_section_depth = round(sum(stats.section_depths) / stats.total, 2) if stats.total else 0
    return {
        "records_count": len(stats.record_counts),
        "blocks_count": stats.total,
        "avg_blocks_per_record": avg_blocks,
        "block_type_counts": dict(stats.block_type_counts),
        "candidate_hint_counts": dict(stats.hint_counts),
        "source_counts": dict(stats.source_counts),
        "missing_counts": dict(stats.missing_counts),
        "quality_counts": dict(stats.quality_counts),
        "avg_token_estimate": avg_tokens,
        "max_token_estimate": max_tokens,
        "avg_text_length": avg_text_len,
        "max_text_length": max_text_len,
        "avg_section_depth": avg_section_depth,
    }


def summarize_blocks(rows: Iterable[JsonDict]) -> JsonDict:
    stats = BlockSummaryStats()
    for row in rows:
        _update_block_stats(stats, row)
    return _block_summary(stats)


def sample_blocks(
    rows: Iterable[JsonDict],
    per_group: int = 5,
    max_text_chars: int = 500,
) -> List[JsonDict]:
    groups: Dict[str, List[JsonDict]] = defaultdict(list)
    for row in rows:
        labels = _hint_values(row) or [f"type:{row.get('block_type') or 'unknown'}"]
        if _token_estimate(row) >= 350 or _text_len(row) >= 1600:
            labels.append("long_block")
        if not row.get("guideline_id") or not row.get("record_id"):
            labels.append("missing_link")
        quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if quality.get("skip_candidate_extraction"):
            labels.append("skip_candidate_extraction")
        if metadata.get("reference_like"):
            labels.append("reference_like")

        for label in labels:
            if len(groups[label]) >= per_group:
                continue
            groups[label].append(
                {
                    "sample_group": label,
                    "block_id": row.get("block_id", ""),
                    "record_id": row.get("record_id", ""),
                    "guideline_id": row.get("guideline_id", ""),
                    "source": row.get("source", ""),
                    "title": row.get("title", ""),
                    "section_path": row.get("section_path", []),
                    "block_type": row.get("block_type", ""),
                    "candidate_hints": row.get("candidate_hints", []),
                    "quality": row.get("quality", {}),
                    "token_estimate": row.get("token_estimate", 0),
                    "text": str(row.get("text") or "")[:max_text_chars],
                }
            )

    samples: List[JsonDict] = []
    for label in sorted(groups):
        samples.extend(groups[label])
    return samples


def build_report(input_path: str | Path, sample_size: int = 5, max_text_chars: int = 500) -> JsonDict:
    rows = list(iter_jsonl(input_path))
    return {
        "input_path": str(input_path),
        "summary": summarize_blocks(rows),
        "samples": sample_blocks(rows, per_group=sample_size, max_text_chars=max_text_chars),
    }


def write_report(report: JsonDict, summary_output: str | Path, samples_output: str | Path) -> None:
    report_common.write_report(summary_output, report["summary"])
    write_jsonl(samples_output, report["samples"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a quality report for SourceBlock JSONL output.")
    parser.add_argument("--input", required=True, help="Input SourceBlock JSONL path.")
    parser.add_argument("--summary-output", required=True, help="Output one-line JSONL summary path.")
    parser.add_argument("--samples-output", required=True, help="Output sampled block JSONL path.")
    parser.add_argument("--sample-size", type=int, default=5, help="Samples per hint/type group.")
    parser.add_argument("--max-text-chars", type=int, default=500, help="Maximum text chars per sample.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.input, sample_size=args.sample_size, max_text_chars=args.max_text_chars)
    write_report(report, args.summary_output, args.samples_output)
    summary = report["summary"]
    print(
        "records={records_count} blocks={blocks_count} avg_blocks={avg_blocks_per_record} "
        "hints={candidate_hint_counts}".format(**summary)
    )


if __name__ == "__main__":
    main()

