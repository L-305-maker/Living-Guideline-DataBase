"""在解析和抽取前，按质量路由已清洗的 source record。

质量门故意保持保守: 它不会丢弃记录，而是把数据流拆成 ready、
layout-repair 和 parse-failed 等输出，让后续阶段避免从畸形文本中抽取推荐。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, TextIO

from src.common.extraction_common import JsonDict
from src.common.process_jsonl import iter_jsonl, write_jsonl_obj


READY = "ready"
NEEDS_LAYOUT_REPAIR = "needs_layout_repair"
PARSE_FAILED = "parse_failed"
SKIPPED = "skipped"

ERROR = "error"
WARNING = "warning"
INFO = "info"

MIN_CLEAN_CHARS = 100
HUGE_SINGLE_SECTION_CHARS = 500_000
LARGE_WEAK_STRUCTURE_CHARS = 50_000
TABLE_HEAVY_MARKER_MIN = 8

SECTION_WORD_RE = re.compile(
    r"\b(?:abstract|background|methods?|results?|discussion|conclusions?|recommendations?|evidence|rationale)\b",
    re.I,
)
TABLE_MARKER_RE = re.compile(r"\b(?:table|figure|forest plot|risk ratio|odds ratio|hazard ratio|95%\s*ci)\b", re.I)
GUIDELINE_SIGNAL_RE = re.compile(
    r"\b(?:guideline|guidance|recommendations?|we recommend|we suggest|practice guideline|consensus statement|nice|who)\b",
    re.I,
)
PAPER_SIGNAL_RE = re.compile(
    r"\b(?:abstract|methods?|results?|discussion|randomi[sz]ed|cohort|systematic review|meta-analysis|journal|trial)\b",
    re.I,
)


@dataclass(frozen=True)
class GateResult:
    """最终路由决策，以及写给下游的增强记录。"""

    record: JsonDict
    status: str
    score: float
    flags: list[JsonDict]
    record_type: str
    record_type_confidence: float
    record_type_reasons: list[str]


@dataclass(frozen=True)
class CleanRecordMetrics:
    """从一条已清洗记录中收集到的可观测质量信号。"""

    raw_content: str
    clean_content: str
    raw_len: int
    clean_len: int
    section_count: int
    table_count: int
    references_len: int
    raw_clean_ratio: float
    table_marker_count: int
    line_metrics: JsonDict
    record_type: str
    record_type_confidence: float
    record_type_reasons: list[str]


@dataclass(frozen=True)
class QualityGatePaths:
    """CLI 和脚本共用的输入/输出路径集合。"""

    input_path: str | Path
    ready_output: str | Path
    layout_repair_output: str | Path
    parse_failed_output: str | Path
    skipped_output: str | Path
    report_output: str | Path


@dataclass(frozen=True)
class GateEvaluation:
    """记录增强前的质量分数、状态和问题标记。"""

    status: str
    score: float
    flags: list[JsonDict]
    metrics: CleanRecordMetrics


@dataclass
class GateRouteStats:
    """路由 JSONL 文件时累计的计数器和示例样本。"""

    counts: Counter[str] = field(default_factory=Counter)
    flags: Counter[str] = field(default_factory=Counter)
    record_types: Counter[str] = field(default_factory=Counter)
    samples: dict[str, list[JsonDict]] = field(default_factory=dict)


def _text(record: JsonDict, *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if value:
            return str(value)
    return ""


def _sections_count(record: JsonDict) -> int:
    sections = record.get("sections")
    return len(sections) if isinstance(sections, list) else 0


def _flag(code: str, severity: str, message: str, evidence: JsonDict | None = None) -> JsonDict:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "evidence": evidence or {},
    }


def _line_metrics(text: str) -> JsonDict:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return {
            "line_count": 0,
            "short_line_ratio": 0.0,
            "average_line_chars": 0.0,
            "max_line_chars": 0,
            "inline_section_heading_hits": 0,
        }

    short_lines = sum(1 for line in lines if len(line) <= 45)
    inline_heading_hits = 0
    for line in lines:
        if len(line) < 120:
            continue
        matches = list(SECTION_WORD_RE.finditer(line))
        inline_heading_hits += sum(1 for match in matches if match.start() > 20)

    return {
        "line_count": len(lines),
        "short_line_ratio": short_lines / len(lines),
        "average_line_chars": sum(len(line) for line in lines) / len(lines),
        "max_line_chars": max(len(line) for line in lines),
        "inline_section_heading_hits": inline_heading_hits,
    }


def detect_record_type(record: JsonDict) -> tuple[str, float, list[str]]:
    """根据文本线索把记录分类为 guideline、paper、mixed 或 unknown。"""

    title = str(record.get("title") or "")
    source = str(record.get("source") or "")
    text = f"{title}\n{source}\n{_text(record, 'clean_content', 'content')[:20_000]}"

    guideline_hits = len(GUIDELINE_SIGNAL_RE.findall(text))
    paper_hits = len(PAPER_SIGNAL_RE.findall(text))
    reasons: list[str] = []
    if guideline_hits:
        reasons.append(f"guideline_signals={guideline_hits}")
    if paper_hits:
        reasons.append(f"paper_signals={paper_hits}")

    if guideline_hits >= 2 and paper_hits >= 3:
        confidence = min(0.9, 0.55 + 0.03 * (guideline_hits + paper_hits))
        return "mixed", round(confidence, 3), reasons
    if guideline_hits > paper_hits:
        margin = guideline_hits - paper_hits
        confidence = min(0.95, 0.6 + 0.08 * margin)
        return "guideline", round(confidence, 3), reasons
    if paper_hits > guideline_hits:
        margin = paper_hits - guideline_hits
        confidence = min(0.95, 0.6 + 0.08 * margin)
        return "paper", round(confidence, 3), reasons
    return "unknown", 0.3, reasons or ["no_clear_record_type_signal"]


def collect_record_metrics(record: JsonDict) -> CleanRecordMetrics:
    """收集长度、版面、表格、参考文献和记录类型相关质量信号。"""

    raw_content = _text(record, "raw_content", "content_markdown", "content", "abstract")
    clean_content = _text(record, "clean_content", "content")
    raw_len = len(raw_content.strip())
    clean_len = len(clean_content.strip())
    record_type, record_type_confidence, record_type_reasons = detect_record_type(record)
    return CleanRecordMetrics(
        raw_content=raw_content,
        clean_content=clean_content,
        raw_len=raw_len,
        clean_len=clean_len,
        section_count=_sections_count(record),
        table_count=int(record.get("table_count") or 0),
        references_len=len(str(record.get("references_text") or "")),
        raw_clean_ratio=clean_len / raw_len if raw_len else 0.0,
        table_marker_count=len(TABLE_MARKER_RE.findall(clean_content)),
        line_metrics=_line_metrics(clean_content),
        record_type=record_type,
        record_type_confidence=record_type_confidence,
        record_type_reasons=record_type_reasons,
    )


def score_content_presence(metrics: CleanRecordMetrics) -> tuple[list[JsonDict], float]:
    """标记缺少足够可用文本、无法安全解析的记录。"""

    flags: list[JsonDict] = []
    score = 1.0
    if metrics.raw_len == 0:
        flags.append(_flag("empty_raw_content", ERROR, "Raw content is empty.", {"raw_length": metrics.raw_len}))
        score = 0.0
    if metrics.clean_len == 0:
        flags.append(_flag("empty_clean_content", ERROR, "Clean content is empty.", {"clean_length": metrics.clean_len}))
        score = 0.0
    elif metrics.clean_len < MIN_CLEAN_CHARS:
        flags.append(
            _flag(
                "short_clean_content",
                ERROR,
                "Clean content is too short for reliable downstream extraction.",
                {"clean_length": metrics.clean_len, "min_clean_chars": MIN_CLEAN_CHARS},
            )
        )
        score = min(score, 0.2)
    return flags, score


def check_section_structure(metrics: CleanRecordMetrics) -> tuple[list[JsonDict], float]:
    """检测可能丢失章节边界的大体量文档。"""

    if metrics.clean_len >= HUGE_SINGLE_SECTION_CHARS and metrics.section_count <= 1:
        return [
            _flag(
                "single_section_huge_document",
                ERROR,
                "Very large clean content has one or zero detected sections; layout repair is required.",
                {"clean_length": metrics.clean_len, "section_count": metrics.section_count},
            )
        ], 0.55
    if metrics.clean_len >= LARGE_WEAK_STRUCTURE_CHARS and metrics.section_count <= 1:
        return [
            _flag(
                "single_section_large_document",
                WARNING,
                "Large clean content has weak section structure.",
                {"clean_length": metrics.clean_len, "section_count": metrics.section_count},
            )
        ], 0.25
    return [], 0.0


def check_cleaning_ratio(metrics: CleanRecordMetrics) -> tuple[list[JsonDict], float]:
    """检测过度清洗或清洗不足导致的 raw/clean 尺寸异常变化。"""

    if metrics.raw_len and metrics.raw_clean_ratio < 0.2:
        return [
            _flag(
                "cleaning_removed_most_content",
                WARNING,
                "Clean content is much smaller than raw content; check for over-cleaning or parse failure.",
                {
                    "raw_length": metrics.raw_len,
                    "clean_length": metrics.clean_len,
                    "clean_to_raw_ratio": round(metrics.raw_clean_ratio, 3),
                },
            )
        ], 0.3
    if metrics.raw_len and metrics.raw_clean_ratio > 1.05:
        return [
            _flag(
                "clean_content_larger_than_raw",
                WARNING,
                "Clean content is unexpectedly larger than raw content.",
                {
                    "raw_length": metrics.raw_len,
                    "clean_length": metrics.clean_len,
                    "clean_to_raw_ratio": round(metrics.raw_clean_ratio, 3),
                },
            )
        ], 0.15
    return [], 0.0


def check_layout_quality(metrics: CleanRecordMetrics) -> tuple[list[JsonDict], float]:
    """检测可能破坏章节解析的扁平化或粘连 PDF 文本。"""

    line_metrics = metrics.line_metrics
    if metrics.clean_len >= 5_000 and int(line_metrics["line_count"]) == 1 and int(line_metrics["inline_section_heading_hits"]) >= 4:
        return [
            _flag(
                "single_line_flattened_text",
                WARNING,
                "Clean content is stored as one long line; downstream section parsing may be weak.",
                line_metrics,
            )
        ], 0.1
    if (
        metrics.clean_len >= 5_000
        and metrics.section_count <= 1
        and int(line_metrics["line_count"]) >= 10
        and int(line_metrics["inline_section_heading_hits"]) >= 4
        and (float(line_metrics["short_line_ratio"]) >= 0.35 or float(line_metrics["average_line_chars"]) <= 140)
    ):
        return [
            _flag(
                "suspected_multi_column_glue",
                ERROR,
                "Section headings appear inside long lines; PDF layout may be glued across columns.",
                line_metrics,
            )
        ], 0.4
    return [], 0.0


def check_table_quality(metrics: CleanRecordMetrics) -> tuple[list[JsonDict], float]:
    """当文本表格信号很多但缺少结构化表格 payload 时打标。"""

    if metrics.table_count == 0 and metrics.table_marker_count >= TABLE_HEAVY_MARKER_MIN:
        return [
            _flag(
                "table_heavy_without_structured_tables",
                WARNING,
                "Text contains many table/effect-size markers but no structured tables.",
                {"table_marker_count": metrics.table_marker_count, "table_count": metrics.table_count},
            )
        ], 0.25
    return [], 0.0


def check_reference_balance(metrics: CleanRecordMetrics) -> tuple[list[JsonDict], float]:
    """检测被参考文献文本主导、临床内容占比较低的记录。"""

    if metrics.references_len and metrics.clean_len and metrics.references_len / max(metrics.clean_len, 1) > 0.4:
        return [
            _flag(
                "reference_heavy_document",
                WARNING,
                "References are large relative to clean content.",
                {"references_length": metrics.references_len, "clean_length": metrics.clean_len},
            )
        ], 0.1
    return [], 0.0


def check_record_type(metrics: CleanRecordMetrics) -> tuple[list[JsonDict], float]:
    """标记 guideline 与 paper 路由信号过弱的记录。"""

    if metrics.record_type == "unknown":
        return [
            _flag(
                "record_type_unknown",
                WARNING,
                "Record type could not be confidently classified.",
                {"record_type_confidence": metrics.record_type_confidence},
            )
        ], 0.1
    return [], 0.0


def evaluate_cleaned_record(record: JsonDict) -> GateResult:
    """运行全部质量检查，并返回可路由的增强记录。"""

    # 质量门的核心思路：
    # 先把 record 转成一组可观测指标，再让每个 check 返回“问题标记 + 扣分”。
    # 最终 status 由硬性问题、严重程度和总分共同决定。
    metrics = collect_record_metrics(record)
    flags, score = score_content_presence(metrics)
    for check in (
        check_section_structure,
        check_cleaning_ratio,
        check_layout_quality,
        check_table_quality,
        check_reference_balance,
        check_record_type,
    ):
        new_flags, penalty = check(metrics)
        flags.extend(new_flags)
        score -= penalty

    score = max(0.0, min(1.0, score))
    status = status_for(score, flags)
    evaluation = GateEvaluation(status, score, flags, metrics)
    enriched = enrich_record(record, evaluation)
    return GateResult(
        enriched,
        status,
        score,
        flags,
        metrics.record_type,
        metrics.record_type_confidence,
        metrics.record_type_reasons,
    )


def status_for(score: float, flags: list[JsonDict]) -> str:
    """把质量分数和硬性问题标记转换成下游路由状态。"""

    codes = {str(flag.get("code") or "") for flag in flags}
    severities = {str(flag.get("severity") or "") for flag in flags}

    # 内容为空或太短属于“无法安全解析”，直接进入 parse_failed。
    if "empty_raw_content" in codes or "empty_clean_content" in codes or "short_clean_content" in codes:
        return PARSE_FAILED
    # 版式坏但内容可能还在的记录进入 layout repair 队列，后续有机会修复后再跑。
    if "single_section_huge_document" in codes or "suspected_multi_column_glue" in codes:
        return NEEDS_LAYOUT_REPAIR
    if score < 0.35:
        return NEEDS_LAYOUT_REPAIR
    if ERROR in severities:
        return PARSE_FAILED
    return READY


def enrich_record(
    record: JsonDict,
    evaluation: GateEvaluation,
) -> JsonDict:
    """在不丢失原始 cleaned payload 的前提下附加质量元数据。"""

    # enrich_record 不改变原始业务字段的含义，只附加 cleaning_status、score、flags 等审计信息。
    # 下游 parser 仍读取 content/clean_content，但报告和排障可以读取这些质量字段。
    enriched = dict(record)
    existing_warnings = list(enriched.get("cleaning_warnings") or [])
    gate_warnings = [
        {
            "warning_type": flag["code"],
            "severity": flag["severity"],
            "message": flag["message"],
            "evidence": flag.get("evidence", {}),
        }
        for flag in evaluation.flags
        if flag.get("severity") != INFO
    ]
    enriched["cleaning_status"] = evaluation.status
    enriched["cleaning_quality_score"] = round(evaluation.score, 3)
    enriched["cleaning_quality_flags"] = evaluation.flags
    enriched["cleaning_warnings"] = existing_warnings + gate_warnings
    enriched["record_type"] = evaluation.metrics.record_type
    enriched["record_type_confidence"] = evaluation.metrics.record_type_confidence
    enriched["record_type_reasons"] = evaluation.metrics.record_type_reasons
    direct = dict(enriched.get("direct_extraction") or {})
    direct.update(
        {
            "cleaning_status": evaluation.status,
            "cleaning_quality_score": round(evaluation.score, 3),
            "record_type": evaluation.metrics.record_type,
            "record_type_confidence": evaluation.metrics.record_type_confidence,
        }
    )
    enriched["direct_extraction"] = direct
    return enriched


def _open_output(path: str | Path) -> TextIO:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output.open("w", encoding="utf-8")


def validate_route_paths(input_path: str | Path, output_paths: Iterable[str | Path]) -> None:
    """脚本运行质量门时，防止输出路径误覆盖输入文件。"""

    input_resolved = Path(input_path).resolve()
    resolved_outputs = [Path(path).resolve() for path in output_paths]
    if input_resolved in resolved_outputs:
        raise ValueError("Cleaning quality gate output paths must not overwrite the input file.")
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise ValueError("Cleaning quality gate output paths must be distinct.")


def update_route_stats(stats: GateRouteStats, result: GateResult) -> None:
    """更新聚合计数器，并为每类问题保留少量示例。"""

    stats.counts[result.status] += 1
    stats.record_types[result.record_type] += 1
    for flag in result.flags:
        code = str(flag.get("code") or "")
        stats.flags[code] += 1
        if code and len(stats.samples.setdefault(code, [])) < 3:
            stats.samples[code].append(
                {
                    "record_id": result.record.get("record_id", ""),
                    "title": result.record.get("title", ""),
                    "source": result.record.get("source", ""),
                    "status": result.status,
                    "score": result.score,
                    "evidence": flag.get("evidence", {}),
                }
            )


def route_records(input_path: str | Path, handles: dict[str, TextIO]) -> GateRouteStats:
    """逐行评估 JSONL，并写入对应状态的输出句柄。"""

    stats = GateRouteStats()
    for row in iter_jsonl(input_path):
        # 单条记录在这里被分诊：ready 继续主流程，repair/failed/skipped 写入各自队列。
        result = evaluate_cleaned_record(row)
        update_route_stats(stats, result)
        write_jsonl_obj(handles[result.status], result.record)
    return stats


def build_route_report(paths: QualityGatePaths, stats: GateRouteStats) -> JsonDict:
    """生成供审计与运行日志消费的单行 JSONL 报告。"""

    return {
        "input": str(paths.input_path),
        "outputs": {
            READY: str(paths.ready_output),
            NEEDS_LAYOUT_REPAIR: str(paths.layout_repair_output),
            PARSE_FAILED: str(paths.parse_failed_output),
            SKIPPED: str(paths.skipped_output),
        },
        "total_records": sum(stats.counts.values()),
        "status_counts": dict(stats.counts),
        "record_type_counts": dict(stats.record_types),
        "flag_counts": dict(stats.flags),
        "flag_samples": stats.samples,
    }


def route_file(paths: QualityGatePaths) -> JsonDict:
    """路由 cleaned-record JSONL，并写出各状态对应的输出文件。"""

    validate_route_paths(
        paths.input_path,
        [paths.ready_output, paths.layout_repair_output, paths.parse_failed_output, paths.skipped_output, paths.report_output],
    )
    with (
        _open_output(paths.ready_output) as ready_handle,
        _open_output(paths.layout_repair_output) as layout_handle,
        _open_output(paths.parse_failed_output) as failed_handle,
        _open_output(paths.skipped_output) as skipped_handle,
    ):
        handles = {
            READY: ready_handle,
            NEEDS_LAYOUT_REPAIR: layout_handle,
            PARSE_FAILED: failed_handle,
            SKIPPED: skipped_handle,
        }
        stats = route_records(paths.input_path, handles)

    report = build_route_report(paths, stats)
    Path(paths.report_output).parent.mkdir(parents=True, exist_ok=True)
    with Path(paths.report_output).open("w", encoding="utf-8") as handle:
        write_jsonl_obj(handle, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route cleaned records through the cleaning quality gate.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--ready-output", required=True)
    parser.add_argument("--layout-repair-output", required=True)
    parser.add_argument("--parse-failed-output", required=True)
    parser.add_argument("--skipped-output", required=True)
    parser.add_argument("--report-output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = route_file(
        QualityGatePaths(
            args.input,
            args.ready_output,
            args.layout_repair_output,
            args.parse_failed_output,
            args.skipped_output,
            args.report_output,
        )
    )
    print(
        "Cleaning quality gate routed "
        f"{report['total_records']} records: "
        f"{json.dumps(report['status_counts'], ensure_ascii=False, sort_keys=True)}"
    )


if __name__ == "__main__":
    main()
