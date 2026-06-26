from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.pipeline.quality.common import report_common
from src.common.process_jsonl import iter_jsonl, write_jsonl


JsonDict = Dict[str, Any]


def _parsed(row: JsonDict) -> JsonDict:
    parsed = row.get("parsed_result")
    return parsed if isinstance(parsed, dict) else {}


def _avg(values: List[float]) -> Optional[float]:
    return round(sum(values) / len(values), 4) if values else None


def _is_auto_merge_ready(row: JsonDict) -> bool:
    parsed = _parsed(row)
    if row.get("task_type") == "grade_candidate_review":
        return (
            row.get("status") == "validated"
            and parsed.get("is_valid_grade") is True
            and parsed.get("needs_human_review") is False
            and isinstance(parsed.get("confidence"), (int, float))
            and float(parsed["confidence"]) >= 0.7
        )
    return (
        row.get("status") == "validated"
        and parsed.get("is_valid_recommendation") is True
        and parsed.get("needs_human_review") is False
        and isinstance(parsed.get("confidence"), (int, float))
        and float(parsed["confidence"]) >= 0.7
    )


@dataclass
class OutputSummaryStats:
    status_counts: Counter[str] = field(default_factory=Counter)
    task_counts: Counter[str] = field(default_factory=Counter)
    validation_errors: Counter[str] = field(default_factory=Counter)
    valid_counts: Counter[str] = field(default_factory=Counter)
    valid_grade_counts: Counter[str] = field(default_factory=Counter)
    human_review_counts: Counter[str] = field(default_factory=Counter)
    strength_counts: Counter[str] = field(default_factory=Counter)
    certainty_counts: Counter[str] = field(default_factory=Counter)
    direction_counts: Counter[str] = field(default_factory=Counter)
    grade_system_counts: Counter[str] = field(default_factory=Counter)
    grade_domain_counts: Dict[str, Counter[str]] = field(
        default_factory=lambda: {
            "risk_of_bias": Counter(),
            "inconsistency": Counter(),
            "indirectness": Counter(),
            "imprecision": Counter(),
            "publication_bias": Counter(),
        }
    )
    confidence_values: List[float] = field(default_factory=list)
    runtime_values: List[float] = field(default_factory=list)
    total_tokens: List[float] = field(default_factory=list)
    prompt_tokens: List[float] = field(default_factory=list)
    completion_tokens: List[float] = field(default_factory=list)
    trace_success_counts: Counter[str] = field(default_factory=Counter)


def _collect_output_stats(outputs: Iterable[JsonDict]) -> OutputSummaryStats:
    stats = OutputSummaryStats()
    for row in outputs:
        parsed = _parsed(row)
        task_type = str(row.get("task_type") or "unknown")
        stats.status_counts[str(row.get("status") or "unknown")] += 1
        stats.task_counts[task_type] += 1
        for error in row.get("validation_errors") or []:
            stats.validation_errors[str(error)] += 1
        if task_type == "grade_candidate_review":
            stats.valid_grade_counts[str(parsed.get("is_valid_grade"))] += 1
            stats.grade_system_counts[str(parsed.get("grade_system"))] += 1
            for field, counter in stats.grade_domain_counts.items():
                counter[str(parsed.get(field))] += 1
        else:
            stats.valid_counts[str(parsed.get("is_valid_recommendation"))] += 1
            stats.direction_counts[str(parsed.get("direction"))] += 1
        stats.human_review_counts[str(parsed.get("needs_human_review"))] += 1
        stats.strength_counts[str(parsed.get("strength"))] += 1
        stats.certainty_counts[str(parsed.get("certainty"))] += 1
        if isinstance(parsed.get("confidence"), (int, float)):
            stats.confidence_values.append(float(parsed["confidence"]))
    return stats


def _collect_trace_stats(traces: Iterable[JsonDict], stats: OutputSummaryStats) -> None:
    for trace in traces:
        stats.trace_success_counts["success" if trace.get("success") else "failed"] += 1
        if isinstance(trace.get("runtime_ms"), int):
            stats.runtime_values.append(float(trace["runtime_ms"]))
        usage = trace.get("token_usage")
        usage = usage if isinstance(usage, dict) else {}
        if isinstance(usage.get("total_tokens"), int):
            stats.total_tokens.append(float(usage["total_tokens"]))
        if isinstance(usage.get("prompt_tokens"), int):
            stats.prompt_tokens.append(float(usage["prompt_tokens"]))
        if isinstance(usage.get("completion_tokens"), int):
            stats.completion_tokens.append(float(usage["completion_tokens"]))


def _rejected_by_llm(outputs: Iterable[JsonDict]) -> int:
    return sum(
        1
        for row in outputs
        if row.get("status") == "validated"
        and (
            _parsed(row).get("is_valid_recommendation") is False
            or _parsed(row).get("is_valid_grade") is False
        )
    )


def summarize_outputs(outputs: List[JsonDict], traces: List[JsonDict]) -> JsonDict:
    """汇总 LLM 输出质量、校验错误、人工复核压力和调用成本信号。"""

    stats = _collect_output_stats(outputs)
    _collect_trace_stats(traces, stats)
    auto_merge_ready = sum(1 for row in outputs if _is_auto_merge_ready(row))
    validated_outputs = stats.status_counts.get("validated", 0)
    validation_success_rate = round(validated_outputs / len(outputs), 4) if outputs else 0.0
    manual_review_requested = stats.human_review_counts.get("True", 0)
    manual_review_rate = round(manual_review_requested / len(outputs), 4) if outputs else 0.0
    return {
        "output_items": len(outputs),
        "trace_items": len(traces),
        "status_counts": dict(stats.status_counts),
        "task_counts": dict(stats.task_counts),
        "validation_error_counts": dict(stats.validation_errors),
        "is_valid_recommendation_counts": dict(stats.valid_counts),
        "is_valid_grade_counts": dict(stats.valid_grade_counts),
        "needs_human_review_counts": dict(stats.human_review_counts),
        "direction_counts": dict(stats.direction_counts),
        "strength_counts": dict(stats.strength_counts),
        "certainty_counts": dict(stats.certainty_counts),
        "grade_system_counts": dict(stats.grade_system_counts),
        "grade_domain_counts": {field: dict(counter) for field, counter in stats.grade_domain_counts.items()},
        "confidence_avg": _avg(stats.confidence_values),
        "auto_merge_ready": auto_merge_ready,
        "validation_success_rate": validation_success_rate,
        "manual_review_requested": manual_review_requested,
        "manual_review_rate": manual_review_rate,
        "rejected_by_llm": _rejected_by_llm(outputs),
        "trace_success_counts": dict(stats.trace_success_counts),
        "runtime_ms_avg": _avg(stats.runtime_values),
        "runtime_ms_max": max(stats.runtime_values) if stats.runtime_values else None,
        "total_tokens_sum": int(sum(stats.total_tokens)),
        "total_tokens_avg": _avg(stats.total_tokens),
        "prompt_tokens_avg": _avg(stats.prompt_tokens),
        "completion_tokens_avg": _avg(stats.completion_tokens),
    }


def build_samples(outputs: Iterable[JsonDict], limit: int = 20) -> List[JsonDict]:
    samples: List[JsonDict] = []
    for row in outputs:
        parsed = _parsed(row)
        is_grade = row.get("task_type") == "grade_candidate_review"
        validity = parsed.get("is_valid_grade") if is_grade else parsed.get("is_valid_recommendation")
        should_sample = row.get("status") != "validated" or parsed.get("needs_human_review") is True or validity is False
        if not should_sample:
            continue
        sample = {
            "llm_output_id": row.get("llm_output_id"),
            "queue_id": row.get("queue_id"),
            "task_type": row.get("task_type"),
            "candidate_id": parsed.get("grade_candidate_id") if is_grade else row.get("recommendation_candidate_id"),
            "status": row.get("status"),
            "validation_errors": row.get("validation_errors", []),
            "strength": parsed.get("strength"),
            "certainty": parsed.get("certainty"),
            "needs_human_review": parsed.get("needs_human_review"),
            "confidence": parsed.get("confidence"),
            "reject_reason": parsed.get("reject_reason"),
        }
        if is_grade:
            sample.update(
                {
                    "is_valid_grade": parsed.get("is_valid_grade"),
                    "grade_system": parsed.get("grade_system"),
                    "recommendation_candidate_id": parsed.get("recommendation_candidate_id"),
                    "risk_of_bias": parsed.get("risk_of_bias"),
                    "inconsistency": parsed.get("inconsistency"),
                    "indirectness": parsed.get("indirectness"),
                    "imprecision": parsed.get("imprecision"),
                    "publication_bias": parsed.get("publication_bias"),
                }
            )
        else:
            sample.update(
                {
                    "is_valid_recommendation": parsed.get("is_valid_recommendation"),
                    "direction": parsed.get("direction"),
                    "population": parsed.get("population"),
                    "intervention": parsed.get("intervention"),
                }
            )
        samples.append(sample)
        if len(samples) >= limit:
            break
    return samples


def build_report_file(
    outputs_input: str | Path,
    traces_input: str | Path,
    summary_output: str | Path,
    samples_output: str | Path,
    sample_limit: int = 20,
) -> JsonDict:
    """构建 LLM 复核输出和 trace 的质量摘要/样本报告。"""

    outputs = list(iter_jsonl(outputs_input))
    traces = list(iter_jsonl(traces_input))
    summary = summarize_outputs(outputs, traces)
    summary["outputs_input"] = str(outputs_input)
    summary["traces_input"] = str(traces_input)
    samples = build_samples(outputs, limit=sample_limit)
    report_common.write_report(summary_output, summary)
    write_jsonl(samples_output, samples)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a quality report for LLM review outputs.")
    parser.add_argument("--outputs-input", required=True)
    parser.add_argument("--traces-input", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--samples-output", required=True)
    parser.add_argument("--sample-limit", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_report_file(
        outputs_input=args.outputs_input,
        traces_input=args.traces_input,
        summary_output=args.summary_output,
        samples_output=args.samples_output,
        sample_limit=args.sample_limit,
    )
    print("output_items={output_items} status={status_counts} auto_merge_ready={auto_merge_ready}".format(**summary))


if __name__ == "__main__":
    main()
