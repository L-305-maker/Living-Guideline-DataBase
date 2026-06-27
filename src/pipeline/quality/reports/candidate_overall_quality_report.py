"""质量报告文件：读取流水线产物并生成摘要、计数和样本，帮助定位抽取或路由质量问题。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.pipeline.quality.common import report_common
from src.common.process_jsonl import iter_jsonl, write_jsonl


JsonDict = Dict[str, Any]


def _rec_block_id(row: JsonDict) -> str:
    payload = row.get("normalized_payload")
    if isinstance(payload, dict):
        return str(payload.get("block_id") or "")
    return ""


def _quality_notes(row: JsonDict) -> List[str]:
    payload = row.get("normalized_payload")
    notes = payload.get("quality_notes") if isinstance(payload, dict) else []
    return [str(note) for note in notes] if isinstance(notes, list) else []


def _association_reason(row: JsonDict) -> str:
    payload = row.get("normalized_payload")
    if isinstance(payload, dict):
        return str(payload.get("association_reason") or "")
    return ""


def _association_quality(row: JsonDict) -> str:
    payload = row.get("normalized_payload")
    if isinstance(payload, dict):
        return str(payload.get("association_quality") or "")
    return ""


def summarize_candidates(recommendations: List[JsonDict], grades: List[JsonDict], traces: List[JsonDict]) -> JsonDict:
    """汇总推荐/GRADE 候选的数量、状态、关联缺口和复核压力。"""

    grade_by_rec: Dict[str, List[JsonDict]] = defaultdict(list)
    for grade in grades:
        rec_id = str(grade.get("recommendation_candidate_id") or "")
        if rec_id:
            grade_by_rec[rec_id].append(grade)

    rec_status = report_common.counts_by(recommendations, "status")
    rec_strength = report_common.counts_by(recommendations, "strength")
    rec_certainty = report_common.counts_by(recommendations, "certainty")
    grade_status = report_common.counts_by(grades, "status")
    grade_strength = report_common.counts_by(grades, "strength")
    grade_certainty = report_common.counts_by(grades, "certainty")
    grade_system = report_common.counts_by(grades, "grade_system")
    association = Counter(_association_reason(row) or "unknown" for row in grades)
    association_quality = Counter(_association_quality(row) or "unknown" for row in grades)
    trace_task = Counter(str(row.get("task_type") or "unknown") for row in traces)
    trace_success = Counter("success" if row.get("success") else "failed" for row in traces)

    rec_with_grade = sum(1 for row in recommendations if str(row.get("candidate_id") or "") in grade_by_rec)
    rec_without_grade = len(recommendations) - rec_with_grade
    grade_without_rec = sum(1 for row in grades if not row.get("recommendation_candidate_id"))
    rec_needing_llm = sum(1 for row in recommendations if row.get("status") == "needs_review" or row.get("certainty") == "unclear" or row.get("strength") == "unclear")
    grade_needing_llm = sum(1 for row in grades if row.get("status") == "needs_review" or row.get("certainty") == "unclear" or row.get("strength") == "unclear")
    recommendation_review_backlog = sum(1 for row in recommendations if row.get("status") == "needs_review")
    grade_review_backlog = sum(1 for row in grades if row.get("status") == "needs_review")
    accepted_recommendations = rec_status.get("accepted", 0)
    candidate_readiness_rate = round(accepted_recommendations / len(recommendations), 4) if recommendations else 0.0
    grade_link_coverage = round(rec_with_grade / len(recommendations), 4) if recommendations else 0.0

    return {
        "recommendation_candidates": len(recommendations),
        "grade_candidates": len(grades),
        "model_traces": len(traces),
        "recommendations_with_grade": rec_with_grade,
        "recommendations_without_grade": rec_without_grade,
        "grade_without_recommendation": grade_without_rec,
        "recommendation_status_counts": rec_status,
        "recommendation_strength_counts": rec_strength,
        "recommendation_certainty_counts": rec_certainty,
        "grade_status_counts": grade_status,
        "grade_strength_counts": grade_strength,
        "grade_certainty_counts": grade_certainty,
        "grade_system_counts": grade_system,
        "grade_association_counts": dict(association),
        "grade_association_quality_counts": dict(association_quality),
        "trace_task_counts": dict(trace_task),
        "trace_success_counts": dict(trace_success),
        "recommendations_needing_llm": rec_needing_llm,
        "grades_needing_llm": grade_needing_llm,
        "recommendation_review_backlog": recommendation_review_backlog,
        "grade_review_backlog": grade_review_backlog,
        "candidate_readiness_rate": candidate_readiness_rate,
        "grade_link_coverage": grade_link_coverage,
    }


def _grades_by_recommendation(grades: Iterable[JsonDict]) -> Dict[str, List[JsonDict]]:
    grade_by_rec: Dict[str, List[JsonDict]] = defaultdict(list)
    for grade in grades:
        rec_id = str(grade.get("recommendation_candidate_id") or "")
        if rec_id:
            grade_by_rec[rec_id].append(grade)
    return grade_by_rec


def _recommendation_labels(rec: JsonDict, linked_grades: List[JsonDict]) -> List[str]:
    labels: List[str] = []
    if not linked_grades:
        labels.append("recommendation_without_grade")
    if rec.get("status") == "needs_review":
        labels.append("recommendation_needs_review")
    if rec.get("strength") == "unclear":
        labels.append("recommendation_strength_unclear")
    if rec.get("certainty") == "unclear":
        labels.append("recommendation_certainty_unclear")
    labels.extend(f"recommendation_note:{note}" for note in _quality_notes(rec))
    if linked_grades and any(grade.get("certainty") != "unclear" for grade in linked_grades):
        labels.append("recommendation_with_informative_grade")
    return labels


def _recommendation_sample(rec: JsonDict, label: str, linked_grades: List[JsonDict], max_text_chars: int) -> JsonDict:
    return {
        "sample_group": label,
        "candidate_id": str(rec.get("candidate_id") or ""),
        "block_id": _rec_block_id(rec),
        "record_id": rec.get("record_id", ""),
        "guideline_id": rec.get("guideline_id", ""),
        "status": rec.get("status", ""),
        "direction": rec.get("direction", ""),
        "strength": rec.get("strength", ""),
        "certainty": rec.get("certainty", ""),
        "linked_grade_count": len(linked_grades),
        "linked_grade_certainties": [grade.get("certainty") for grade in linked_grades],
        "linked_grade_strengths": [grade.get("strength") for grade in linked_grades],
        "quality_notes": _quality_notes(rec),
        "source_section": rec.get("source_section", ""),
        "recommendation_text": report_common.truncate(rec.get("recommendation_text"), max_text_chars),
    }


def _grade_labels(grade: JsonDict) -> List[str]:
    labels = []
    if not grade.get("recommendation_candidate_id"):
        labels.append("grade_without_recommendation")
    if grade.get("status") == "needs_review":
        labels.append("grade_needs_review")
    if grade.get("certainty") == "unclear":
        labels.append("grade_certainty_unclear")
    if grade.get("strength") == "unclear":
        labels.append("grade_strength_unclear")
    if grade.get("certainty") != "unclear" or grade.get("strength") != "unclear":
        labels.append("informative_grade")
    if _association_quality(grade) == "weak":
        labels.append("weak_grade_association")
    if _association_quality(grade) == "missing":
        labels.append("missing_grade_association")
    return labels


def _grade_sample(grade: JsonDict, label: str, max_text_chars: int) -> JsonDict:
    return {
        "sample_group": label,
        "grade_candidate_id": grade.get("grade_candidate_id", ""),
        "recommendation_candidate_id": grade.get("recommendation_candidate_id", ""),
        "record_id": grade.get("record_id", ""),
        "guideline_id": grade.get("guideline_id", ""),
        "status": grade.get("status", ""),
        "grade_system": grade.get("grade_system", ""),
        "certainty": grade.get("certainty", ""),
        "strength": grade.get("strength", ""),
        "association_reason": _association_reason(grade),
        "association_quality": _association_quality(grade),
        "source_section": grade.get("source_section", ""),
        "source_text": report_common.truncate(grade.get("source_text"), max_text_chars),
    }


def sample_overall_quality(recommendations: List[JsonDict], grades: List[JsonDict], per_group: int = 5, max_text_chars: int = 600) -> List[JsonDict]:
    grade_by_rec = _grades_by_recommendation(grades)
    groups: Dict[str, List[JsonDict]] = report_common.limited_groups()
    for rec in recommendations:
        rec_id = str(rec.get("candidate_id") or "")
        linked_grades = grade_by_rec.get(rec_id, [])
        for label in _recommendation_labels(rec, linked_grades):
            report_common.append_limited(groups, label, _recommendation_sample(rec, label, linked_grades, max_text_chars), per_group)
    for grade in grades:
        for label in _grade_labels(grade):
            report_common.append_limited(groups, label, _grade_sample(grade, label, max_text_chars), per_group)
    return report_common.flatten_groups(groups)


def build_report(
    recommendations_input: str | Path,
    grades_input: str | Path,
    traces_inputs: Iterable[str | Path],
    sample_size: int = 5,
    max_text_chars: int = 600,
) -> JsonDict:
    """构建 recommendation/GRADE 综合质量摘要和抽样行。"""

    recommendations = list(iter_jsonl(recommendations_input))
    grades = list(iter_jsonl(grades_input))
    traces: List[JsonDict] = []
    for path in traces_inputs:
        traces.extend(iter_jsonl(path))
    return {
        "recommendations_input": str(recommendations_input),
        "grades_input": str(grades_input),
        "traces_inputs": [str(path) for path in traces_inputs],
        "summary": summarize_candidates(recommendations, grades, traces),
        "samples": sample_overall_quality(recommendations, grades, per_group=sample_size, max_text_chars=max_text_chars),
    }


def write_report(report: JsonDict, summary_output: str | Path, samples_output: str | Path) -> None:
    write_jsonl(summary_output, [report["summary"]])
    write_jsonl(samples_output, report["samples"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an overall quality report for recommendation and grade candidates.")
    parser.add_argument("--recommendations-input", required=True)
    parser.add_argument("--grades-input", required=True)
    parser.add_argument("--traces-input", action="append", default=[], help="ModelTrace JSONL path. Can be repeated.")
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--samples-output", required=True)
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--max-text-chars", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        args.recommendations_input,
        args.grades_input,
        args.traces_input,
        sample_size=args.sample_size,
        max_text_chars=args.max_text_chars,
    )
    write_report(report, args.summary_output, args.samples_output)
    summary = report["summary"]
    print(
        "recommendations={recommendation_candidates} grades={grade_candidates} "
        "rec_without_grade={recommendations_without_grade} grade_without_rec={grade_without_recommendation}".format(
            **summary
        )
    )


if __name__ == "__main__":
    main()

