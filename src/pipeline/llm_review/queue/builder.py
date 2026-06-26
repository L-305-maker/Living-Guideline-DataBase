"""根据 recommendation 和 GRADE 候选构建 LLM 复核队列。

队列只是优先级层，不是第二套抽取器。它把相关 recommendation/GRADE 状态
合并到同一个复核上下文中，记录为什么需要复核，并默认把干净的 P3 项排除在
常规复核工作量之外。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.domain.common import stable_id
from src.common.process_jsonl import iter_jsonl, write_jsonl


JsonDict = Dict[str, Any]

EXPECTED_RECOMMENDATION_SCHEMA = {
    "recommendation_text": "string",
    "direction": "for|against|neutral|no_recommendation|unclear",
    "strength": "strong|conditional|weak|good_practice|none|unclear",
    "certainty": "high|moderate|low|very_low|none|unclear",
    "population": "string|null",
    "intervention": "string|null",
    "comparator": "string|null",
    "outcomes": "list",
    "rationale": "string|null",
    "remarks": "string|null",
    "needs_human_review": "boolean",
}
EXPECTED_GRADE_SCHEMA = {
    "grade_system": "GRADE|COR_LOE|LETTER_GRADE|NUMERIC_LETTER_GRADE|VERB_BASED|unknown",
    "certainty": "high|moderate|low|very_low|none|unclear",
    "strength": "strong|conditional|weak|good_practice|none|unclear",
    "risk_of_bias": "no_concern|serious|very_serious|unclear|not_extracted",
    "inconsistency": "no_concern|serious|very_serious|unclear|not_extracted",
    "indirectness": "no_concern|serious|very_serious|unclear|not_extracted",
    "imprecision": "no_concern|serious|very_serious|unclear|not_extracted",
    "publication_bias": "undetected|suspected|strongly_suspected|unclear|not_extracted",
    "reasons_for_downgrade": "list[str]",
    "reasons_for_upgrade": "list[str]",
    "needs_human_review": "boolean",
}


# 从 candidate 取出下游复核需要的标准化信息。
def payload(row: JsonDict) -> JsonDict:
    """读取候选的 normalized_payload；不存在时返回空对象。"""

    payload = row.get("normalized_payload")
    return payload if isinstance(payload, dict) else {}


def quality_notes(row: JsonDict) -> List[str]:
    """读取抽取/QC 阶段写入的候选质量备注。"""

    notes = payload(row).get("quality_notes")
    return [str(note) for note in notes] if isinstance(notes, list) else []


def block_id(row: JsonDict) -> str:
    """读取候选 payload 中携带的来源 block ID。"""

    return str(payload(row).get("block_id") or "")


def grade_association(row: JsonDict) -> str:
    """读取 GRADE 候选与推荐候选的关联方式。"""

    return str(payload(row).get("association_reason") or "")
###########################################################


def grade_is_informative(grade: JsonDict) -> bool:
    """判断 GRADE 行是否能补充 certainty 或 strength 信息。"""

    return grade.get("certainty") != "unclear" or grade.get("strength") != "unclear"


# 复核原因检测 #############################################
# 汇总推荐候选需要 LLM/人工复核的原因，既看自身字段，也看是否可由 GRADE 补全。
def recommendation_review_reasons(rec: JsonDict, linked_grades: List[JsonDict]) -> List[str]:
    """说明推荐候选为什么需要进入 LLM/人工复核。"""

    reasons: List[str] = []
    if rec.get("status") == "needs_review":
        reasons.append("recommendation_needs_review")
    if rec.get("certainty") == "unclear":
        reasons.append("recommendation_certainty_unclear")
    if rec.get("strength") == "unclear":
        reasons.append("recommendation_strength_unclear")
    if quality_notes(rec):
        reasons.extend(f"quality_note:{note}" for note in quality_notes(rec))
    if linked_grades and any(grade_is_informative(grade) for grade in linked_grades):
        if rec.get("certainty") == "unclear" or rec.get("strength") == "unclear":
            reasons.append("linked_grade_can_improve_recommendation")
    if not linked_grades:
        reasons.append("recommendation_without_grade")
    return list(dict.fromkeys(reasons))


# 只有会明显破坏候选文本完整性的质量问题才升级到 P0，避免 P0 队列重新爆炸。
def has_p0_quality_note(reasons: List[str]) -> bool:
    """判断复核原因里是否包含文本完整性硬伤。"""

    p0_notes = {
        "quality_note:starts_with_fragment",
        "quality_note:starts_with_punctuation",
        "quality_note:too_short_fragment",
        "quality_note:incomplete_action_phrase",
        "quality_note:layout_glued_statement",
        "quality_note:table_or_figure_contaminated",
    }
    return any(reason in p0_notes for reason in reasons)


def grade_review_reasons(grade: JsonDict) -> List[str]:
    """说明 GRADE 候选为什么需要进入复核。"""

    reasons: List[str] = []
    if grade.get("status") == "needs_review":
        reasons.append("grade_needs_review")
    if grade.get("certainty") == "unclear":
        reasons.append("grade_certainty_unclear")
    if grade.get("strength") == "unclear":
        reasons.append("grade_strength_unclear")
    if not grade.get("recommendation_candidate_id"):
        reasons.append("grade_without_recommendation")
    if grade_association(grade).startswith("nearest_order"):
        reasons.append("weak_grade_recommendation_association")
    return reasons
###########################################################


# 复核重要优先级排序 ########################################
# 根据复核原因给 recommendation review item 分级；质量硬伤优先于普通缺字段。
def priority_for_recommendation(rec: JsonDict, linked_grades: List[JsonDict], reasons: List[str]) -> str:
    """给 recommendation 复核任务分配 P0-P3 优先级。"""

    if "recommendation_needs_review" in reasons and has_p0_quality_note(reasons):
        return "P0"
    if has_p0_quality_note(reasons):
        return "P0"
    if "linked_grade_can_improve_recommendation" in reasons:
        return "P0"
    if "recommendation_needs_review" in reasons:
        return "P1"
    if any(reason.startswith("quality_note:") for reason in reasons):
        return "P1"
    if rec.get("strength") == "unclear" or rec.get("certainty") == "unclear":
        return "P1"
    if not linked_grades:
        return "P2"
    return "P3"


def priority_for_grade(grade: JsonDict, reasons: List[str]) -> str:
    """给 GRADE 复核任务分配 P0-P3 优先级。"""

    if "grade_without_recommendation" in reasons:
        return "P2"
    if "grade_needs_review" in reasons:
        return "P1"
    if "grade_certainty_unclear" in reasons or "grade_strength_unclear" in reasons:
        return "P1"
    return "P3"
###########################################################


def build_recommendation_queue_item(rec: JsonDict, linked_grades: List[JsonDict]) -> JsonDict:
    """构造一条带关联 GRADE 上下文的推荐复核队列项。"""

    reasons = recommendation_review_reasons(rec, linked_grades)
    priority = priority_for_recommendation(rec, linked_grades, reasons)
    queue_id = stable_id("llm_queue", "recommendation", rec.get("candidate_id"), priority)
    return {
        "queue_id": queue_id,
        "priority": priority,
        "task_type": "recommendation_candidate_review",
        "recommendation_candidate_id": rec.get("candidate_id", ""),
        "grade_candidate_ids": [grade.get("grade_candidate_id", "") for grade in linked_grades],
        "model_trace_ids": [rec.get("model_trace_id", ""), *[grade.get("model_trace_id", "") for grade in linked_grades]],
        "record_id": rec.get("record_id", ""),
        "guideline_id": rec.get("guideline_id", ""),
        "block_id": block_id(rec),
        "source_section": rec.get("source_section", ""),
        "source_text": rec.get("source_text", ""),
        "current_candidate_state": {
            "recommendation": rec,
            "linked_grades": linked_grades,
        },
        "review_reasons": reasons,
        "expected_llm_output_schema": EXPECTED_RECOMMENDATION_SCHEMA,
    }


def build_grade_queue_item(grade: JsonDict) -> JsonDict:
    """构造一条独立的 GRADE 复核队列项。"""

    reasons = grade_review_reasons(grade)
    priority = priority_for_grade(grade, reasons)
    queue_id = stable_id("llm_queue", "grade", grade.get("grade_candidate_id"), priority)
    return {
        "queue_id": queue_id,
        "priority": priority,
        "task_type": "grade_candidate_review",
        "recommendation_candidate_id": grade.get("recommendation_candidate_id", ""),
        "grade_candidate_ids": [grade.get("grade_candidate_id", "")],
        "model_trace_ids": [grade.get("model_trace_id", "")],
        "record_id": grade.get("record_id", ""),
        "guideline_id": grade.get("guideline_id", ""),
        "block_id": block_id(grade),
        "source_section": grade.get("source_section", ""),
        "source_text": grade.get("source_text", ""),
        "current_candidate_state": {
            "grade": grade,
        },
        "review_reasons": reasons,
        "expected_llm_output_schema": EXPECTED_GRADE_SCHEMA,
    }


# 构建 LLM review queue：先按 recommendation 关联 grade，再生成 recommendation 和 grade 复核任务。
def build_queue(recommendations: Iterable[JsonDict], grades: Iterable[JsonDict], include_p3: bool = False) -> List[JsonDict]:
    """从 recommendation/GRADE 候选构建已排序的复核队列。"""

    recs = list(recommendations)
    grade_rows = list(grades)
    grades_by_rec: Dict[str, List[JsonDict]] = defaultdict(list)
    for grade in grade_rows:
        rec_id = str(grade.get("recommendation_candidate_id") or "")
        if rec_id:
            grades_by_rec[rec_id].append(grade)

    queue: List[JsonDict] = []
    for rec in recs:
        linked_grades = grades_by_rec.get(str(rec.get("candidate_id") or ""), [])
        item = build_recommendation_queue_item(rec, linked_grades)
        if include_p3 or item["priority"] != "P3":
            queue.append(item)

    for grade in grade_rows:
        item = build_grade_queue_item(grade)
        if include_p3 or item["priority"] != "P3":
            queue.append(item)

    priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    queue.sort(key=lambda item: (priority_order.get(str(item.get("priority")), 9), str(item.get("record_id")), str(item.get("queue_id"))))
    return queue


def summarize_queue(queue: List[JsonDict]) -> JsonDict:
    """统计队列规模、优先级、任务类型和复核原因。"""

    priority_counts: Dict[str, int] = defaultdict(int)
    task_counts: Dict[str, int] = defaultdict(int)
    reason_counts: Dict[str, int] = defaultdict(int)
    for item in queue:
        priority_counts[str(item.get("priority") or "unknown")] += 1
        task_counts[str(item.get("task_type") or "unknown")] += 1
        for reason in item.get("review_reasons") or []:
            reason_counts[str(reason)] += 1
    return {
        "queue_items": len(queue),
        "priority_counts": dict(priority_counts),
        "task_counts": dict(task_counts),
        "reason_counts": dict(reason_counts),
    }


# 读取候选文件，构建完整复核队列并输出摘要。
def build_queue_file(
    recommendations_input: str | Path,
    grades_input: str | Path,
    queue_output: str | Path,
    summary_output: str | Path,
    include_p3: bool = False,
) -> JsonDict:
    """读取候选 JSONL，写出 LLM 复核队列和摘要。"""

    queue = build_queue(iter_jsonl(recommendations_input), iter_jsonl(grades_input), include_p3=include_p3)
    summary = summarize_queue(queue)
    summary["recommendations_input"] = str(recommendations_input)
    summary["grades_input"] = str(grades_input)
    write_jsonl(queue_output, queue)
    write_jsonl(summary_output, [summary])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LLM review queue from recommendation and grade candidates.")
    parser.add_argument("--recommendations-input", required=True)
    parser.add_argument("--grades-input", required=True)
    parser.add_argument("--queue-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--include-p3", action="store_true", help="Include already low-priority clean items.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_queue_file(
        args.recommendations_input,
        args.grades_input,
        args.queue_output,
        args.summary_output,
        include_p3=args.include_p3,
    )
    print("queue_items={queue_items} priority={priority_counts} tasks={task_counts}".format(**summary))


if __name__ == "__main__":
    main()
