"""人工复核队列与复核结果回流模块。

本模块负责两件事: 一是从候选、PICO、证据和版本行中筛出需要人工看的记录；
二是把已完成的人工复核 JSONL 回写到原始实体，同时保留审计历史。
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.common.extraction_common import utc_now
from src.common.process_jsonl import iter_jsonl, write_jsonl
from src.domain.common import stable_id


JsonDict = Dict[str, Any]

VERSION = "manual_review_gate_v1"

ID_FIELDS = {
    "recommendation_candidate": "candidate_id",
    "grade_candidate": "grade_candidate_id",
    "pico_question": "pico_id",
    "evidence_item": "evidence_id",
    "recommendation_version": "recommendation_version_id",
}

CANDIDATE_STATUS_DECISIONS = {"accepted", "rejected", "needs_review", "pending"}
EVIDENCE_STATUS_DECISIONS = {"included", "excluded", "uncertain", "pending"}
READY_REVIEW_STATUSES = {"reviewed", "closed", "approved", "applied"}


@dataclass
class ReviewQueueOptions:
    """构建人工复核队列时使用的过滤选项。"""

    include_pending: bool = False
    include_unclear_fields: bool = False
    confidence_threshold: Optional[float] = None
    id_field: Optional[str] = None


@dataclass
class ReviewValidationState:
    """校验已复核行时累计的索引、去重集合和错误信息。"""

    # seen 集合只记录已通过校验的行，避免一个坏行阻断后面有效的修正行。
    indexed: Dict[str, JsonDict]
    seen_review_ids: set[str]
    seen_entities: set[str]
    errors: List[JsonDict]
    warnings: List[JsonDict]
    total: int = 0
    skipped_not_ready: int = 0


def payload(row: JsonDict, field: str = "normalized_payload") -> JsonDict:
    value = row.get(field)
    return value if isinstance(value, dict) else {}


def compact_text(value: Any, limit: int = 500) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def source_order(row: JsonDict) -> int:
    value = row.get("source_order")
    if value is None:
        value = payload(row).get("source_order")
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def get_nested(row: JsonDict, *keys: str) -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def entity_id(row: JsonDict, entity_type: str, id_field: Optional[str] = None) -> str:
    field = id_field or ID_FIELDS.get(entity_type)
    return str(row.get(field or "") or "")


def add_reason(reasons: List[str], reason: str) -> None:
    if reason and reason not in reasons:
        reasons.append(reason)


def base_reasons(row: JsonDict, include_pending: bool, confidence_threshold: Optional[float]) -> List[str]:
    reasons: List[str] = []
    status = str(row.get("status") or "")
    screening_status = str(row.get("screening_status") or "")
    if status == "needs_review":
        add_reason(reasons, "status_needs_review")
    if status == "under_review":
        add_reason(reasons, "status_under_review")
    if include_pending and status == "pending":
        add_reason(reasons, "status_pending")
    if screening_status == "uncertain":
        add_reason(reasons, "screening_uncertain")
    if include_pending and screening_status == "pending":
        add_reason(reasons, "screening_pending")

    if confidence_threshold is not None:
        confidence = row.get("extraction_confidence")
        if isinstance(confidence, (int, float)) and confidence < confidence_threshold:
            add_reason(reasons, "low_extraction_confidence")
    return reasons


def candidate_reasons(row: JsonDict, entity_type: str, include_pending: bool, include_unclear_fields: bool) -> List[str]:
    reasons: List[str] = []
    if row.get("review_note"):
        add_reason(reasons, "has_review_note")
    enhancement = payload(row).get("enhancement")
    if isinstance(enhancement, dict):
        decision = str(enhancement.get("auto_qc_decision") or enhancement.get("enhancement_status") or "")
        if decision in {"needs_review", "manual_review"}:
            add_reason(reasons, "auto_qc_needs_review")
        if include_pending and decision == "not_processed":
            add_reason(reasons, "not_processed_by_enhancer")
    if include_unclear_fields:
        if entity_type == "recommendation_candidate":
            for field in ["direction", "strength", "certainty"]:
                if str(row.get(field) or "") in {"", "unclear"}:
                    add_reason(reasons, f"unclear_{field}")
            if not row.get("population"):
                add_reason(reasons, "missing_population")
            if not row.get("intervention"):
                add_reason(reasons, "missing_intervention")
        if entity_type == "grade_candidate":
            for field in ["grade_system", "strength", "certainty"]:
                if str(row.get(field) or "") in {"", "unknown", "unclear"}:
                    add_reason(reasons, f"unclear_{field}")
            if not row.get("recommendation_candidate_id"):
                add_reason(reasons, "missing_recommendation_candidate_id")
            association_quality = str(payload(row).get("association_quality") or "")
            if association_quality == "weak":
                add_reason(reasons, "weak_grade_association")
    return reasons


def pico_reasons(row: JsonDict, include_unclear_fields: bool) -> List[str]:
    reasons: List[str] = []
    if include_unclear_fields:
        for field in ["clinical_question", "population", "intervention"]:
            if not row.get(field):
                add_reason(reasons, f"missing_{field}")
    return reasons


def evidence_reasons(row: JsonDict, include_unclear_fields: bool) -> List[str]:
    reasons: List[str] = []
    if not row.get("pico_id"):
        add_reason(reasons, "missing_pico_id")
    if include_unclear_fields:
        if str(row.get("study_design") or "") in {"", "unclear"}:
            add_reason(reasons, "unclear_study_design")
        if str(row.get("effect_direction") or "") in {"", "uncertain"}:
            add_reason(reasons, "uncertain_effect_direction")
    return reasons


def version_reasons(row: JsonDict) -> List[str]:
    reasons: List[str] = []
    normalized = payload(row)
    notes = normalized.get("builder_notes")
    if isinstance(notes, list):
        for note in notes:
            add_reason(reasons, str(note))
    if not row.get("pico_id"):
        add_reason(reasons, "no_linked_pico")
    linked_pico = normalized.get("linked_pico")
    if isinstance(linked_pico, dict) and linked_pico.get("match_quality") == "weak":
        add_reason(reasons, "weak_pico_match")
    linked_evidence = normalized.get("linked_evidence")
    if isinstance(linked_evidence, dict) and int(linked_evidence.get("linked_evidence_count") or 0) == 0:
        add_reason(reasons, "no_linked_evidence")
    conflicts = normalized.get("field_conflicts")
    if isinstance(conflicts, list) and conflicts:
        add_reason(reasons, "recommendation_grade_strength_or_certainty_conflict")
    return reasons


# 根据实体类型选择对应规则，生成该条记录进入人工复核队列的原因。
def review_reasons(
    row: JsonDict,
    entity_type: str,
    include_pending: bool,
    include_unclear_fields: bool,
    confidence_threshold: Optional[float],
) -> List[str]:
    """返回实体进入人工复核队列的去重原因。"""

    reasons = base_reasons(row, include_pending, confidence_threshold)
    if entity_type in {"recommendation_candidate", "grade_candidate"}:
        reasons.extend(candidate_reasons(row, entity_type, include_pending, include_unclear_fields))
    elif entity_type == "pico_question":
        reasons.extend(pico_reasons(row, include_unclear_fields))
    elif entity_type == "evidence_item":
        reasons.extend(evidence_reasons(row, include_unclear_fields))
    elif entity_type == "recommendation_version":
        reasons.extend(version_reasons(row))
    return list(dict.fromkeys(reasons))


# 已被人工接受/拒绝的原因不再重复入队，保证复核闭环不会反复打扰 reviewer。
def filter_resolved_reasons(row: JsonDict, reasons: List[str]) -> List[str]:
    if str(row.get("status") or "") == "rejected":
        return []
    if str(row.get("quality_status") or "") == "manual_review_rejected":
        return []
    last_review = payload(row).get("last_manual_review")
    if not isinstance(last_review, dict):
        return reasons
    decision = str(last_review.get("review_decision") or "")
    if decision not in {"accepted", "included"}:
        return reasons
    resolved = {str(reason) for reason in last_review.get("review_reasons") or []}
    if not resolved:
        return reasons
    return [reason for reason in reasons if reason not in resolved]


# 将复核原因映射为人工队列优先级，P0 只留给缺失关键语义或质量硬伤。
def priority_for(reasons: List[str]) -> str:
    p0 = {
        "status_needs_review",
        "auto_qc_needs_review",
        "no_linked_pico",
        "missing_pico_id",
        "recommendation_grade_strength_or_certainty_conflict",
    }
    p1 = {
        "weak_pico_match",
        "screening_uncertain",
        "low_extraction_confidence",
        "linked_grade_candidates_not_accepted",
        "has_review_note",
        "status_under_review",
        "weak_grade_association",
    }
    if any(reason in p0 for reason in reasons):
        return "P0"
    if any(reason in p1 or reason.startswith(("missing_", "unclear_", "uncertain_")) for reason in reasons):
        return "P1"
    return "P2"


def common_state(row: JsonDict) -> JsonDict:
    return {
        "record_id": row.get("record_id") or row.get("source_record_id"),
        "guideline_id": row.get("guideline_id"),
        "paper_id": row.get("paper_id"),
        "source_block_id": row.get("source_block_id") or payload(row).get("block_id"),
        "source_order": source_order(row),
        "source_section": row.get("source_section"),
        "source_url": row.get("source_url"),
        "status": row.get("status"),
        "screening_status": row.get("screening_status"),
        "quality_status": row.get("quality_status"),
        "extraction_confidence": row.get("extraction_confidence"),
        "review_note": row.get("review_note"),
    }


def recommendation_state(row: JsonDict) -> JsonDict:
    return {
        "recommendation_text": row.get("recommendation_text"),
        "direction": row.get("direction"),
        "strength": row.get("strength"),
        "certainty": row.get("certainty"),
        "population": row.get("population"),
        "intervention": row.get("intervention"),
    }


def grade_state(row: JsonDict) -> JsonDict:
    return {
        "recommendation_candidate_id": row.get("recommendation_candidate_id"),
        "grade_system": row.get("grade_system"),
        "strength": row.get("strength"),
        "certainty": row.get("certainty"),
    }


def pico_state(row: JsonDict) -> JsonDict:
    return {
        "clinical_question": row.get("clinical_question"),
        "population": row.get("population"),
        "intervention": row.get("intervention"),
        "comparator": row.get("comparator"),
        "outcomes": row.get("outcomes"),
    }


def evidence_state(row: JsonDict) -> JsonDict:
    return {
        "source_record_id": row.get("source_record_id"),
        "paper_id": row.get("paper_id"),
        "pico_id": row.get("pico_id"),
        "recommendation_candidate_id": row.get("recommendation_candidate_id"),
        "study_design": row.get("study_design"),
        "effect_direction": row.get("effect_direction"),
        "source_text": compact_text(row.get("source_text")),
    }


def version_state(row: JsonDict) -> JsonDict:
    return {
        "recommendation_version_id": row.get("recommendation_version_id"),
        "recommendation_candidate_id": row.get("recommendation_candidate_id"),
        "grade_candidate_id": row.get("grade_candidate_id"),
        "pico_id": row.get("pico_id"),
        "recommendation_text": row.get("recommendation_text"),
        "strength": row.get("strength"),
        "certainty": row.get("certainty"),
        "linked_pico": get_nested(row, "normalized_payload", "linked_pico"),
        "linked_evidence": get_nested(row, "normalized_payload", "linked_evidence"),
        "field_conflicts": get_nested(row, "normalized_payload", "field_conflicts"),
        "builder_notes": get_nested(row, "normalized_payload", "builder_notes"),
    }


ENTITY_STATE_BUILDERS = {
    "recommendation_candidate": recommendation_state,
    "grade_candidate": grade_state,
    "pico_question": pico_state,
    "evidence_item": evidence_state,
    "recommendation_version": version_state,
}


def compact_current_state(row: JsonDict, entity_type: str) -> JsonDict:
    common = common_state(row)
    builder = ENTITY_STATE_BUILDERS.get(entity_type)
    if builder:
        common.update(builder(row))
    return {key: value for key, value in common.items() if value not in (None, "", [])}


# 构造人工复核 item，保留当前候选状态和复核原因，供人工界面或 JSONL 编辑。
def build_review_item(row: JsonDict, entity_type: str, reasons: List[str], id_field: Optional[str] = None) -> JsonDict:
    item_entity_id = entity_id(row, entity_type, id_field)
    review_id = stable_id("manual_review", VERSION, entity_type, item_entity_id, ",".join(reasons))
    now = utc_now()
    return {
        "review_id": review_id,
        "review_gate_version": VERSION,
        "entity_type": entity_type,
        "entity_id": item_entity_id,
        "priority": priority_for(reasons),
        "review_status": "open",
        "review_reasons": reasons,
        "current_state": compact_current_state(row, entity_type),
        "review_decision": None,
        "reviewed_payload": {},
        "reviewer": None,
        "reviewed_at": None,
        "review_note": None,
        "created_at": now,
        "updated_at": now,
    }


# 批量构建人工复核队列，并过滤已处理原因。
def build_queue(
    rows: Iterable[JsonDict],
    entity_type: str,
    options: Optional[ReviewQueueOptions] = None,
) -> tuple[List[JsonDict], JsonDict]:
    options = options or ReviewQueueOptions()
    queue: List[JsonDict] = []
    total = 0
    skipped_without_id = 0
    reason_counts: Counter[str] = Counter()
    priority_counts: Counter[str] = Counter()
    for row in rows:
        total += 1
        if not entity_id(row, entity_type, options.id_field):
            skipped_without_id += 1
            continue
        reasons = review_reasons(
            row,
            entity_type,
            options.include_pending,
            options.include_unclear_fields,
            options.confidence_threshold,
        )
        reasons = filter_resolved_reasons(row, reasons)
        if not reasons:
            continue
        item = build_review_item(row, entity_type, reasons, options.id_field)
        queue.append(item)
        priority_counts[item["priority"]] += 1
        reason_counts.update(reasons)
    summary = {
        "review_gate_version": VERSION,
        "entity_type": entity_type,
        "input_records": total,
        "queued_records": len(queue),
        "skipped_without_id": skipped_without_id,
        "priority_counts": dict(priority_counts),
        "reason_counts": dict(reason_counts),
        "include_pending": options.include_pending,
        "include_unclear_fields": options.include_unclear_fields,
        "confidence_threshold": options.confidence_threshold,
    }
    return queue, summary


# 递归合并人工修订内容，避免覆盖未被 reviewer 修改的嵌套字段。
def deep_merge(left: JsonDict, right: JsonDict) -> JsonDict:
    merged = deepcopy(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def review_ready(item: JsonDict) -> bool:
    decision = item.get("review_decision")
    if not decision:
        return False
    return str(item.get("review_status") or "").lower() in READY_REVIEW_STATUSES


def allowed_decisions(entity_type: str) -> set[str]:
    if entity_type in {"recommendation_candidate", "grade_candidate", "pico_question"}:
        return CANDIDATE_STATUS_DECISIONS
    if entity_type == "evidence_item":
        return EVIDENCE_STATUS_DECISIONS
    if entity_type == "recommendation_version":
        return {"accepted", "rejected", "needs_review"}
    return set()


def _new_validation_state() -> ReviewValidationState:
    return ReviewValidationState(indexed={}, seen_review_ids=set(), seen_entities=set(), errors=[], warnings=[])


def _review_row_issues(row: JsonDict, allowed: set[str], state: ReviewValidationState) -> tuple[List[str], List[str]]:
    entity = str(row.get("entity_id") or "")
    review_id = str(row.get("review_id") or "")
    decision = str(row.get("review_decision") or "")
    status = str(row.get("review_status") or "").lower()
    row_errors: List[str] = []
    row_warnings: List[str] = []
    if not entity:
        row_errors.append("missing_entity_id")
    if not review_id:
        row_errors.append("missing_review_id")
    elif review_id in state.seen_review_ids:
        row_errors.append("duplicate_review_id")
    if entity and entity in state.seen_entities:
        row_errors.append("duplicate_entity_id")
    if decision not in allowed:
        row_errors.append("invalid_review_decision")
    if status not in READY_REVIEW_STATUSES:
        row_errors.append("invalid_review_status")
    reviewed_payload = row.get("reviewed_payload")
    # reviewed_payload 可以为空；一旦提供就必须是对象，后续 deep_merge 依赖这个约束。
    if reviewed_payload is not None and not isinstance(reviewed_payload, dict):
        row_errors.append("reviewed_payload_must_be_object")
    if not row.get("reviewer"):
        row_warnings.append("missing_reviewer")
    if not row.get("reviewed_at"):
        row_warnings.append("missing_reviewed_at")
    return row_errors, row_warnings


def _accept_review_row(state: ReviewValidationState, row: JsonDict, index: int, row_errors: List[str], row_warnings: List[str]) -> None:
    """记录已通过校验的复核行，或收集该行的校验错误。"""

    entity = str(row.get("entity_id") or "")
    review_id = str(row.get("review_id") or "")
    if row_errors:
        state.errors.append({"row": index, "entity_id": entity, "review_id": review_id, "errors": row_errors})
        return
    if row_warnings:
        state.warnings.append({"row": index, "entity_id": entity, "review_id": review_id, "warnings": row_warnings})
    state.indexed[entity] = row
    state.seen_entities.add(entity)
    state.seen_review_ids.add(review_id)


def _validation_summary(state: ReviewValidationState) -> JsonDict:
    summary = {
        "review_rows": state.total,
        "valid_reviewed_rows": len(state.indexed),
        "skipped_not_ready": state.skipped_not_ready,
        "validation_error_count": len(state.errors),
        "validation_warning_count": len(state.warnings),
        "validation_errors": state.errors[:10],
        "validation_warnings": state.warnings[:10],
    }
    return summary


def validate_review_items(rows: Iterable[JsonDict], entity_type: str) -> tuple[Dict[str, JsonDict], JsonDict]:
    state = _new_validation_state()
    allowed = allowed_decisions(entity_type)
    for index, row in enumerate(rows, 1):
        state.total += 1
        if str(row.get("entity_type") or "") != entity_type:
            continue
        if not review_ready(row):
            # 队列里允许存在草稿复核记录；这里只应用已经完成复核的行。
            state.skipped_not_ready += 1
            continue
        row_errors, row_warnings = _review_row_issues(row, allowed, state)
        _accept_review_row(state, row, index, row_errors, row_warnings)
    return state.indexed, _validation_summary(state)


def index_review_items(rows: Iterable[JsonDict], entity_type: str) -> Dict[str, JsonDict]:
    indexed: Dict[str, JsonDict] = {}
    for row in rows:
        if str(row.get("entity_type") or "") != entity_type:
            continue
        if not review_ready(row):
            continue
        entity = str(row.get("entity_id") or "")
        if entity:
            indexed[entity] = row
    return indexed


def apply_decision(row: JsonDict, entity_type: str, decision: str) -> None:
    if entity_type in {"recommendation_candidate", "grade_candidate", "pico_question"}:
        if decision in CANDIDATE_STATUS_DECISIONS:
            row["status"] = decision
    elif entity_type == "evidence_item":
        if decision in EVIDENCE_STATUS_DECISIONS:
            row["screening_status"] = decision
    elif entity_type == "recommendation_version":
        if decision == "accepted":
            row["quality_status"] = "manual_review_accepted"
        elif decision == "rejected":
            row["quality_status"] = "manual_review_rejected"
        elif decision == "needs_review":
            row["quality_status"] = "manual_review_needs_review"
        else:
            row["quality_status"] = f"manual_review_{decision}"


def attach_manual_review(row: JsonDict, review_item: JsonDict) -> None:
    now = utc_now()
    normalized = payload(row)
    history = normalized.get("manual_review_history")
    if not isinstance(history, list):
        history = []
    record = {
        "review_id": review_item.get("review_id"),
        "review_gate_version": review_item.get("review_gate_version") or VERSION,
        "review_decision": review_item.get("review_decision"),
        "review_reasons": review_item.get("review_reasons", []),
        "reviewer": review_item.get("reviewer"),
        "reviewed_at": review_item.get("reviewed_at"),
        "review_note": review_item.get("review_note"),
        "applied_at": now,
    }
    history.append(record)
    normalized["manual_review_history"] = history
    normalized["last_manual_review"] = record
    row["normalized_payload"] = normalized
    row["updated_at"] = now
    if review_item.get("review_note"):
        row["review_note"] = review_item.get("review_note")


def apply_review_to_row(row: JsonDict, review_item: JsonDict, entity_type: str) -> JsonDict:
    """把一条已完成复核应用到单个实体副本，并返回回写后的实体。"""

    merged = deepcopy(row)
    decision = str(review_item.get("review_decision") or "")
    apply_decision(merged, entity_type, decision)
    reviewed_payload = review_item.get("reviewed_payload")
    if isinstance(reviewed_payload, dict) and reviewed_payload:
        merged = deep_merge(merged, reviewed_payload)
    attach_manual_review(merged, review_item)
    return merged


# 将人工复核结果回写到候选/版本实体，并附加 review audit 信息。
def apply_reviews(
    base_rows: Iterable[JsonDict],
    reviewed_rows: Iterable[JsonDict],
    entity_type: str,
    id_field: Optional[str] = None,
) -> tuple[List[JsonDict], JsonDict]:
    reviews_by_entity, validation = validate_review_items(reviewed_rows, entity_type)
    output: List[JsonDict] = []
    applied = 0
    decision_counts: Counter[str] = Counter()
    for row in base_rows:
        entity = entity_id(row, entity_type, id_field)
        review_item = reviews_by_entity.get(entity)
        if not review_item:
            output.append(row)
            continue
        decision = str(review_item.get("review_decision") or "")
        merged = apply_review_to_row(row, review_item, entity_type)
        output.append(merged)
        applied += 1
        decision_counts[decision] += 1
    summary = {
        "review_gate_version": VERSION,
        "entity_type": entity_type,
        "base_records": len(output),
        "reviewed_records": len(reviews_by_entity),
        "applied_reviews": applied,
        "unmatched_reviews": max(len(reviews_by_entity) - applied, 0),
        "decision_counts": dict(decision_counts),
        "review_validation": validation,
    }
    return output, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and apply generic manual review queues.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-queue", help="Extract records that need human review.")
    build.add_argument("--entity-type", required=True, choices=sorted(ID_FIELDS))
    build.add_argument("--input", required=True)
    build.add_argument("--queue-output", required=True)
    build.add_argument("--summary-output", required=True)
    build.add_argument("--id-field")
    build.add_argument("--include-pending", action="store_true")
    build.add_argument("--include-unclear-fields", action="store_true")
    build.add_argument("--confidence-threshold", type=float)

    apply = subparsers.add_parser("apply", help="Apply reviewed queue rows back to a base JSONL file.")
    apply.add_argument("--entity-type", required=True, choices=sorted(ID_FIELDS))
    apply.add_argument("--base-input", required=True)
    apply.add_argument("--reviewed-input", required=True)
    apply.add_argument("--output", required=True)
    apply.add_argument("--summary-output", required=True)
    apply.add_argument("--id-field")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build-queue":
        queue, summary = build_queue(
            iter_jsonl(args.input),
            entity_type=args.entity_type,
            options=ReviewQueueOptions(
                include_pending=args.include_pending,
                include_unclear_fields=args.include_unclear_fields,
                confidence_threshold=args.confidence_threshold,
                id_field=args.id_field,
            ),
        )
        summary["input"] = args.input
        summary["queue_output"] = args.queue_output
        write_jsonl(args.queue_output, queue)
        write_jsonl(args.summary_output, [summary])
        print("queued_records={queued_records} priority={priority_counts} reasons={reason_counts}".format(**summary))
        return

    output, summary = apply_reviews(
        iter_jsonl(args.base_input),
        iter_jsonl(args.reviewed_input),
        entity_type=args.entity_type,
        id_field=args.id_field,
    )
    summary["base_input"] = args.base_input
    summary["reviewed_input"] = args.reviewed_input
    summary["output"] = args.output
    write_jsonl(args.output, output)
    write_jsonl(args.summary_output, [summary])
    print("applied_reviews={applied_reviews} decisions={decision_counts}".format(**summary))


if __name__ == "__main__":
    main()
