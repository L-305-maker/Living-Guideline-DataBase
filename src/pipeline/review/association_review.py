"""复核阶段文件：处理人工/规则辅助审核、关联修复和 backlog 回写，保护正式发布包质量。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.common.extraction_common import utc_now
from src.common.process_jsonl import iter_jsonl, write_jsonl
from src.domain.common import stable_id
from src.pipeline.review.rule_assisted_backfill import (
    BackfillInputs,
    NearbyIndex,
    can_accept_recommendation,
    candidate_grade_score,
    evidence_has_usable_signal,
    evidence_match_score,
    grouped_by,
    index_by,
    order_distance,
    payload,
    pico_match_score,
    section_overlap,
    source_order,
    token_overlap,
)


JsonDict = Dict[str, Any]

ASSOCIATION_REVIEW_VERSION = "association_review_v1"
PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2}
LINK_BLOCKERS = {"no_usable_grade", "no_usable_pico", "no_linked_evidence"}
RESOLVED_ASSOCIATION_DECISIONS = {"accepted", "rejected", "needs_review"}


@dataclass(frozen=True)
class AssociationReviewOptions:
    min_grade_score: float = 0.45
    min_pico_score: float = 0.25
    min_evidence_score: float = 0.35
    min_evidence_pico_score: float = 0.25
    high_evidence_pico_score: float = 0.55
    max_grade_suggestions: int = 3
    max_pico_suggestions: int = 3
    max_evidence_suggestions: int = 8
    batch_size: int = 100
    max_recommendation_records: Optional[int] = None
    max_evidence_pico_records: Optional[int] = None
    include_accepted_recommendations: bool = False


def compact_text(value: Any, limit: int = 420) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def record_key(row: JsonDict) -> str:
    return str(row.get("record_id") or row.get("source_record_id") or "")


def same_record_or_guideline(left: JsonDict, right: JsonDict) -> bool:
    left_record = record_key(left)
    right_record = record_key(right)
    if left_record and left_record == right_record:
        return True
    left_guideline = str(left.get("guideline_id") or "")
    right_guideline = str(right.get("guideline_id") or "")
    return bool(left_guideline and left_guideline == right_guideline)


def row_id(row: JsonDict, entity_type: str) -> str:
    fields = {
        "grade": "grade_candidate_id",
        "pico": "pico_id",
        "evidence": "evidence_id",
        "recommendation": "candidate_id",
    }
    return str(row.get(fields[entity_type]) or "")


def manual_association_processing(row: JsonDict) -> JsonDict:
    value = payload(row).get("manual_association_processing")
    return value if isinstance(value, dict) else {}


def association_review_resolved(row: JsonDict) -> bool:
    manual = manual_association_processing(row)
    return str(manual.get("review_decision") or "") in RESOLVED_ASSOCIATION_DECISIONS


def suggestion_state(row: JsonDict, entity_type: str, score: float, reasons: List[str]) -> JsonDict:
    common = {
        "entity_type": entity_type,
        "entity_id": row_id(row, entity_type),
        "association_score": round(score, 4),
        "association_reasons": reasons,
        "record_id": row.get("record_id") or row.get("source_record_id"),
        "guideline_id": row.get("guideline_id"),
        "source_order": source_order(row),
        "source_section": row.get("source_section"),
        "status": row.get("status"),
        "screening_status": row.get("screening_status"),
        "extraction_confidence": row.get("extraction_confidence"),
    }
    if entity_type == "grade":
        common.update(
            {
                "recommendation_candidate_id": row.get("recommendation_candidate_id"),
                "grade_system": row.get("grade_system"),
                "strength": row.get("strength"),
                "certainty": row.get("certainty"),
                "source_text": compact_text(row.get("source_text"), 300),
                "association_quality": payload(row).get("association_quality"),
            }
        )
    elif entity_type == "pico":
        common.update(
            {
                "clinical_question": compact_text(row.get("clinical_question"), 220),
                "population": row.get("population"),
                "intervention": row.get("intervention"),
                "comparator": row.get("comparator"),
                "outcomes": row.get("outcomes"),
            }
        )
    elif entity_type == "evidence":
        common.update(
            {
                "recommendation_candidate_id": row.get("recommendation_candidate_id"),
                "pico_id": row.get("pico_id"),
                "study_design": row.get("study_design"),
                "effect_direction": row.get("effect_direction"),
                "effect_size": row.get("effect_size"),
                "confidence_interval": row.get("confidence_interval"),
                "source_text": compact_text(row.get("source_text"), 360),
            }
        )
    return {key: value for key, value in common.items() if value not in (None, "", [])}


def dedupe_suggestions(items: Iterable[tuple[float, JsonDict, List[str]]], entity_type: str, limit: int) -> List[JsonDict]:
    seen: set[str] = set()
    selected: List[JsonDict] = []
    for score, row, reasons in sorted(items, key=lambda item: item[0], reverse=True):
        entity_id = row_id(row, entity_type)
        if not entity_id or entity_id in seen:
            continue
        seen.add(entity_id)
        selected.append(suggestion_state(row, entity_type, score, reasons))
        if len(selected) >= limit:
            break
    return selected


def grade_suggestions(
    rec: JsonDict,
    direct_grades: List[JsonDict],
    grade_index: NearbyIndex,
    options: AssociationReviewOptions,
) -> List[JsonDict]:
    rec_id = str(rec.get("candidate_id") or "")
    scored: List[tuple[float, JsonDict, List[str]]] = []
    candidates = list(direct_grades)
    candidates.extend(row for row in grade_index.candidates_for(rec) if row not in direct_grades)
    for grade in candidates:
        linked_rec = str(grade.get("recommendation_candidate_id") or "")
        if linked_rec and linked_rec != rec_id:
            continue
        score, reasons = candidate_grade_score(rec, grade)
        if score >= options.min_grade_score:
            scored.append((score, grade, reasons))
    return dedupe_suggestions(scored, "grade", options.max_grade_suggestions)


def pico_suggestions(
    rec: JsonDict,
    evidence_rows: List[JsonDict],
    picos_by_id: Dict[str, JsonDict],
    pico_index: NearbyIndex,
    options: AssociationReviewOptions,
) -> List[JsonDict]:
    scored: List[tuple[float, JsonDict, List[str]]] = []
    candidates: List[JsonDict] = []
    direct_pico_id = str(rec.get("pico_id") or "")
    if direct_pico_id and direct_pico_id in picos_by_id:
        candidates.append(picos_by_id[direct_pico_id])
    for evidence in evidence_rows:
        pico_id = str(evidence.get("pico_id") or "")
        if pico_id and pico_id in picos_by_id:
            candidates.append(picos_by_id[pico_id])
    candidates.extend(pico_index.candidates_for(rec))
    for pico in candidates:
        score, reasons = pico_match_score(rec, pico)
        if score >= options.min_pico_score or str(pico.get("pico_id") or "") == direct_pico_id:
            scored.append((max(score, options.min_pico_score), pico, reasons))
    return dedupe_suggestions(scored, "pico", options.max_pico_suggestions)


def evidence_suggestions(
    rec: JsonDict,
    pico: Optional[JsonDict],
    direct_evidence: List[JsonDict],
    evidence_index: NearbyIndex,
    options: AssociationReviewOptions,
) -> List[JsonDict]:
    rec_id = str(rec.get("candidate_id") or "")
    scored: List[tuple[float, JsonDict, List[str]]] = []
    candidates = list(direct_evidence)
    candidates.extend(row for row in evidence_index.candidates_for(rec) if row not in direct_evidence)
    for evidence in candidates:
        linked_rec = str(evidence.get("recommendation_candidate_id") or "")
        if linked_rec and linked_rec != rec_id:
            continue
        score, reasons = evidence_match_score(rec, evidence, pico)
        if score >= options.min_evidence_score:
            scored.append((score, evidence, reasons))
    return dedupe_suggestions(scored, "evidence", options.max_evidence_suggestions)


def evidence_pico_match_score(evidence: JsonDict, pico: JsonDict) -> tuple[float, List[str]]:
    reasons: List[str] = []
    if not same_record_or_guideline(evidence, pico):
        return 0.0, ["different_record_or_guideline"]
    if not evidence_has_usable_signal(evidence):
        return 0.0, ["evidence_signal_not_usable"]
    distance = order_distance(evidence, pico)
    max_distance = 24
    if distance > max_distance:
        return 0.0, [f"pico_order_distance_{distance}_too_large"]
    score = 0.2 + max(0.0, 0.2 * (1 - distance / max_distance))
    reasons.append("same_record_or_guideline")
    evidence_text = evidence.get("source_text") or evidence.get("source_span")
    population_overlap = token_overlap(evidence_text, pico.get("population"))
    intervention_overlap = token_overlap(evidence_text, pico.get("intervention"))
    question_overlap = token_overlap(evidence_text, pico.get("clinical_question"))
    outcomes_overlap = token_overlap(evidence_text, pico.get("outcomes"))
    section_score = section_overlap(evidence, pico)
    score += 0.2 * population_overlap
    score += 0.24 * intervention_overlap
    score += 0.12 * question_overlap
    score += 0.08 * outcomes_overlap
    score += 0.08 * section_score
    if population_overlap:
        reasons.append("population_overlap")
    if intervention_overlap:
        reasons.append("intervention_overlap")
    if question_overlap:
        reasons.append("question_overlap")
    if outcomes_overlap:
        reasons.append("outcomes_overlap")
    if section_score:
        reasons.append("section_overlap")
    if str(pico.get("status") or "") == "active":
        score += 0.04
        reasons.append("active_pico")
    return round(score, 4), reasons


def evidence_pico_suggestions(
    evidence: JsonDict,
    pico_index: NearbyIndex,
    options: AssociationReviewOptions,
) -> List[JsonDict]:
    scored: List[tuple[float, JsonDict, List[str]]] = []
    for pico in pico_index.candidates_for(evidence):
        score, reasons = evidence_pico_match_score(evidence, pico)
        if score >= options.min_evidence_pico_score:
            scored.append((score, pico, reasons))
    return dedupe_suggestions(scored, "pico", options.max_pico_suggestions)


def suggestion_lookup(suggestions: List[JsonDict], rows_by_id: Dict[str, JsonDict]) -> Optional[JsonDict]:
    if not suggestions:
        return None
    return rows_by_id.get(str(suggestions[0].get("entity_id") or ""))


def linked_rows_from_suggestions(suggestions: List[JsonDict], rows_by_id: Dict[str, JsonDict]) -> List[JsonDict]:
    rows: List[JsonDict] = []
    for suggestion in suggestions:
        row = rows_by_id.get(str(suggestion.get("entity_id") or ""))
        if row:
            rows.append(row)
    return rows


def recommendation_priority(blockers: List[str], grade_count: int, pico_count: int, evidence_count: int) -> str:
    complete_context = bool(grade_count and pico_count and evidence_count)
    non_link_blockers = [reason for reason in blockers if reason not in LINK_BLOCKERS]
    if complete_context and not non_link_blockers:
        return "P0"
    if complete_context or sum(bool(count) for count in [grade_count, pico_count, evidence_count]) >= 2:
        return "P1"
    return "P2"


def recommendation_state(rec: JsonDict) -> JsonDict:
    return {
        "candidate_id": rec.get("candidate_id"),
        "record_id": rec.get("record_id"),
        "guideline_id": rec.get("guideline_id"),
        "source_order": source_order(rec),
        "source_section": rec.get("source_section"),
        "status": rec.get("status"),
        "extraction_confidence": rec.get("extraction_confidence"),
        "recommendation_text": compact_text(rec.get("recommendation_text"), 480),
        "direction": rec.get("direction"),
        "strength": rec.get("strength"),
        "certainty": rec.get("certainty"),
        "population": rec.get("population"),
        "intervention": rec.get("intervention"),
        "pico_id": rec.get("pico_id"),
    }


def build_recommendation_association_items(inputs: BackfillInputs, options: AssociationReviewOptions) -> List[JsonDict]:
    grades_by_rec = grouped_by(inputs.grades, "recommendation_candidate_id")
    evidence_by_rec = grouped_by(inputs.evidence, "recommendation_candidate_id")
    grades_by_id = index_by(inputs.grades, "grade_candidate_id")
    picos_by_id = index_by(inputs.picos, "pico_id")
    evidence_by_id = index_by(inputs.evidence, "evidence_id")
    grade_index = NearbyIndex(inputs.grades)
    pico_index = NearbyIndex(inputs.picos, record_field="source_record_id")
    evidence_index = NearbyIndex(inputs.evidence, record_field="source_record_id")

    items: List[JsonDict] = []
    for rec in inputs.recommendations:
        status = str(rec.get("status") or "")
        if status == "rejected":
            continue
        if association_review_resolved(rec):
            continue
        if status == "accepted" and not options.include_accepted_recommendations:
            continue
        rec_id = str(rec.get("candidate_id") or "")
        direct_evidence = evidence_by_rec.get(rec_id, [])
        grade_links = grade_suggestions(rec, grades_by_rec.get(rec_id, []), grade_index, options)
        pico_links = pico_suggestions(rec, direct_evidence, picos_by_id, pico_index, options)
        selected_pico = suggestion_lookup(pico_links, picos_by_id)
        evidence_links = evidence_suggestions(rec, selected_pico, direct_evidence, evidence_index, options)
        selected_grade = suggestion_lookup(grade_links, grades_by_id)
        selected_evidence = linked_rows_from_suggestions(evidence_links, evidence_by_id)
        _, blockers = can_accept_recommendation(rec, selected_grade, selected_pico, selected_evidence)
        if not blockers and not (grade_links or pico_links or evidence_links):
            continue
        review_reasons = list(dict.fromkeys([*blockers, "association_context_review"]))
        priority = recommendation_priority(blockers, len(grade_links), len(pico_links), len(evidence_links))
        item = {
            "review_id": stable_id("association_review", ASSOCIATION_REVIEW_VERSION, "recommendation", rec_id),
            "review_gate_version": ASSOCIATION_REVIEW_VERSION,
            "entity_type": "recommendation_association",
            "entity_id": rec_id,
            "priority": priority,
            "review_status": "open",
            "review_reasons": review_reasons,
            "current_state": recommendation_state(rec),
            "association_state": {
                "blockers_after_top_suggestions": blockers,
                "suggested_grade_count": len(grade_links),
                "suggested_pico_count": len(pico_links),
                "suggested_evidence_count": len(evidence_links),
                "direct_grade_count": len(grades_by_rec.get(rec_id, [])),
                "direct_evidence_count": len(direct_evidence),
            },
            "suggested_links": {
                "grade_candidates": grade_links,
                "pico_questions": pico_links,
                "evidence_items": evidence_links,
            },
            "reviewed_payload_template": {
                "recommendation_candidate_id": rec_id,
                "grade_candidate_id": grade_links[0]["entity_id"] if grade_links else None,
                "pico_id": pico_links[0]["entity_id"] if pico_links else None,
                "evidence_ids": [item["entity_id"] for item in evidence_links],
            },
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        items.append(item)
    return assign_batches(items, options.batch_size, options.max_recommendation_records)


def evidence_pico_priority(suggestions: List[JsonDict], options: AssociationReviewOptions) -> str:
    if not suggestions:
        return "P2"
    top_score = float(suggestions[0].get("association_score") or 0.0)
    if top_score >= options.high_evidence_pico_score:
        return "P0"
    if top_score >= 0.35:
        return "P1"
    return "P2"


def evidence_pico_review_reasons(evidence: JsonDict, suggestions: List[JsonDict]) -> List[str]:
    reasons = ["missing_pico_id"]
    if not suggestions:
        reasons.append("no_nearby_pico_suggestion")
    else:
        reasons.append("nearby_pico_suggestion")
    if evidence.get("recommendation_candidate_id"):
        reasons.append("has_recommendation_candidate_link")
    if str(evidence.get("screening_status") or "") == "uncertain":
        reasons.append("screening_uncertain")
    return reasons


def evidence_state(evidence: JsonDict) -> JsonDict:
    return {
        "evidence_id": evidence.get("evidence_id"),
        "record_id": evidence.get("record_id") or evidence.get("source_record_id"),
        "guideline_id": evidence.get("guideline_id"),
        "paper_id": evidence.get("paper_id"),
        "source_order": source_order(evidence),
        "source_section": evidence.get("source_section"),
        "screening_status": evidence.get("screening_status"),
        "recommendation_candidate_id": evidence.get("recommendation_candidate_id"),
        "study_design": evidence.get("study_design"),
        "effect_direction": evidence.get("effect_direction"),
        "effect_size": evidence.get("effect_size"),
        "confidence_interval": evidence.get("confidence_interval"),
        "source_text": compact_text(evidence.get("source_text"), 480),
    }


def build_evidence_pico_items(inputs: BackfillInputs, options: AssociationReviewOptions) -> List[JsonDict]:
    pico_index = NearbyIndex(inputs.picos, record_field="source_record_id")
    items: List[JsonDict] = []
    for evidence in inputs.evidence:
        if evidence.get("pico_id"):
            continue
        evidence_id = str(evidence.get("evidence_id") or "")
        suggestions = evidence_pico_suggestions(evidence, pico_index, options)
        priority = evidence_pico_priority(suggestions, options)
        item = {
            "review_id": stable_id("association_review", ASSOCIATION_REVIEW_VERSION, "evidence_pico", evidence_id),
            "review_gate_version": ASSOCIATION_REVIEW_VERSION,
            "entity_type": "evidence_pico_association",
            "entity_id": evidence_id,
            "priority": priority,
            "review_status": "open",
            "review_reasons": evidence_pico_review_reasons(evidence, suggestions),
            "current_state": evidence_state(evidence),
            "association_state": {
                "suggested_pico_count": len(suggestions),
                "top_pico_score": suggestions[0]["association_score"] if suggestions else None,
            },
            "suggested_links": {"pico_questions": suggestions},
            "reviewed_payload_template": {
                "evidence_id": evidence_id,
                "pico_id": suggestions[0]["entity_id"] if suggestions else None,
            },
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        items.append(item)
    return assign_batches(items, options.batch_size, options.max_evidence_pico_records)


def review_group_key(row: JsonDict) -> str:
    current_state = row.get("current_state") if isinstance(row.get("current_state"), dict) else {}
    for key in ["guideline_id", "record_id", "paper_id"]:
        value = str(current_state.get(key) or "")
        if value:
            return value
    return str(row.get("entity_id") or "")


def priority_sort_key(row: JsonDict) -> tuple[int, int, str, str]:
    current_state = row.get("current_state") if isinstance(row.get("current_state"), dict) else {}
    try:
        order = int(current_state.get("source_order") or 0)
    except (TypeError, ValueError):
        order = 0
    return (
        PRIORITY_ORDER.get(str(row.get("priority") or "P2"), 99),
        order,
        str(row.get("entity_type") or ""),
        str(row.get("entity_id") or ""),
    )


def round_robin_by_group(rows: List[JsonDict]) -> List[JsonDict]:
    buckets: Dict[str, List[JsonDict]] = defaultdict(list)
    for row in sorted(rows, key=priority_sort_key):
        buckets[review_group_key(row)].append(row)
    keys = sorted(buckets, key=lambda key: priority_sort_key(buckets[key][0]))
    ordered: List[JsonDict] = []
    while keys:
        next_keys: List[str] = []
        for key in keys:
            bucket = buckets[key]
            ordered.append(bucket.pop(0))
            if bucket:
                next_keys.append(key)
        keys = next_keys
    return ordered


def assign_batches(rows: List[JsonDict], batch_size: int, max_records: Optional[int]) -> List[JsonDict]:
    ordered = round_robin_by_group(rows)
    if max_records is not None and max_records > 0:
        ordered = ordered[:max_records]
    size = batch_size if batch_size > 0 else len(ordered) or 1
    output: List[JsonDict] = []
    for index, row in enumerate(ordered):
        batch_index = index // size + 1
        item = dict(row)
        item["batch_index"] = batch_index
        item["batch_item_index"] = index % size + 1
        item["batch_id"] = stable_id("association_review_batch", ASSOCIATION_REVIEW_VERSION, row.get("entity_type"), batch_index)
        output.append(item)
    return output


def summarize_items(items: List[JsonDict]) -> JsonDict:
    priorities = Counter(str(item.get("priority") or "P2") for item in items)
    reasons = Counter(reason for item in items for reason in item.get("review_reasons", []))
    batches = Counter(str(item.get("batch_index") or "") for item in items)
    with_suggestions = sum(
        1
        for item in items
        if any(item.get("suggested_links", {}).get(key) for key in ["grade_candidates", "pico_questions", "evidence_items"])
    )
    return {
        "records": len(items),
        "records_with_suggestions": with_suggestions,
        "priority_counts": dict(priorities),
        "reason_counts": dict(reasons),
        "batch_count": len([key for key in batches if key]),
    }


def build_association_review_package(
    inputs: BackfillInputs,
    output_dir: str | Path,
    options: AssociationReviewOptions | None = None,
) -> JsonDict:
    options = options or AssociationReviewOptions()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    recommendation_items = build_recommendation_association_items(inputs, options)
    evidence_pico_items = build_evidence_pico_items(inputs, options)
    recommendation_output = output_path / "recommendation_association_review_queue.jsonl"
    evidence_pico_output = output_path / "evidence_pico_review_queue.jsonl"
    summary_output = output_path / "association_review_summary.jsonl"
    write_jsonl(recommendation_output, recommendation_items)
    write_jsonl(evidence_pico_output, evidence_pico_items)
    summary = {
        "association_review_version": ASSOCIATION_REVIEW_VERSION,
        "output_dir": str(output_path),
        "input_counts": {
            "recommendations": len(inputs.recommendations),
            "grades": len(inputs.grades),
            "picos": len(inputs.picos),
            "evidence": len(inputs.evidence),
            "evidence_without_pico": sum(1 for row in inputs.evidence if not row.get("pico_id")),
        },
        "options": {
            "min_grade_score": options.min_grade_score,
            "min_pico_score": options.min_pico_score,
            "min_evidence_score": options.min_evidence_score,
            "min_evidence_pico_score": options.min_evidence_pico_score,
            "batch_size": options.batch_size,
            "max_recommendation_records": options.max_recommendation_records,
            "max_evidence_pico_records": options.max_evidence_pico_records,
        },
        "recommendation_association_queue": {
            **summarize_items(recommendation_items),
            "output_file": str(recommendation_output),
        },
        "evidence_pico_queue": {
            **summarize_items(evidence_pico_items),
            "output_file": str(evidence_pico_output),
            "high_confidence_suggestions": sum(
                1
                for item in evidence_pico_items
                if float(item.get("association_state", {}).get("top_pico_score") or 0.0) >= options.high_evidence_pico_score
            ),
        },
        "created_at": utc_now(),
    }
    write_jsonl(summary_output, [summary])
    return summary


def read_inputs(
    recommendations_input: str | Path,
    grades_input: str | Path,
    picos_input: str | Path,
    evidence_input: str | Path,
) -> BackfillInputs:
    return BackfillInputs(
        recommendations=list(iter_jsonl(recommendations_input)),
        grades=list(iter_jsonl(grades_input)),
        picos=list(iter_jsonl(picos_input)),
        evidence=list(iter_jsonl(evidence_input)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build association-focused review queues.")
    parser.add_argument("--recommendations-input", required=True)
    parser.add_argument("--grades-input", required=True)
    parser.add_argument("--picos-input", required=True)
    parser.add_argument("--evidence-input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-recommendation-records", type=int, default=None)
    parser.add_argument("--max-evidence-pico-records", type=int, default=None)
    parser.add_argument("--min-grade-score", type=float, default=0.45)
    parser.add_argument("--min-pico-score", type=float, default=0.25)
    parser.add_argument("--min-evidence-score", type=float, default=0.35)
    parser.add_argument("--min-evidence-pico-score", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_association_review_package(
        read_inputs(args.recommendations_input, args.grades_input, args.picos_input, args.evidence_input),
        args.output_dir,
        AssociationReviewOptions(
            min_grade_score=args.min_grade_score,
            min_pico_score=args.min_pico_score,
            min_evidence_score=args.min_evidence_score,
            min_evidence_pico_score=args.min_evidence_pico_score,
            batch_size=args.batch_size,
            max_recommendation_records=args.max_recommendation_records,
            max_evidence_pico_records=args.max_evidence_pico_records,
        ),
    )
    print(
        "recommendation_association_records={rec} evidence_pico_records={evi}".format(
            rec=summary["recommendation_association_queue"]["records"],
            evi=summary["evidence_pico_queue"]["records"],
        )
    )


if __name__ == "__main__":
    main()

