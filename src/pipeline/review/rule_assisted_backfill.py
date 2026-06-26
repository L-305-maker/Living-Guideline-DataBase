"""规则辅助复核与数据库构建前字段回填。

本模块用于把已经进入审核包的候选数据先做一轮保守、可审计的自动回填：
- 只接受高置信、低噪声、存在强/中 GRADE 关联且有证据/PICO 上下文的推荐；
- 将 GRADE 的 strength/certainty 回填到推荐候选，减少版本构建冲突；
- 为 RecommendationVersion 补足 block 级 source span 坐标；
- 输出 reviewed JSONL 和 summary，原始 run 文件不被覆盖。

注意：这里的 reviewer 标记为 ``rule_assisted_backfill``，表示机器规则辅助复核，
不是临床专家人工确认。
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.common.extraction_common import utc_now
from src.common.process_jsonl import iter_jsonl, write_jsonl
from src.domain.common import stable_id


JsonDict = Dict[str, Any]

REVIEWER = "rule_assisted_backfill"
REVIEW_VERSION = "rule_assisted_backfill_v1"
BAD_RECOMMENDATION_NOTES = {
    "methodology_statement",
    "evidence_context_statement",
    "missing_clinical_object",
    "starts_with_fragment",
    "starts_with_punctuation",
    "incomplete_action_phrase",
    "citation_heavy",
    "too_short_fragment",
    "short_statement",
}
BAD_GRADE_PHRASES_RE = re.compile(
    r"\b(?:grading system|grade the evidence|quality of evidence was used|methodology|methods?|task force|"
    r"systematic review of the literature was performed)\b",
    re.I,
)
ACTION_RE = re.compile(
    r"\b(?:we\s+(?:recommend|suggest)|should|should\s+not|is\s+recommended|are\s+recommended|"
    r"is\s+indicated|are\s+indicated|offer|consider|use|avoid)\b",
    re.I,
)


@dataclass(frozen=True)
class BackfillInputs:
    recommendations: List[JsonDict]
    grades: List[JsonDict]
    picos: List[JsonDict]
    evidence: List[JsonDict]


def payload(row: JsonDict) -> JsonDict:
    value = row.get("normalized_payload")
    return value if isinstance(value, dict) else {}


def raw_payload(row: JsonDict) -> JsonDict:
    value = row.get("raw_payload")
    return value if isinstance(value, dict) else {}


def source_order(row: JsonDict) -> int:
    try:
        return int(row.get("source_order") or payload(row).get("source_order") or 0)
    except (TypeError, ValueError):
        return 0


def quality_notes(row: JsonDict) -> set[str]:
    notes = payload(row).get("quality_notes")
    return {str(note) for note in notes} if isinstance(notes, list) else set()


def record_quality_flags(row: JsonDict) -> set[str]:
    quality = raw_payload(row).get("quality")
    flags = quality.get("record_quality_flags") if isinstance(quality, dict) else []
    return {str(flag) for flag in flags} if isinstance(flags, list) else set()


def extraction_confidence(row: JsonDict) -> float:
    try:
        return float(row.get("extraction_confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def non_empty(value: Any) -> bool:
    return value not in (None, "", [], {})


def compact_review_record(decision: str, reasons: List[str], note: str = "") -> JsonDict:
    now = utc_now()
    return {
        "reviewer": REVIEWER,
        "review_decision": decision,
        "review_reasons": reasons,
        "review_note": note,
        "reviewed_at": now,
        "applied_at": now,
        "review_version": REVIEW_VERSION,
    }


def attach_backfill_review(row: JsonDict, decision: str, reasons: List[str], reviewed_payload: JsonDict | None = None) -> JsonDict:
    """给实体附加机器复核审计信息，并合并 reviewed_payload。"""

    output = deepcopy(row)
    if reviewed_payload:
        for key, value in reviewed_payload.items():
            output[key] = value
    normalized = dict(payload(output))
    history = normalized.get("manual_review_history")
    if not isinstance(history, list):
        history = []
    record = compact_review_record(decision, reasons)
    history.append(record)
    normalized["manual_review_history"] = history
    normalized["last_manual_review"] = record
    normalized["rule_assisted_backfill"] = {
        "review_version": REVIEW_VERSION,
        "decision": decision,
        "reasons": reasons,
        "applied_at": record["applied_at"],
    }
    output["normalized_payload"] = normalized
    output["updated_at"] = record["applied_at"]
    output["review_note"] = f"{REVIEWER}: {decision}; reasons={','.join(reasons)}"
    return output


def index_by(rows: Iterable[JsonDict], key: str) -> Dict[str, JsonDict]:
    return {str(row.get(key) or ""): row for row in rows if row.get(key)}


def grouped_by(rows: Iterable[JsonDict], key: str) -> Dict[str, List[JsonDict]]:
    grouped: Dict[str, List[JsonDict]] = defaultdict(list)
    for row in rows:
        value = str(row.get(key) or "")
        if value:
            grouped[value].append(row)
    return grouped


def best_grade_for_recommendation(grades: List[JsonDict]) -> Optional[JsonDict]:
    """选择最适合回填的 GRADE；优先强/中关联、字段更完整、置信度更高。"""

    candidates = []
    for grade in grades:
        association_quality = str(payload(grade).get("association_quality") or "")
        if association_quality not in {"strong", "medium"}:
            continue
        if str(grade.get("grade_system") or "") == "unknown":
            continue
        if grade.get("certainty") == "unclear" and grade.get("strength") == "unclear":
            continue
        if BAD_GRADE_PHRASES_RE.search(str(grade.get("source_text") or "")):
            continue
        score = extraction_confidence(grade)
        if grade.get("certainty") != "unclear":
            score += 0.2
        if grade.get("strength") != "unclear":
            score += 0.15
        if association_quality == "strong":
            score += 0.15
        candidates.append((score, grade))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def evidence_for_recommendation(rec: JsonDict, evidence_by_rec: Dict[str, List[JsonDict]]) -> List[JsonDict]:
    rows = evidence_by_rec.get(str(rec.get("candidate_id") or ""), [])
    return [row for row in rows if row.get("pico_id") and str(row.get("screening_status") or "") in {"pending", "included"}]


def best_pico_from_evidence(evidence_rows: List[JsonDict], picos_by_id: Dict[str, JsonDict]) -> Optional[JsonDict]:
    counts = Counter(str(row.get("pico_id") or "") for row in evidence_rows if row.get("pico_id"))
    for pico_id, _ in counts.most_common():
        pico = picos_by_id.get(pico_id)
        if pico:
            return pico
    return None


def source_span_patch(rec: JsonDict) -> JsonDict:
    """补 block 级 source span 坐标，满足版本构建和数据库追溯字段。"""

    source_text = str(rec.get("source_text") or rec.get("source_span") or rec.get("recommendation_text") or "")
    recommendation_text = str(rec.get("recommendation_text") or "")
    start = source_text.find(recommendation_text) if source_text and recommendation_text else -1
    if start < 0:
        start = 0
        end = len(source_text)
    else:
        end = start + len(recommendation_text)
    return {
        "source_span": source_text,
        "source_span_ref": rec.get("source_block_id") or payload(rec).get("block_id"),
        "start_char": start,
        "end_char": end,
    }


def recommendation_acceptance_patch(rec: JsonDict, grade: JsonDict, pico: JsonDict, evidence_rows: List[JsonDict]) -> JsonDict:
    """把高置信关联上下文回填到推荐候选。"""

    patch = {
        "status": "accepted",
        "recommendation_id": stable_id("recommendation", rec.get("guideline_id"), rec.get("candidate_id")),
        "pico_id": pico.get("pico_id"),
        "population": rec.get("population") or pico.get("population"),
        "intervention": rec.get("intervention") or pico.get("intervention"),
        "comparator": rec.get("comparator") or pico.get("comparator"),
        "outcomes": rec.get("outcomes") or pico.get("outcomes") or [],
        "certainty": grade.get("certainty") if grade.get("certainty") != "unclear" else rec.get("certainty"),
        "strength": grade.get("strength") if grade.get("strength") != "unclear" else rec.get("strength"),
    }
    patch.update(source_span_patch(rec))
    normalized = dict(payload(rec))
    normalized["backfilled_links"] = {
        "grade_candidate_id": grade.get("grade_candidate_id"),
        "pico_id": pico.get("pico_id"),
        "evidence_ids": [row.get("evidence_id") for row in evidence_rows[:20]],
    }
    patch["normalized_payload"] = normalized
    return patch


def can_accept_recommendation(rec: JsonDict, grade: Optional[JsonDict], pico: Optional[JsonDict], evidence_rows: List[JsonDict]) -> tuple[bool, List[str]]:
    reasons: List[str] = []
    text = str(rec.get("recommendation_text") or "")
    notes = quality_notes(rec)
    if extraction_confidence(rec) < 0.72:
        reasons.append("recommendation_confidence_below_0_72")
    if notes & BAD_RECOMMENDATION_NOTES:
        reasons.append("recommendation_has_noise_notes")
    if len(text) < 35 or len(text) > 620:
        reasons.append("recommendation_text_length_out_of_range")
    if not ACTION_RE.search(text):
        reasons.append("recommendation_missing_action_signal")
    if not rec.get("guideline_id"):
        reasons.append("missing_guideline_id")
    if "table_heavy_without_structured_tables" in record_quality_flags(rec):
        reasons.append("table_heavy_source_needs_manual_review")
    if not grade:
        reasons.append("no_usable_grade")
    if not pico:
        reasons.append("no_usable_pico")
    if not evidence_rows:
        reasons.append("no_linked_evidence")
    return not reasons, reasons


def backfill_inputs(inputs: BackfillInputs) -> tuple[JsonDict, Dict[str, List[JsonDict]]]:
    """执行规则辅助复核，返回 summary 和各实体 reviewed 行。"""

    grades_by_rec = grouped_by(inputs.grades, "recommendation_candidate_id")
    evidence_by_rec = grouped_by(inputs.evidence, "recommendation_candidate_id")
    picos_by_id = index_by(inputs.picos, "pico_id")
    accepted_rec_ids: set[str] = set()
    accepted_grade_ids: set[str] = set()
    accepted_pico_ids: set[str] = set()
    included_evidence_ids: set[str] = set()
    reject_reasons: Counter[str] = Counter()

    reviewed_recommendations: List[JsonDict] = []
    for rec in inputs.recommendations:
        rec_id = str(rec.get("candidate_id") or "")
        grade = best_grade_for_recommendation(grades_by_rec.get(rec_id, []))
        evidence_rows = evidence_for_recommendation(rec, evidence_by_rec)
        pico = best_pico_from_evidence(evidence_rows, picos_by_id)
        accepted, reasons = can_accept_recommendation(rec, grade, pico, evidence_rows)
        if not accepted:
            reject_reasons.update(reasons)
            reviewed_recommendations.append(rec)
            continue
        assert grade is not None and pico is not None
        patch = recommendation_acceptance_patch(rec, grade, pico, evidence_rows)
        reviewed_recommendations.append(
            attach_backfill_review(
                rec,
                "accepted",
                ["high_confidence_rule_review", "usable_grade_pico_evidence_context"],
                patch,
            )
        )
        accepted_rec_ids.add(rec_id)
        accepted_grade_ids.add(str(grade.get("grade_candidate_id") or ""))
        accepted_pico_ids.add(str(pico.get("pico_id") or ""))
        included_evidence_ids.update(str(row.get("evidence_id") or "") for row in evidence_rows if row.get("evidence_id"))

    reviewed_grades: List[JsonDict] = []
    for grade in inputs.grades:
        grade_id = str(grade.get("grade_candidate_id") or "")
        if grade_id in accepted_grade_ids:
            reviewed_grades.append(
                attach_backfill_review(
                    grade,
                    "accepted",
                    ["selected_as_best_grade_for_accepted_recommendation"],
                    {"status": "accepted"},
                )
            )
        else:
            reviewed_grades.append(grade)

    reviewed_picos: List[JsonDict] = []
    for pico in inputs.picos:
        pico_id = str(pico.get("pico_id") or "")
        if pico_id in accepted_pico_ids and pico.get("status") != "active":
            reviewed_picos.append(
                attach_backfill_review(
                    pico,
                    "accepted",
                    ["linked_to_accepted_recommendation"],
                    {"status": "active"},
                )
            )
        else:
            reviewed_picos.append(pico)

    reviewed_evidence: List[JsonDict] = []
    for item in inputs.evidence:
        evidence_id = str(item.get("evidence_id") or "")
        if evidence_id in included_evidence_ids and item.get("screening_status") != "included":
            reviewed_evidence.append(
                attach_backfill_review(
                    item,
                    "included",
                    ["linked_to_accepted_recommendation"],
                    {"screening_status": "included"},
                )
            )
        else:
            reviewed_evidence.append(item)

    outputs = {
        "recommendations": reviewed_recommendations,
        "grades": reviewed_grades,
        "picos": reviewed_picos,
        "evidence": reviewed_evidence,
    }
    summary = {
        "review_version": REVIEW_VERSION,
        "input_counts": {
            "recommendations": len(inputs.recommendations),
            "grades": len(inputs.grades),
            "picos": len(inputs.picos),
            "evidence": len(inputs.evidence),
        },
        "accepted_recommendations": len(accepted_rec_ids),
        "accepted_grades": len(accepted_grade_ids),
        "activated_picos": len(accepted_pico_ids),
        "included_evidence_items": len(included_evidence_ids),
        "remaining_recommendations_for_review": len(inputs.recommendations) - len(accepted_rec_ids),
        "recommendation_reject_reason_counts": dict(reject_reasons),
    }
    return summary, outputs


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


def run_backfill(
    recommendations_input: str | Path,
    grades_input: str | Path,
    picos_input: str | Path,
    evidence_input: str | Path,
    output_dir: str | Path,
) -> JsonDict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary, outputs = backfill_inputs(read_inputs(recommendations_input, grades_input, picos_input, evidence_input))
    files = {
        "recommendations": output_path / "recommendation_candidates.reviewed.jsonl",
        "grades": output_path / "grade_candidates.reviewed.jsonl",
        "picos": output_path / "pico_questions.reviewed.jsonl",
        "evidence": output_path / "evidence_items.reviewed.jsonl",
        "summary": output_path / "rule_assisted_backfill_summary.jsonl",
    }
    write_jsonl(files["recommendations"], outputs["recommendations"])
    write_jsonl(files["grades"], outputs["grades"])
    write_jsonl(files["picos"], outputs["picos"])
    write_jsonl(files["evidence"], outputs["evidence"])
    write_jsonl(output_path / "recommendation_candidates.jsonl", outputs["recommendations"])
    write_jsonl(output_path / "grade_candidates.jsonl", outputs["grades"])
    write_jsonl(output_path / "pico_questions.jsonl", outputs["picos"])
    write_jsonl(output_path / "evidence_items.jsonl", outputs["evidence"])
    summary["output_files"] = {key: str(value) for key, value in files.items() if key != "summary"}
    write_jsonl(files["summary"], [summary])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="执行规则辅助复核与字段回填。")
    parser.add_argument("--recommendations-input", required=True)
    parser.add_argument("--grades-input", required=True)
    parser.add_argument("--picos-input", required=True)
    parser.add_argument("--evidence-input", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_backfill(
        args.recommendations_input,
        args.grades_input,
        args.picos_input,
        args.evidence_input,
        args.output_dir,
    )
    print(
        "accepted_recommendations={accepted_recommendations} accepted_grades={accepted_grades} "
        "included_evidence={included_evidence_items}".format(**summary)
    )


if __name__ == "__main__":
    main()
