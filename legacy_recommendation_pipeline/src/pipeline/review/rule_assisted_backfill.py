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
    r"systematic review of the literature was performed)\b|(?:分级方法|证据分级方法|文献检索|指南制定方法)",
    re.I,
)
ACTION_RE = re.compile(
    r"\b(?:we\s+(?:recommend|suggest)|should|should\s+not|is\s+recommended|are\s+recommended|"
    r"is\s+indicated|are\s+indicated|offer|consider|use|avoid)\b|"
    r"(?:推荐|建议|应当|应|应该|宜|可考虑|可以考虑|可用于|首选|优先|不推荐|不建议|避免|禁用|适用于)",
    re.I,
)
MAX_ASSOCIATION_ORDER_DISTANCE = 24
MAX_EVIDENCE_REPAIR_ROWS = 30
MIN_DIRECT_WEAK_GRADE_REPAIR_SCORE = 0.82
MIN_NEARBY_GRADE_REPAIR_SCORE = 0.62
MIN_NEARBY_PICO_REPAIR_SCORE = 0.30
MIN_NEARBY_EVIDENCE_REPAIR_SCORE = 0.45
MIN_RECOMMENDATION_CONFIDENCE = 0.68
MIN_RELAXED_RECOMMENDATION_CONFIDENCE = 0.60
STRICT_BAD_RECOMMENDATION_NOTES = {
    "methodology_statement",
    "evidence_context_statement",
    "missing_clinical_object",
    "starts_with_fragment",
    "starts_with_punctuation",
    "incomplete_action_phrase",
}
RELAXABLE_BAD_RECOMMENDATION_NOTES = {
    "citation_heavy",
    "too_short_fragment",
    "short_statement",
}
TOKEN_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "are",
    "was",
    "were",
    "should",
    "recommend",
    "suggest",
    "patients",
    "people",
    "adults",
    "children",
}


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


def source_metadata(row: JsonDict) -> JsonDict:
    metadata = payload(row).get("source_metadata")
    return metadata if isinstance(metadata, dict) else {}


def source_name(row: JsonDict) -> str:
    return str(row.get("source") or source_metadata(row).get("source") or "").lower()


def has_cjk_text(value: Any) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(value or "")))


def is_chinese_context(row: JsonDict) -> bool:
    return source_name(row) == "china" or has_cjk_text(
        " ".join(
            [
                str(row.get("recommendation_text") or ""),
                str(row.get("source_text") or ""),
                str(row.get("source_section") or ""),
            ]
        )
    )


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


def text_tokens(value: Any) -> set[str]:
    """Return lightweight tokens for conservative local association scoring."""

    text = str(value or "").lower()
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", text)
        if len(token) > 2 and token not in TOKEN_STOPWORDS
    }
    zh_terms = re.findall(
        r"(?:血友病|淋巴瘤|白血病|骨髓瘤|血小板减少|感染|移植|化疗|免疫|靶向|"
        r"诊断|治疗|预防|防治|管理|患者|成人|儿童|青少年|抗菌|疫苗|血栓|出血|"
        r"利妥昔单抗|奥妥珠单抗|维奈克拉|苯达莫司汀|泊马度胺|CAR-T|造血干细胞)",
        text,
        flags=re.I,
    )
    tokens.update(term.lower() for term in zh_terms if len(term) >= 2)
    cjk = re.sub(r"[^\u4e00-\u9fff]", "", text)
    tokens.update(cjk[index : index + 2] for index in range(max(0, len(cjk) - 1)))
    tokens.update(cjk[index : index + 3] for index in range(max(0, len(cjk) - 2)))
    return tokens


def token_overlap(left: Any, right: Any) -> float:
    left_tokens = text_tokens(left)
    right_tokens = text_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def section_overlap(left: JsonDict, right: JsonDict) -> float:
    return token_overlap(left.get("source_section") or payload(left).get("source_metadata"), right.get("source_section") or payload(right).get("source_metadata"))


def order_distance(left: JsonDict, right: JsonDict) -> int:
    return abs(source_order(left) - source_order(right))


def same_record_or_guideline(left: JsonDict, right: JsonDict) -> bool:
    if str(left.get("record_id") or "") and str(left.get("record_id") or "") == str(right.get("record_id") or right.get("source_record_id") or ""):
        return True
    return bool(str(left.get("guideline_id") or "") and str(left.get("guideline_id") or "") == str(right.get("guideline_id") or ""))


def merge_payload(row: JsonDict, values: JsonDict) -> JsonDict:
    normalized = dict(payload(row))
    normalized.update(values)
    return normalized


class NearbyIndex:
    """Small in-memory index for same-record/guideline association repair."""

    def __init__(self, rows: Iterable[JsonDict], *, record_field: str = "record_id") -> None:
        self.by_record: Dict[str, List[JsonDict]] = defaultdict(list)
        self.by_guideline: Dict[str, List[JsonDict]] = defaultdict(list)
        for row in rows:
            record_id = str(row.get(record_field) or row.get("source_record_id") or row.get("record_id") or "")
            guideline_id = str(row.get("guideline_id") or "")
            if record_id:
                self.by_record[record_id].append(row)
            if guideline_id:
                self.by_guideline[guideline_id].append(row)
        for group in [*self.by_record.values(), *self.by_guideline.values()]:
            group.sort(key=source_order)

    def candidates_for(self, row: JsonDict) -> List[JsonDict]:
        record_id = str(row.get("record_id") or row.get("source_record_id") or "")
        guideline_id = str(row.get("guideline_id") or "")
        return self.by_record.get(record_id) or self.by_guideline.get(guideline_id) or []


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


def grade_signal_is_usable(grade: JsonDict) -> bool:
    if str(grade.get("grade_system") or "") == "unknown":
        return False
    if grade.get("certainty") == "unclear" and grade.get("strength") == "unclear":
        return False
    if BAD_GRADE_PHRASES_RE.search(str(grade.get("source_text") or "")):
        return False
    return True


def candidate_grade_score(rec: JsonDict, grade: JsonDict) -> tuple[float, List[str]]:
    reasons: List[str] = []
    if not same_record_or_guideline(rec, grade):
        return 0.0, ["different_record_or_guideline"]
    distance = order_distance(rec, grade)
    if distance > MAX_ASSOCIATION_ORDER_DISTANCE:
        return 0.0, [f"grade_order_distance_{distance}_too_large"]
    if not grade_signal_is_usable(grade):
        return 0.0, ["grade_signal_not_usable"]

    association_quality = str(payload(grade).get("association_quality") or "")
    score = extraction_confidence(grade)
    if association_quality == "strong":
        score += 0.25
        reasons.append("strong_existing_grade_association")
    elif association_quality == "medium":
        score += 0.18
        reasons.append("medium_existing_grade_association")
    elif association_quality == "weak":
        score += 0.04
        reasons.append("weak_existing_grade_association")

    if grade.get("recommendation_candidate_id") == rec.get("candidate_id"):
        score += 0.35
        reasons.append("direct_recommendation_candidate_id")
    elif not grade.get("recommendation_candidate_id"):
        score += 0.08
        reasons.append("missing_grade_link_repaired")

    score += max(0.0, 0.18 * (1 - distance / MAX_ASSOCIATION_ORDER_DISTANCE))
    text_score = token_overlap(rec.get("recommendation_text"), grade.get("source_text") or grade.get("judgement_rationale"))
    section_score = section_overlap(rec, grade)
    score += 0.18 * text_score
    score += 0.08 * section_score
    if text_score >= 0.08:
        reasons.append("recommendation_grade_text_overlap")
    if section_score > 0:
        reasons.append("recommendation_grade_section_overlap")
    return round(score, 4), reasons


def best_grade_for_recommendation_with_repair(
    rec: JsonDict,
    direct_grades: List[JsonDict],
    grade_index: NearbyIndex,
    used_repaired_grade_ids: set[str],
) -> tuple[Optional[JsonDict], JsonDict]:
    direct = best_grade_for_recommendation(direct_grades)
    if direct:
        return direct, {"association_repair": False, "association_reason": "direct_grade_link"}

    scored_direct: List[tuple[float, JsonDict, List[str]]] = []
    for grade in direct_grades:
        score, reasons = candidate_grade_score(rec, grade)
        if score >= MIN_DIRECT_WEAK_GRADE_REPAIR_SCORE:
            scored_direct.append((score, grade, reasons))
    if scored_direct:
        score, grade, reasons = max(scored_direct, key=lambda item: item[0])
        return grade, {
            "association_repair": True,
            "association_reason": "direct_weak_grade_score_repair",
            "association_score": score,
            "association_reasons": reasons,
        }

    scored: List[tuple[float, JsonDict, List[str]]] = []
    for grade in grade_index.candidates_for(rec):
        grade_id = str(grade.get("grade_candidate_id") or "")
        if grade_id in used_repaired_grade_ids:
            continue
        if grade.get("recommendation_candidate_id"):
            continue
        score, reasons = candidate_grade_score(rec, grade)
        if score >= MIN_NEARBY_GRADE_REPAIR_SCORE:
            scored.append((score, grade, reasons))
    if not scored:
        return None, {"association_repair": False, "association_reason": "no_usable_grade"}
    score, grade, reasons = max(scored, key=lambda item: item[0])
    return grade, {
        "association_repair": True,
        "association_reason": "nearby_grade_repair",
        "association_score": score,
        "association_reasons": reasons,
    }


def evidence_has_usable_signal(row: JsonDict) -> bool:
    if str(row.get("screening_status") or "") not in {"pending", "included", "association_review"}:
        return False
    normalized = payload(row)
    if normalized.get("has_structured_evidence_signal") is True:
        return True
    return any(
        [
            row.get("study_design") not in (None, "", "unclear", "guideline"),
            bool(row.get("effect_size")),
            bool(row.get("confidence_interval")),
            str(row.get("effect_direction") or "") not in {"", "uncertain"},
            bool(row.get("outcomes_extracted")),
        ]
    )


def evidence_for_recommendation(rec: JsonDict, evidence_by_rec: Dict[str, List[JsonDict]]) -> List[JsonDict]:
    rows = evidence_by_rec.get(str(rec.get("candidate_id") or ""), [])
    return [
        row
        for row in rows
        if row.get("pico_id") and (str(row.get("screening_status") or "") in {"pending", "included"} or evidence_has_usable_signal(row))
    ]


def best_pico_from_evidence(evidence_rows: List[JsonDict], picos_by_id: Dict[str, JsonDict]) -> Optional[JsonDict]:
    counts = Counter(str(row.get("pico_id") or "") for row in evidence_rows if row.get("pico_id"))
    for pico_id, _ in counts.most_common():
        pico = picos_by_id.get(pico_id)
        if pico:
            return pico
    return None


def pico_match_score(rec: JsonDict, pico: JsonDict) -> tuple[float, List[str]]:
    if not same_record_or_guideline(rec, pico):
        return 0.0, ["different_record_or_guideline"]
    distance = order_distance(rec, pico)
    if distance > MAX_ASSOCIATION_ORDER_DISTANCE:
        return 0.0, [f"pico_order_distance_{distance}_too_large"]
    score = 0.25
    reasons = ["same_record_or_guideline"]
    score += max(0.0, 0.22 * (1 - distance / MAX_ASSOCIATION_ORDER_DISTANCE))
    population_overlap = token_overlap(rec.get("recommendation_text"), pico.get("population"))
    intervention_overlap = token_overlap(rec.get("recommendation_text"), pico.get("intervention"))
    question_overlap = token_overlap(rec.get("recommendation_text"), pico.get("clinical_question"))
    section_score = section_overlap(rec, pico)
    score += 0.2 * population_overlap
    score += 0.22 * intervention_overlap
    score += 0.08 * question_overlap
    score += 0.08 * section_score
    if population_overlap:
        reasons.append("population_overlap")
    if intervention_overlap:
        reasons.append("intervention_overlap")
    if question_overlap:
        reasons.append("question_overlap")
    if section_score:
        reasons.append("section_overlap")
    if str(pico.get("status") or "") == "active":
        score += 0.05
        reasons.append("active_pico")
    return round(score, 4), reasons


def best_pico_for_recommendation(
    rec: JsonDict,
    evidence_rows: List[JsonDict],
    picos_by_id: Dict[str, JsonDict],
    pico_index: NearbyIndex,
) -> tuple[Optional[JsonDict], JsonDict]:
    direct_pico = best_pico_from_evidence(evidence_rows, picos_by_id)
    if direct_pico:
        return direct_pico, {"association_repair": False, "association_reason": "evidence_pico_majority"}

    direct_id = str(rec.get("pico_id") or "")
    if direct_id and direct_id in picos_by_id:
        return picos_by_id[direct_id], {"association_repair": False, "association_reason": "direct_candidate_pico_id"}

    scored: List[tuple[float, JsonDict, List[str]]] = []
    for pico in pico_index.candidates_for(rec):
        score, reasons = pico_match_score(rec, pico)
        if score >= MIN_NEARBY_PICO_REPAIR_SCORE:
            scored.append((score, pico, reasons))
    if not scored:
        return None, {"association_repair": False, "association_reason": "no_usable_pico"}
    score, pico, reasons = max(scored, key=lambda item: item[0])
    return pico, {
        "association_repair": True,
        "association_reason": "nearby_pico_repair",
        "association_score": score,
        "association_reasons": reasons,
    }


def evidence_match_score(rec: JsonDict, evidence: JsonDict, pico: Optional[JsonDict]) -> tuple[float, List[str]]:
    if not same_record_or_guideline(rec, evidence):
        return 0.0, ["different_record_or_guideline"]
    linked_rec_id = str(evidence.get("recommendation_candidate_id") or "")
    rec_id = str(rec.get("candidate_id") or "")
    if linked_rec_id and linked_rec_id != rec_id:
        return 0.0, ["evidence_linked_to_other_recommendation"]
    if not evidence_has_usable_signal(evidence):
        return 0.0, ["evidence_signal_not_usable"]
    distance = order_distance(rec, evidence)
    if distance > MAX_ASSOCIATION_ORDER_DISTANCE:
        return 0.0, [f"evidence_order_distance_{distance}_too_large"]
    score = 0.25
    reasons = ["same_record_or_guideline"]
    if evidence.get("recommendation_candidate_id") == rec.get("candidate_id"):
        score += 0.35
        reasons.append("direct_recommendation_candidate_id")
    elif not evidence.get("recommendation_candidate_id"):
        score += 0.08
        reasons.append("missing_evidence_recommendation_link_repaired")
    if pico and evidence.get("pico_id") == pico.get("pico_id"):
        score += 0.2
        reasons.append("direct_pico_id")
    elif pico and not evidence.get("pico_id"):
        score += 0.08
        reasons.append("missing_evidence_pico_link_repaired")
    score += max(0.0, 0.18 * (1 - distance / MAX_ASSOCIATION_ORDER_DISTANCE))
    text_score = token_overlap(rec.get("recommendation_text"), evidence.get("source_text") or evidence.get("source_span"))
    section_score = section_overlap(rec, evidence)
    score += 0.12 * text_score
    score += 0.08 * section_score
    if text_score >= 0.05:
        reasons.append("recommendation_evidence_text_overlap")
    if section_score:
        reasons.append("recommendation_evidence_section_overlap")
    return round(score, 4), reasons


def evidence_for_recommendation_with_repair(
    rec: JsonDict,
    pico: Optional[JsonDict],
    evidence_by_rec: Dict[str, List[JsonDict]],
    evidence_index: NearbyIndex,
    used_evidence_ids: set[str],
) -> tuple[List[JsonDict], JsonDict]:
    direct = [row for row in evidence_for_recommendation(rec, evidence_by_rec) if str(row.get("evidence_id") or "") not in used_evidence_ids]
    if direct:
        return direct, {"association_repair": False, "association_reason": "direct_evidence_link"}

    scored: List[tuple[float, JsonDict, List[str]]] = []
    for evidence in evidence_index.candidates_for(rec):
        evidence_id = str(evidence.get("evidence_id") or "")
        if evidence_id in used_evidence_ids:
            continue
        score, reasons = evidence_match_score(rec, evidence, pico)
        if score >= MIN_NEARBY_EVIDENCE_REPAIR_SCORE:
            scored.append((score, evidence, reasons))
    if not scored:
        return [], {"association_repair": False, "association_reason": "no_usable_evidence"}
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [item[1] for item in scored[:MAX_EVIDENCE_REPAIR_ROWS]]
    return selected, {
        "association_repair": True,
        "association_reason": "nearby_evidence_repair",
        "association_scores": [item[0] for item in scored[:MAX_EVIDENCE_REPAIR_ROWS]],
        "association_reasons": [item[2] for item in scored[:MAX_EVIDENCE_REPAIR_ROWS]],
    }


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


def recommendation_acceptance_patch(
    rec: JsonDict,
    grade: JsonDict,
    pico: JsonDict,
    evidence_rows: List[JsonDict],
    *,
    grade_association: JsonDict | None = None,
    pico_association: JsonDict | None = None,
    evidence_association: JsonDict | None = None,
) -> JsonDict:
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
        "grade_association": grade_association or {},
        "pico_association": pico_association or {},
        "evidence_association": evidence_association or {},
    }
    patch["normalized_payload"] = normalized
    return patch


def recommendation_text_length_ok(rec: JsonDict, text: str) -> bool:
    if is_chinese_context(rec):
        return 10 <= len(text) <= 760
    return 28 <= len(text) <= 680


def relaxed_noise_notes_ok(rec: JsonDict, notes: set[str]) -> bool:
    relaxable = notes & RELAXABLE_BAD_RECOMMENDATION_NOTES
    if not relaxable:
        return True
    text = str(rec.get("recommendation_text") or "")
    unresolved = set(relaxable)
    if "citation_heavy" in relaxable and ACTION_RE.search(text) and len(text) >= 20:
        unresolved.discard("citation_heavy")
    if {"short_statement", "too_short_fragment"} & relaxable:
        if is_chinese_context(rec) and ACTION_RE.search(text) and len(text) >= 10:
            unresolved.discard("short_statement")
            unresolved.discard("too_short_fragment")
    return not unresolved


def can_accept_recommendation(rec: JsonDict, grade: Optional[JsonDict], pico: Optional[JsonDict], evidence_rows: List[JsonDict]) -> tuple[bool, List[str]]:
    reasons: List[str] = []
    text = str(rec.get("recommendation_text") or "")
    notes = quality_notes(rec)
    min_confidence = MIN_RELAXED_RECOMMENDATION_CONFIDENCE if is_chinese_context(rec) else MIN_RECOMMENDATION_CONFIDENCE
    if extraction_confidence(rec) < min_confidence:
        reasons.append(f"recommendation_confidence_below_{str(min_confidence).replace('.', '_')}")
    if notes & STRICT_BAD_RECOMMENDATION_NOTES:
        reasons.append("recommendation_has_noise_notes")
        reasons.append("recommendation_has_strict_noise_notes")
    if not relaxed_noise_notes_ok(rec, notes):
        reasons.append("recommendation_has_unresolved_relaxable_noise_notes")
    if not recommendation_text_length_ok(rec, text):
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
    grade_index = NearbyIndex(inputs.grades)
    pico_index = NearbyIndex(inputs.picos, record_field="source_record_id")
    evidence_index = NearbyIndex(inputs.evidence, record_field="source_record_id")
    picos_by_id = index_by(inputs.picos, "pico_id")
    accepted_rec_ids: set[str] = set()
    accepted_grade_ids: set[str] = set()
    accepted_pico_ids: set[str] = set()
    included_evidence_ids: set[str] = set()
    used_repaired_grade_ids: set[str] = set()
    grade_link_repairs: Dict[str, JsonDict] = {}
    evidence_link_repairs: Dict[str, JsonDict] = {}
    reject_reasons: Counter[str] = Counter()
    association_repairs: Counter[str] = Counter()

    reviewed_recommendations: List[JsonDict] = []
    for rec in inputs.recommendations:
        rec_id = str(rec.get("candidate_id") or "")
        grade, grade_association = best_grade_for_recommendation_with_repair(
            rec,
            grades_by_rec.get(rec_id, []),
            grade_index,
            used_repaired_grade_ids,
        )
        early_evidence_rows = evidence_for_recommendation(rec, evidence_by_rec)
        pico, pico_association = best_pico_for_recommendation(rec, early_evidence_rows, picos_by_id, pico_index)
        evidence_rows, evidence_association = evidence_for_recommendation_with_repair(
            rec,
            pico,
            evidence_by_rec,
            evidence_index,
            included_evidence_ids,
        )
        if not pico and evidence_rows:
            pico, pico_association = best_pico_for_recommendation(rec, evidence_rows, picos_by_id, pico_index)
        accepted, reasons = can_accept_recommendation(rec, grade, pico, evidence_rows)
        if not accepted:
            reject_reasons.update(reasons)
            reviewed_recommendations.append(rec)
            continue
        assert grade is not None and pico is not None
        grade_id = str(grade.get("grade_candidate_id") or "")
        if grade_association.get("association_repair"):
            used_repaired_grade_ids.add(grade_id)
            association_repairs["grade_link_repaired"] += 1
            grade_link_repairs[grade_id] = {
                "recommendation_candidate_id": rec_id,
                "normalized_payload": merge_payload(
                    grade,
                    {
                        "association_quality": "strong",
                        "association_reason": grade_association.get("association_reason"),
                        "association_repair": grade_association,
                    },
                ),
            }
        if pico_association.get("association_repair"):
            association_repairs["pico_link_repaired"] += 1
        if evidence_association.get("association_repair"):
            association_repairs["evidence_link_repaired"] += 1
        patch = recommendation_acceptance_patch(
            rec,
            grade,
            pico,
            evidence_rows,
            grade_association=grade_association,
            pico_association=pico_association,
            evidence_association=evidence_association,
        )
        reviewed_recommendations.append(
            attach_backfill_review(
                rec,
                "accepted",
                ["high_confidence_rule_review", "usable_grade_pico_evidence_context"],
                patch,
            )
        )
        accepted_rec_ids.add(rec_id)
        accepted_grade_ids.add(grade_id)
        accepted_pico_ids.add(str(pico.get("pico_id") or ""))
        for row in evidence_rows:
            evidence_id = str(row.get("evidence_id") or "")
            if not evidence_id:
                continue
            included_evidence_ids.add(evidence_id)
            evidence_patch: JsonDict = {}
            if row.get("recommendation_candidate_id") != rec_id:
                evidence_patch["recommendation_candidate_id"] = rec_id
            if row.get("pico_id") != pico.get("pico_id"):
                evidence_patch["pico_id"] = pico.get("pico_id")
            if evidence_patch:
                evidence_patch["normalized_payload"] = merge_payload(
                    row,
                    {
                        "association_repair": {
                            "recommendation_candidate_id": rec_id,
                            "pico_id": pico.get("pico_id"),
                            "reason": evidence_association.get("association_reason"),
                        }
                    },
                )
                evidence_link_repairs[evidence_id] = evidence_patch

    reviewed_grades: List[JsonDict] = []
    for grade in inputs.grades:
        grade_id = str(grade.get("grade_candidate_id") or "")
        if grade_id in accepted_grade_ids:
            patch = {"status": "accepted"}
            patch.update(grade_link_repairs.get(grade_id, {}))
            reviewed_grades.append(
                attach_backfill_review(
                    grade,
                    "accepted",
                    ["selected_as_best_grade_for_accepted_recommendation"],
                    patch,
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
        if evidence_id in included_evidence_ids and (item.get("screening_status") != "included" or evidence_id in evidence_link_repairs):
            patch = {"screening_status": "included"}
            patch.update(evidence_link_repairs.get(evidence_id, {}))
            reviewed_evidence.append(
                attach_backfill_review(
                    item,
                    "included",
                    ["linked_to_accepted_recommendation"],
                    patch,
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
        "association_repair_counts": dict(association_repairs),
        "repaired_grade_links": len(grade_link_repairs),
        "repaired_evidence_links": len(evidence_link_repairs),
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
