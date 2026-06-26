"""从增强后的候选推荐生成 RecommendationVersion。

本阶段只把 accepted 推荐候选物化为指南知识，并把可选的 GRADE、PICO、
证据和指南画像上下文写入版本 payload，方便下游指南库消费者审计来源。
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.domain.common import stable_id
from src.domain.knowledge import RecommendationVersion
from src.common.process_jsonl import iter_jsonl, write_jsonl


JsonDict = Dict[str, Any]
BUILDER_VERSION = "recommendation_version_builder_v2"
MAX_PICO_ORDER_DISTANCE = 12
MIN_PICO_MATCH_SCORE = 0.28
STRONG_PICO_MATCH_SCORE = 0.45


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def payload(row: JsonDict) -> JsonDict:
    value = row.get("normalized_payload")
    return value if isinstance(value, dict) else {}


def enhancement(row: JsonDict) -> JsonDict:
    value = payload(row).get("enhancement")
    return value if isinstance(value, dict) else {}


def auto_qc(row: JsonDict) -> JsonDict:
    value = payload(row).get("auto_qc")
    return value if isinstance(value, dict) else {}


def profile_context(row: JsonDict) -> JsonDict:
    context = enhancement(row).get("profile_context") or auto_qc(row).get("profile_context")
    return context if isinstance(context, dict) else {}


def index_grades_by_recommendation(grades: Iterable[JsonDict]) -> Dict[str, List[JsonDict]]:
    grouped: Dict[str, List[JsonDict]] = defaultdict(list)
    for grade in grades:
        rec_id = str(grade.get("recommendation_candidate_id") or "")
        if rec_id:
            grouped[rec_id].append(grade)
    return grouped


def source_order(row: JsonDict) -> int:
    if "source_order" in row:
        return _as_int(row.get("source_order"))
    return _as_int(payload(row).get("source_order"))


def block_id(row: JsonDict) -> str:
    return str(payload(row).get("block_id") or row.get("source_block_id") or "")


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def text_tokens(value: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(value or "").lower())
        if len(token) > 2 and token not in {"the", "and", "for", "with", "that", "this", "are", "was", "were", "from"}
    }


def token_overlap(left: Any, right: Any) -> float:
    left_tokens = text_tokens(left)
    right_tokens = text_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return round(len(left_tokens & right_tokens) / len(left_tokens | right_tokens), 4)


def section_text(row: JsonDict) -> str:
    metadata = payload(row).get("source_metadata")
    if isinstance(metadata, dict):
        section_path = metadata.get("section_path")
        if isinstance(section_path, list):
            return " ".join(str(item) for item in section_path)
    return str(row.get("source_section") or "")


def order_proximity(left: JsonDict, right: JsonDict, max_distance: int = MAX_PICO_ORDER_DISTANCE) -> tuple[float, int]:
    distance = abs(source_order(left) - source_order(right))
    if distance > max_distance:
        return 0.0, distance
    return round(1.0 - (distance / max_distance), 4), distance


def pico_match_score(rec: JsonDict, pico: JsonDict) -> tuple[float, JsonDict]:
    """按 ID、顺序、章节和文本 token 计算 recommendation 到 PICO 的匹配分。"""

    order_score, distance = order_proximity(rec, pico)
    population_overlap = token_overlap(rec.get("population") or rec.get("recommendation_text"), pico.get("population"))
    intervention_overlap = token_overlap(rec.get("intervention") or rec.get("recommendation_text"), pico.get("intervention"))
    question_overlap = token_overlap(rec.get("recommendation_text"), pico.get("clinical_question"))
    section_overlap = token_overlap(section_text(rec), pico.get("source_section") or section_text(pico))
    same_record = bool(str(rec.get("record_id") or "") and str(rec.get("record_id") or "") == str(pico.get("source_record_id") or pico.get("record_id") or ""))
    same_guideline = bool(str(rec.get("guideline_id") or "") and str(rec.get("guideline_id") or "") == str(pico.get("guideline_id") or ""))

    score = 0.0
    score += 0.15 if same_record else 0.0
    score += 0.1 if same_guideline else 0.0
    score += 0.2 * order_score
    score += 0.1 * section_overlap
    score += 0.2 * population_overlap
    score += 0.2 * intervention_overlap
    score += 0.05 * question_overlap
    features = {
        "same_record": same_record,
        "same_guideline": same_guideline,
        "source_order_distance": distance,
        "order_proximity": order_score,
        "section_overlap": section_overlap,
        "population_overlap": population_overlap,
        "intervention_overlap": intervention_overlap,
        "question_overlap": question_overlap,
    }
    return round(score, 4), features


class PicoIndex:
    """为 PICO 行建立直接 ID 和近邻评分关联索引。"""

    def __init__(self, picos: Iterable[JsonDict]) -> None:
        self.by_id: Dict[str, JsonDict] = {}
        self.by_record: Dict[str, List[JsonDict]] = defaultdict(list)
        self.by_guideline: Dict[str, List[JsonDict]] = defaultdict(list)
        for pico in picos:
            pico_id = str(pico.get("pico_id") or "")
            record_id = str(pico.get("source_record_id") or pico.get("record_id") or "")
            guideline_id = str(pico.get("guideline_id") or "")
            if pico_id:
                self.by_id[pico_id] = pico
            if record_id:
                self.by_record[record_id].append(pico)
            if guideline_id:
                self.by_guideline[guideline_id].append(pico)
        for rows in [*self.by_record.values(), *self.by_guideline.values()]:
            rows.sort(key=source_order)

    def find(self, rec: JsonDict) -> tuple[Optional[JsonDict], JsonDict]:
        """返回与推荐最匹配的 PICO 行及匹配元数据。"""

        direct_id = str(rec.get("pico_id") or "")
        if direct_id and direct_id in self.by_id:
            return self.by_id[direct_id], {
                "association_reason": "direct_candidate_pico_id",
                "match_score": 1.0,
                "match_features": {"direct_candidate_pico_id": True},
            }

        candidates = self.by_record.get(str(rec.get("record_id") or "")) or self.by_guideline.get(str(rec.get("guideline_id") or "")) or []
        if not candidates:
            return None, {"association_reason": "no_pico_candidates", "match_score": 0.0, "match_features": {}}

        scored: List[tuple[float, JsonDict, JsonDict]] = []
        for pico in candidates:
            score, features = pico_match_score(rec, pico)
            scored.append((score, features, pico))
        score, features, best = max(scored, key=lambda item: item[0])
        reason = "generic_score_match" if score >= MIN_PICO_MATCH_SCORE else "low_pico_match_score"
        match = {"association_reason": reason, "match_score": score, "match_features": features}
        if score >= MIN_PICO_MATCH_SCORE:
            return best, match
        return None, match


class EvidenceIndex:
    """按 recommendation 和 PICO 关联键索引 EvidenceItem。"""

    def __init__(self, evidence_items: Iterable[JsonDict]) -> None:
        self.by_rec: Dict[str, List[JsonDict]] = defaultdict(list)
        self.by_pico: Dict[str, List[JsonDict]] = defaultdict(list)
        for item in evidence_items:
            rec_id = str(item.get("recommendation_candidate_id") or "")
            pico_id = str(item.get("pico_id") or "")
            if rec_id:
                self.by_rec[rec_id].append(item)
            if pico_id:
                self.by_pico[pico_id].append(item)

    def find(self, rec: JsonDict, pico_id: Optional[str]) -> tuple[List[JsonDict], str]:
        """优先返回直接关联推荐的证据；没有时回退到 PICO 关联证据。"""

        rec_id = str(rec.get("candidate_id") or "")
        direct = list(self.by_rec.get(rec_id) or [])
        if direct:
            return sorted(direct, key=source_order), "recommendation_candidate_id"
        if pico_id:
            by_pico = list(self.by_pico.get(pico_id) or [])
            if by_pico:
                return sorted(by_pico, key=source_order), "pico_id"
        return [], "not_found"


def grade_rank(grade: JsonDict) -> tuple[int, float]:
    status_rank = {"accepted": 0, "pending": 1, "needs_review": 2, "rejected": 3}
    score = enhancement(grade).get("auto_qc_score")
    score_value = float(score) if isinstance(score, (int, float)) else -1.0
    return status_rank.get(str(grade.get("status") or ""), 9), -score_value


def best_accepted_grade(grades: List[JsonDict]) -> Optional[JsonDict]:
    accepted = [grade for grade in grades if grade.get("status") == "accepted"]
    if not accepted:
        return None
    return sorted(accepted, key=grade_rank)[0]


def summarize_outcomes(outcomes: Any) -> Optional[str]:
    if not isinstance(outcomes, list) or not outcomes:
        return None
    values: List[str] = []
    for item in outcomes:
        if isinstance(item, dict):
            value = item.get("name") or item.get("outcome") or item.get("label")
        else:
            value = item
        if value:
            values.append(str(value))
    return "; ".join(values[:8]) if values else None


def field_conflicts(rec: JsonDict, grade: Optional[JsonDict]) -> List[JsonDict]:
    if not grade:
        return []
    conflicts: List[JsonDict] = []
    for field in ["strength", "certainty"]:
        rec_value = rec.get(field)
        grade_value = grade.get(field)
        if rec_value and grade_value and rec_value != "unclear" and grade_value != "unclear" and rec_value != grade_value:
            conflicts.append({"field": field, "recommendation_value": rec_value, "grade_value": grade_value})
    return conflicts


def compact_grade_link(grade: JsonDict) -> JsonDict:
    return {
        "grade_candidate_id": grade.get("grade_candidate_id"),
        "status": grade.get("status"),
        "grade_system": grade.get("grade_system"),
        "strength": grade.get("strength"),
        "certainty": grade.get("certainty"),
        "auto_qc_decision": enhancement(grade).get("auto_qc_decision"),
        "auto_qc_score": enhancement(grade).get("auto_qc_score"),
        "association_reason": payload(grade).get("association_reason"),
    }


def compact_pico(pico: Optional[JsonDict], match: JsonDict) -> JsonDict:
    if not pico:
        return {
            "pico_id": None,
            "association_reason": match.get("association_reason"),
            "match_score": match.get("match_score", 0.0),
            "match_features": match.get("match_features", {}),
        }
    return {
        "pico_id": pico.get("pico_id"),
        "association_reason": match.get("association_reason"),
        "match_score": match.get("match_score", 0.0),
        "match_quality": "strong" if float(match.get("match_score") or 0.0) >= STRONG_PICO_MATCH_SCORE else "weak",
        "match_features": match.get("match_features", {}),
        "clinical_question": pico.get("clinical_question"),
        "population": pico.get("population"),
        "intervention": pico.get("intervention"),
        "comparator": pico.get("comparator"),
        "outcomes": pico.get("outcomes", []),
        "extraction_confidence": pico.get("extraction_confidence"),
        "source_block_id": pico.get("source_block_id"),
        "source_order": pico.get("source_order"),
    }


def compact_evidence(item: JsonDict) -> JsonDict:
    return {
        "evidence_id": item.get("evidence_id"),
        "pico_id": item.get("pico_id"),
        "recommendation_candidate_id": item.get("recommendation_candidate_id"),
        "study_design": item.get("study_design"),
        "effect_direction": item.get("effect_direction"),
        "screening_status": item.get("screening_status"),
        "extraction_confidence": item.get("extraction_confidence"),
        "source_block_id": item.get("source_block_id"),
        "source_order": item.get("source_order"),
    }


def summarize_evidence(items: List[JsonDict], association_reason: str) -> JsonDict:
    return {
        "association_reason": association_reason,
        "linked_evidence_count": len(items),
        "linked_evidence_ids": [item.get("evidence_id") for item in items[:30]],
        "study_design_counts": dict(Counter(str(item.get("study_design") or "unknown") for item in items)),
        "effect_direction_counts": dict(Counter(str(item.get("effect_direction") or "unknown") for item in items)),
        "screening_status_counts": dict(Counter(str(item.get("screening_status") or "unknown") for item in items)),
        "evidence_items": [compact_evidence(item) for item in items[:20]],
    }


def snapshot(row: JsonDict, keep_text: bool = False) -> JsonDict:
    keys = [
        "candidate_id",
        "grade_candidate_id",
        "recommendation_candidate_id",
        "record_id",
        "guideline_id",
        "model_trace_id",
        "status",
        "extraction_method",
        "extraction_confidence",
        "created_at",
        "updated_at",
    ]
    out = {key: row.get(key) for key in keys if key in row}
    if keep_text:
        out["source_text"] = row.get("source_text")
        out["recommendation_text"] = row.get("recommendation_text")
    return out


@dataclass
class VersionBuildInput:
    """构造一条 RecommendationVersion 所需的输入集合。"""

    rec: JsonDict
    linked_grades: List[JsonDict]
    previous_version: Optional[JsonDict] = None
    linked_pico: Optional[JsonDict] = None
    pico_match: JsonDict = field(default_factory=dict)
    evidence_rows: List[JsonDict] = field(default_factory=list)
    evidence_reason: str = "not_provided"


@dataclass
class VersionBuildContext:
    """构建版本字段、payload 和报告行时使用的派生上下文。"""

    rec: JsonDict
    linked_grades: List[JsonDict]
    accepted_grade: Optional[JsonDict]
    linked_pico: Optional[JsonDict]
    pico_match: JsonDict
    evidence_rows: List[JsonDict]
    evidence_reason: str
    evidence_summary: JsonDict
    profile: JsonDict
    conflicts: List[JsonDict]
    notes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class PublishGateResult:
    """发布门决策，用于判断版本是否可以正式发布。"""

    quality_status: str
    publishable: bool
    blocking_reasons: List[str]
    warning_reasons: List[str]


@dataclass
class VersionBuildRows:
    """一次批量构建中物化后的输入行。"""

    recommendations: List[JsonDict]
    grades: List[JsonDict]
    picos: List[JsonDict]
    evidence_items: List[JsonDict]
    existing_versions: List[JsonDict]


@dataclass
class VersionBuildOutputs:
    """版本行以及逐版本审计/报告行。"""

    versions: List[JsonDict]
    report_items: List[JsonDict]


def build_notes(context: VersionBuildContext) -> List[str]:
    """为缺失或较弱的关联上下文生成审计备注。"""

    linked_grades = context.linked_grades
    accepted_grade = context.accepted_grade
    conflicts = context.conflicts
    linked_pico = context.linked_pico
    pico_match = context.pico_match
    evidence_rows = context.evidence_rows
    notes: List[str] = []
    if not linked_grades:
        notes.append("no_linked_grade_candidate")
    elif not accepted_grade:
        notes.append("linked_grade_candidates_not_accepted")
    if conflicts:
        notes.append("recommendation_grade_strength_or_certainty_conflict")
    if not linked_pico:
        notes.append("no_linked_pico")
    elif float(pico_match.get("match_score") or 0.0) < STRONG_PICO_MATCH_SCORE:
        notes.append("weak_pico_match")
    if not evidence_rows:
        notes.append("no_linked_evidence")
    return notes


def _has_source_span_coordinates(rec: JsonDict) -> bool:
    if not (rec.get("source_span") or rec.get("source_text")):
        return False
    has_raw = rec.get("raw_start_char") is not None and rec.get("raw_end_char") is not None
    has_legacy = rec.get("start_char") is not None and rec.get("end_char") is not None
    return has_raw or has_legacy


def _validation_is_valid(rec: JsonDict) -> bool:
    validation = rec.get("validation")
    if isinstance(validation, dict) and validation.get("is_valid") is False:
        return False
    return True


def _auto_qc_blocks_publish(rec: JsonDict) -> bool:
    qc = auto_qc(rec)
    decision = str(qc.get("decision") or qc.get("auto_qc_decision") or "").lower()
    status = str(qc.get("status") or "").lower()
    return decision in {"reject", "rejected", "failed"} or status in {"failed", "error"}


def default_pico_match() -> JsonDict:
    """返回未提供 PICO 匹配信息时使用的标准空匹配对象。"""

    return {"association_reason": "not_provided", "match_score": 0.0, "match_features": {}}


def evaluate_publish_gate(context: VersionBuildContext) -> PublishGateResult:
    """把版本分类为可发布、需复核或阻断。"""

    blocking: List[str] = []
    warnings: List[str] = []
    rec = context.rec
    if rec.get("status") != "accepted":
        blocking.append("recommendation_candidate_not_accepted")
    if not _has_source_span_coordinates(rec):
        blocking.append("missing_source_span_coordinates")
    if not _validation_is_valid(rec):
        blocking.append("recommendation_validation_failed")
    if context.conflicts:
        blocking.append("recommendation_grade_strength_or_certainty_conflict")
    if _auto_qc_blocks_publish(rec):
        blocking.append("recommendation_auto_qc_failed")
    if not context.accepted_grade:
        warnings.append("no_accepted_grade")
    if not context.linked_pico:
        warnings.append("no_linked_pico")
    elif float(context.pico_match.get("match_score") or 0.0) < STRONG_PICO_MATCH_SCORE:
        warnings.append("weak_pico_match")
    if not context.evidence_rows:
        warnings.append("no_linked_evidence")

    if blocking:
        return PublishGateResult("blocked", False, blocking, warnings)
    if warnings:
        return PublishGateResult("needs_review", False, blocking, warnings)
    return PublishGateResult("publishable", True, blocking, warnings)


def build_context(inputs: VersionBuildInput) -> VersionBuildContext:
    """解析 accepted grade、指南画像、证据摘要和字段冲突。"""

    accepted_grade = best_accepted_grade(inputs.linked_grades)
    context = VersionBuildContext(
        rec=inputs.rec,
        linked_grades=inputs.linked_grades,
        accepted_grade=accepted_grade,
        linked_pico=inputs.linked_pico,
        pico_match=inputs.pico_match or default_pico_match(),
        evidence_rows=inputs.evidence_rows,
        evidence_reason=inputs.evidence_reason,
        evidence_summary=summarize_evidence(inputs.evidence_rows, inputs.evidence_reason),
        profile=profile_context(inputs.rec),
        conflicts=field_conflicts(inputs.rec, accepted_grade),
    )
    context.notes = build_notes(context)
    return context


def build_version_payloads(context: VersionBuildContext) -> tuple[JsonDict, JsonDict]:
    """构造保留来源血缘的 normalized/raw payload。"""

    rec = context.rec
    gate = evaluate_publish_gate(context)
    normalized = {
        "builder_version": BUILDER_VERSION,
        "publish_gate": {
            "quality_status": gate.quality_status,
            "publishable": gate.publishable,
            "blocking_reasons": gate.blocking_reasons,
            "warning_reasons": gate.warning_reasons,
        },
        "profile_context": context.profile,
        "recommendation_auto_qc": auto_qc(rec),
        "recommendation_enhancement": enhancement(rec),
        "linked_grade_candidates": [compact_grade_link(grade) for grade in context.linked_grades],
        "accepted_grade_candidate": compact_grade_link(context.accepted_grade) if context.accepted_grade else None,
        "linked_pico": compact_pico(context.linked_pico, context.pico_match),
        "linked_evidence": context.evidence_summary,
        "field_conflicts": context.conflicts,
        "builder_notes": context.notes,
        "source_offsets": {
            "clean_start_char": rec.get("clean_start_char"),
            "clean_end_char": rec.get("clean_end_char"),
            "raw_start_char": rec.get("raw_start_char"),
            "raw_end_char": rec.get("raw_end_char"),
            "legacy_start_char": rec.get("start_char"),
            "legacy_end_char": rec.get("end_char"),
            "offset_mapping": rec.get("offset_mapping", {}),
        },
        "source_metadata": payload(rec).get("source_metadata", {}),
    }
    raw = {
        "builder_version": BUILDER_VERSION,
        "recommendation_candidate_snapshot": snapshot(rec, keep_text=False),
        "accepted_grade_snapshot": snapshot(context.accepted_grade, keep_text=False) if context.accepted_grade else None,
        "linked_grade_count": len(context.linked_grades),
        "linked_pico_id": context.linked_pico.get("pico_id") if context.linked_pico else None,
        "linked_evidence_count": len(context.evidence_rows),
    }
    return normalized, raw


def build_report_item(version: RecommendationVersion, context: VersionBuildContext) -> JsonDict:
    gate = normalized_payload_gate(version.normalized_payload)
    return {
        "recommendation_version_id": version.recommendation_version_id,
        "recommendation_candidate_id": context.rec.get("candidate_id"),
        "grade_candidate_id": version.grade_candidate_id,
        "record_id": version.record_id,
        "guideline_id": version.guideline_id,
        "issuer": version.issuer,
        "grading_system": version.grading_system,
        "pico_id": version.pico_id,
        "pico_association_reason": context.pico_match.get("association_reason"),
        "pico_match_score": context.pico_match.get("match_score", 0.0),
        "pico_match_features": context.pico_match.get("match_features", {}),
        "linked_evidence_count": len(context.evidence_rows),
        "evidence_association_reason": context.evidence_reason,
        "has_accepted_grade": context.accepted_grade is not None,
        "linked_grade_count": len(context.linked_grades),
        "field_conflicts": context.conflicts,
        "builder_notes": context.notes,
        "quality_status": version.quality_status,
        "publishable": gate.get("publishable", False),
        "blocking_reasons": gate.get("blocking_reasons", []),
        "warning_reasons": gate.get("warning_reasons", []),
    }


def normalized_payload_gate(normalized_payload: JsonDict) -> JsonDict:
    gate = normalized_payload.get("publish_gate") if isinstance(normalized_payload, dict) else {}
    return gate if isinstance(gate, dict) else {}


def recommendation_id_for(rec: JsonDict) -> str:
    code = str(rec.get("recommendation_code") or "").strip()
    generic_codes = {"for", "against", "neutral", "strong", "conditional", "weak", "unclear", "none"}
    identity = code if code and code.lower() not in generic_codes else rec.get("candidate_id")
    return str(rec.get("recommendation_id") or stable_id(
        "recommendation",
        rec.get("guideline_id"),
        identity,
    ))


def version_change(rec: JsonDict, previous_version: Optional[JsonDict]) -> tuple[str, Optional[str], str]:
    previous_number = _as_int(str((previous_version or {}).get("version_number") or "v0").lstrip("vV"))
    version_number = f"v{previous_number + 1}"
    changed_fields = [
        field
        for field in ("recommendation_text", "direction", "strength", "certainty")
        if previous_version and (previous_version.get(field) or "") != (rec.get(field) or "")
    ]
    change_type = "new" if not previous_version else ("modified" if changed_fields else "unchanged")
    change_summary = ", ".join(f"{field} changed" for field in changed_fields) or None
    return version_number, change_summary, change_type


def source_span_fields(rec: JsonDict) -> JsonDict:
    rec_payload = payload(rec)
    raw_start = rec.get("raw_start_char")
    raw_end = rec.get("raw_end_char")
    return {
        "source_span": rec.get("source_span") or rec.get("source_text"),
        "source_span_ref": rec.get("source_span_ref") or rec.get("source_block_id") or rec_payload.get("block_id"),
        "start_char": raw_start if raw_start is not None else rec.get("start_char"),
        "end_char": raw_end if raw_end is not None else rec.get("end_char"),
    }


def recommendation_version_fields(
    context: VersionBuildContext,
    normalized_payload: JsonDict,
    raw_payload: JsonDict,
    previous_version: Optional[JsonDict],
) -> JsonDict:
    rec = context.rec
    accepted_grade = context.accepted_grade
    linked_pico = context.linked_pico
    gate = normalized_payload_gate(normalized_payload)
    recommendation_id = recommendation_id_for(rec)
    version_number, change_summary, change_type = version_change(rec, previous_version)
    return {
        "recommendation_version_id": stable_id("recommendation_version", recommendation_id, version_number),
        "recommendation_candidate_id": str(rec.get("candidate_id") or ""),
        "guideline_id": str(rec.get("guideline_id") or ""),
        "version_number": version_number,
        "recommendation_text": str(rec.get("recommendation_text") or ""),
        "recommendation_id": recommendation_id,
        "previous_version_id": (previous_version or {}).get("recommendation_version_id"),
        "pico_id": str(linked_pico.get("pico_id") or "") if linked_pico else None,
        "record_id": str(rec.get("record_id") or "") or None,
        "grade_candidate_id": accepted_grade.get("grade_candidate_id") if accepted_grade else None,
        "profile_id": context.profile.get("profile_id"),
        "issuer": context.profile.get("issuer"),
        "grading_system": context.profile.get("grading_system"),
        "quality_status": str(gate.get("quality_status") or "needs_review"),
        "recommendation_code": rec.get("recommendation_code"),
        "direction": rec.get("direction") or "unclear",
        "strength": rec.get("strength") or "unclear",
        "certainty": rec.get("certainty") or "unclear",
        "rationale": rec.get("rationale"),
        "remarks": rec.get("remarks"),
        "population": rec.get("population"),
        "intervention": rec.get("intervention"),
        "comparator": rec.get("comparator"),
        "outcome_summary": summarize_outcomes(rec.get("outcomes")),
        "source_text": rec.get("source_text"),
        **source_span_fields(rec),
        "source_section": rec.get("source_section"),
        "source_url": rec.get("source_url"),
        "change_type": change_type,
        "change_summary": change_summary,
        "created_at": utc_now(),
        "published_at": None,
        "normalized_payload": normalized_payload,
        "raw_payload": raw_payload,
    }


def make_recommendation_version(
    context: VersionBuildContext,
    normalized_payload: JsonDict,
    raw_payload: JsonDict,
    previous_version: Optional[JsonDict] = None,
) -> RecommendationVersion:
    """构造 RecommendationVersion 领域对象。"""

    fields = recommendation_version_fields(context, normalized_payload, raw_payload, previous_version)
    return RecommendationVersion(**fields)


def build_version(inputs: VersionBuildInput) -> tuple[RecommendationVersion, JsonDict]:
    """构造一条 RecommendationVersion 及对应报告项。"""

    context = build_context(inputs)
    normalized_payload, raw_payload = build_version_payloads(context)

    version = make_recommendation_version(context, normalized_payload, raw_payload, inputs.previous_version)
    report_item = build_report_item(version, context)
    return version, report_item


def collect_build_rows(
    recommendations: Iterable[JsonDict],
    grades: Iterable[JsonDict],
    picos: Iterable[JsonDict],
    evidence_items: Iterable[JsonDict],
    existing_versions: Iterable[JsonDict],
) -> VersionBuildRows:
    """一次性物化输入迭代器，确保索引和报告使用同一批行。"""

    return VersionBuildRows(
        recommendations=list(recommendations),
        grades=list(grades),
        picos=list(picos),
        evidence_items=list(evidence_items),
        existing_versions=list(existing_versions),
    )


def latest_versions_by_recommendation(existing_versions: Iterable[JsonDict]) -> Dict[str, JsonDict]:
    """按 recommendation_id 返回最新的既有版本。"""

    previous_by_recommendation: Dict[str, JsonDict] = {}
    for existing in existing_versions:
        recommendation_id = str(existing.get("recommendation_id") or "")
        if not recommendation_id:
            continue
        current = previous_by_recommendation.get(recommendation_id)
        if current is None or _as_int(str(existing.get("version_number") or "").lstrip("vV")) > _as_int(
            str(current.get("version_number") or "").lstrip("vV")
        ):
            previous_by_recommendation[recommendation_id] = existing
    return previous_by_recommendation


def previous_version_for(rec: JsonDict, previous_by_recommendation: Dict[str, JsonDict]) -> Optional[JsonDict]:
    """为推荐候选查找上一版 RecommendationVersion。"""

    return previous_by_recommendation.get(recommendation_id_for(rec))


def build_version_rows(rows: VersionBuildRows) -> VersionBuildOutputs:
    """只为 accepted 推荐候选构建版本行。"""

    previous_by_recommendation = latest_versions_by_recommendation(rows.existing_versions)
    grades_by_rec = index_grades_by_recommendation(rows.grades)
    pico_index = PicoIndex(rows.picos)
    evidence_index = EvidenceIndex(rows.evidence_items)
    versions: List[JsonDict] = []
    report_items: List[JsonDict] = []

    for rec in rows.recommendations:
        if rec.get("status") != "accepted":
            continue
        linked_pico, pico_match = pico_index.find(rec)
        linked_evidence, evidence_reason = evidence_index.find(rec, str(linked_pico.get("pico_id") or "") if linked_pico else None)
        version_input = VersionBuildInput(
            rec=rec,
            linked_grades=grades_by_rec.get(str(rec.get("candidate_id") or ""), []),
            previous_version=previous_version_for(rec, previous_by_recommendation),
            linked_pico=linked_pico,
            pico_match=pico_match,
            evidence_rows=linked_evidence,
            evidence_reason=evidence_reason,
        )
        version, report_item = build_version(version_input)
        versions.append(version.to_dict())
        report_items.append(report_item)
    return VersionBuildOutputs(versions=versions, report_items=report_items)


def report_gap_counts(report_items: List[JsonDict]) -> JsonDict:
    """统计版本报告中缺失上下文或存在冲突的数量。"""

    return {
        "conflict_count": sum(1 for item in report_items if item["field_conflicts"]),
        "without_grade": sum(1 for item in report_items if not item["has_accepted_grade"]),
        "without_pico": sum(1 for item in report_items if not item.get("pico_id")),
        "without_evidence": sum(1 for item in report_items if not item.get("linked_evidence_count")),
        "evidence_total": sum(int(item.get("linked_evidence_count") or 0) for item in report_items),
    }


def report_quality_counters(report_items: List[JsonDict]) -> tuple[Counter[str], Counter[str], Counter[str]]:
    """统计版本质量状态、阻断原因和警告原因。"""

    quality_status_counts = Counter(str(item.get("quality_status") or "unknown") for item in report_items)
    blocking_reason_counts = Counter(
        reason
        for item in report_items
        for reason in item.get("blocking_reasons", [])
    )
    warning_reason_counts = Counter(
        reason
        for item in report_items
        for reason in item.get("warning_reasons", [])
    )
    return quality_status_counts, blocking_reason_counts, warning_reason_counts


def build_versions_report(rows: VersionBuildRows, outputs: VersionBuildOutputs) -> JsonDict:
    """汇总版本生成结果和关联上下文缺口。"""

    gaps = report_gap_counts(outputs.report_items)
    missing_profile = sum(1 for item in outputs.report_items if not item.get("issuer") or not item.get("grading_system"))
    status_counts = Counter(str(row.get("status") or "unknown") for row in rows.recommendations)
    grade_status_counts = Counter(str(row.get("status") or "unknown") for row in rows.grades)
    issuer_counts = Counter(str(row.get("issuer") or "unknown") for row in outputs.versions)
    grading_counts = Counter(str(row.get("grading_system") or "unknown") for row in outputs.versions)
    version_count = len(outputs.versions)
    quality_status_counts, blocking_reason_counts, warning_reason_counts = report_quality_counters(outputs.report_items)
    report = {
        "builder_version": BUILDER_VERSION,
        "input_recommendations": len(rows.recommendations),
        "input_grades": len(rows.grades),
        "input_picos": len(rows.picos),
        "input_evidence_items": len(rows.evidence_items),
        "recommendation_status_counts": dict(status_counts),
        "grade_status_counts": dict(grade_status_counts),
        "accepted_recommendations": status_counts.get("accepted", 0),
        "accepted_grades": grade_status_counts.get("accepted", 0),
        "generated_versions": version_count,
        "publishable_versions": quality_status_counts.get("publishable", 0),
        "needs_review_versions": quality_status_counts.get("needs_review", 0),
        "blocked_versions": quality_status_counts.get("blocked", 0),
        "quality_status_counts": dict(quality_status_counts),
        "blocking_reason_counts": dict(blocking_reason_counts),
        "warning_reason_counts": dict(warning_reason_counts),
        "accepted_recommendations_without_accepted_grade": gaps["without_grade"],
        "accepted_recommendations_with_accepted_grade": version_count - gaps["without_grade"],
        "versions_without_pico": gaps["without_pico"],
        "versions_with_pico": version_count - gaps["without_pico"],
        "versions_without_evidence": gaps["without_evidence"],
        "versions_with_evidence": version_count - gaps["without_evidence"],
        "avg_evidence_per_version": round(gaps["evidence_total"] / len(outputs.report_items), 2) if outputs.report_items else 0,
        "recommendation_grade_conflicts": gaps["conflict_count"],
        "missing_profile_context": missing_profile,
        "issuer_counts": dict(issuer_counts),
        "grading_system_counts": dict(grading_counts),
        "version_items": outputs.report_items,
        "created_at": utc_now(),
    }
    return report


def build_versions(
    recommendations: Iterable[JsonDict],
    grades: Iterable[JsonDict],
    picos: Iterable[JsonDict] = (),
    evidence_items: Iterable[JsonDict] = (),
    existing_versions: Iterable[JsonDict] = (),
) -> tuple[List[JsonDict], JsonDict]:
    """基于内存迭代器构建版本行和批量报告。"""

    rows = collect_build_rows(recommendations, grades, picos, evidence_items, existing_versions)
    outputs = build_version_rows(rows)
    report = build_versions_report(rows, outputs)
    return outputs.versions, report


def build_versions_file(
    recommendations_input: str | Path,
    grades_input: str | Path,
    versions_output: str | Path,
    report_output: str | Path,
    picos_input: str | Path | None = None,
    evidence_input: str | Path | None = None,
) -> JsonDict:
    """根据流水线输出构建 RecommendationVersion JSONL 和报告文件。"""

    versions, report = build_versions(
        iter_jsonl(recommendations_input),
        iter_jsonl(grades_input),
        picos=iter_jsonl(picos_input) if picos_input else (),
        evidence_items=iter_jsonl(evidence_input) if evidence_input else (),
    )
    report["recommendations_input"] = str(recommendations_input)
    report["grades_input"] = str(grades_input)
    report["picos_input"] = str(picos_input or "")
    report["evidence_input"] = str(evidence_input or "")
    write_jsonl(versions_output, versions)
    write_jsonl(report_output, [report])
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RecommendationVersion records from enhanced candidates.")
    parser.add_argument("--recommendations-input", required=True)
    parser.add_argument("--grades-input", required=True)
    parser.add_argument("--picos-input", default=None)
    parser.add_argument("--evidence-input", default=None)
    parser.add_argument("--versions-output", required=True)
    parser.add_argument("--report-output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_versions_file(
        recommendations_input=args.recommendations_input,
        grades_input=args.grades_input,
        picos_input=args.picos_input,
        evidence_input=args.evidence_input,
        versions_output=args.versions_output,
        report_output=args.report_output,
    )
    print(
        "generated_versions={generated_versions} with_grade={accepted_recommendations_with_accepted_grade} with_pico={versions_with_pico} with_evidence={versions_with_evidence} conflicts={recommendation_grade_conflicts}".format(
            **report
        )
    )


if __name__ == "__main__":
    main()
