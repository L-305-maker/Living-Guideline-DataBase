"""GRADE 抽取文件：识别证据确定性、推荐强度和 GRADE 相关候选信息，并关联推荐候选。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.domain.candidate import GradeCandidate, ModelTrace
from src.domain.common import stable_id
from src.common.extraction_common import normalize_text, source_metadata, source_section, source_url, utc_now
from src.common.process_jsonl import iter_jsonl, write_jsonl

# 模块职责:
# 用规则抽取 GRADE 候选，并把它们关联到推荐候选。
# 同一 block 的匹配最可靠；同章节附近匹配会保留为可复核候选，而不是静默丢弃。


JsonDict = Dict[str, Any]

RULE_VERSION = "grade_rules_v1"
MAX_GRADE_RECOMMENDATION_DISTANCE = 12

GRADE_SYSTEM_PATTERNS = [
    ("GRADE", re.compile(r"\bGRADE\b|Grading of Recommendations", re.I)),
    ("COR_LOE", re.compile(r"\b(?:class|cor)\s+(?:i|ii|iii|iia|iib)\b|\b(?:loe|level of evidence)\b", re.I)),
    ("NUMERIC_LETTER_GRADE", re.compile(r"\bgrade\s+[1-2][A-D]\b|\b[1-2][A-D]\b", re.I)),
    ("LETTER_GRADE", re.compile(r"\bgrade\s+[A-D]\b|I statement", re.I)),
    ("VERB_BASED", re.compile(r"\b(?:offer|consider|recommend|suggest|should|should not)\b", re.I)),
    ("CHINESE_GRADE", re.compile(r"(证据等级|证据级别|推荐等级|推荐强度|专家共识|A级|B级|C级|D级|Ⅰ级|Ⅱ级|Ⅲ级|强推荐|弱推荐)")),
    ("CHINESE_VERB_BASED", re.compile(r"(推荐|建议|应当|应|应该|宜|可考虑|不推荐|不建议|避免|禁用)")),
]
CERTAINTY_PATTERNS = [
    ("very_low", re.compile(r"\b(very[-\s]+low\s+(?:certainty|quality|evidence)|(?:certainty|quality)(?:\s+of\s+evidence)?\s+(?:was|is|were|are)\s+very[-\s]+low)\b", re.I)),
    ("low", re.compile(r"\b(low\s+(?:certainty|quality|evidence)|(?:certainty|quality)(?:\s+of\s+evidence)?\s+(?:was|is|were|are)\s+low)\b", re.I)),
    ("moderate", re.compile(r"\b(moderate\s+(?:certainty|quality|evidence)|(?:certainty|quality)(?:\s+of\s+evidence)?\s+(?:was|is|were|are)\s+moderate)\b", re.I)),
    ("high", re.compile(r"\b(high\s+(?:certainty|quality|evidence)|(?:certainty|quality)(?:\s+of\s+evidence)?\s+(?:was|is|were|are)\s+high)\b", re.I)),
    ("very_low", re.compile(r"(极低质量证据|证据质量极低|极低证据质量|极低级别证据)")),
    ("low", re.compile(r"(低质量证据|证据质量低|低级别证据)")),
    ("moderate", re.compile(r"(中等质量证据|中等证据质量|中级别证据|中等强度证据)")),
    ("high", re.compile(r"(高质量证据|证据质量高|高级别证据|高强度证据)")),
]
STRENGTH_PATTERNS = [
    ("strong", re.compile(r"\bstrong recommendation\b|\bwe recommend\b|\bstandard\b|\bclass\s+i\b|\bcor\s+i\b", re.I)),
    ("conditional", re.compile(r"\bconditional recommendation\b|\bwe suggest\b|\bsuggest\b|\bclass\s+iia\b|\bcor\s+iia\b", re.I)),
    ("weak", re.compile(r"\bweak recommendation\b|\boption\b|\bclass\s+iib\b|\bcor\s+iib\b", re.I)),
    ("strong", re.compile(r"(强推荐|A级推荐|Ⅰ级推荐|应当|必须|首选|优先推荐)")),
    ("conditional", re.compile(r"(条件推荐|中等推荐|B级推荐|Ⅱ级推荐|建议|可考虑|可以考虑)")),
    ("weak", re.compile(r"(弱推荐|C级推荐|D级推荐|Ⅲ级推荐|可选择)")),
]
DOWNGRADE_RE = re.compile(r"\b(risk of bias|inconsistency|indirectness|imprecision|publication bias)\b|(偏倚风险|不一致|间接性|不精确|发表偏倚)", re.I)


@dataclass(frozen=True)
class AssociationMatch:
    """GRADE 与推荐候选的关联结果。

    ``__iter__`` 兼容旧代码中的 ``candidate_id, reason = index.find(block)`` 写法；新增字段
    用于下游区分强关联、可接受近邻关联、弱关联和缺失关联。
    """

    candidate_id: str
    reason: str
    quality: str
    distance: int | None = None

    def __iter__(self):
        yield self.candidate_id
        yield self.reason


def block_id_from_candidate(candidate: JsonDict) -> str:
    """读取推荐候选 payload 中保存的来源 block ID。"""

    payload = candidate.get("normalized_payload")
    if isinstance(payload, dict):
        return str(payload.get("block_id") or "")
    return ""


def order_from_candidate(candidate: JsonDict) -> int:
    """读取用于近邻关联的 source order。"""

    payload = candidate.get("normalized_payload")
    if isinstance(payload, dict):
        try:
            return int(payload.get("source_order") or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def section_tokens(value: object) -> set[str]:
    """把章节文本切成 token，用于宽松判断是否同章节。"""

    text = normalize_text(value)
    return {
        token
        for token in re.findall(r"[a-zA-Z]{4,}", text.lower())
        if token
        not in {
            "demo",
            "document",
            "evidence",
            "guideline",
            "recommendation",
            "recommendations",
        }
    }


def section_overlap(block: JsonDict, candidate: JsonDict) -> bool:
    """判断 grade block 与推荐候选是否共享章节 token。"""

    block_tokens = section_tokens(source_section(block))
    candidate_tokens = section_tokens(candidate.get("source_section") or "")
    return bool(block_tokens and candidate_tokens and block_tokens.intersection(candidate_tokens))


def infer_grade_system(text: str) -> str:
    """推断文本正在表达哪一种推荐分级体系。"""

    for system, pattern in GRADE_SYSTEM_PATTERNS:
        if pattern.search(text):
            return system
    return "unknown"


def infer_certainty(text: str) -> str:
    """从文本中推断 GRADE 证据确定性/质量等级。"""

    for certainty, pattern in CERTAINTY_PATTERNS:
        if pattern.search(text):
            return certainty
    return "unclear"


def infer_strength(text: str) -> str:
    """根据 GRADE、COR 或动词线索推断推荐强度。"""

    for strength, pattern in STRENGTH_PATTERNS:
        if pattern.search(text):
            return strength
    return "unclear"


def downgrade_reasons(text: str) -> List[str]:
    """提取理由文本中提到的 GRADE 降级领域。"""

    seen: List[str] = []
    zh_map = {
        "偏倚风险": "risk_of_bias",
        "不一致": "inconsistency",
        "间接性": "indirectness",
        "不精确": "imprecision",
        "发表偏倚": "publication_bias",
    }
    for match in DOWNGRADE_RE.finditer(text):
        matched = match.group(1) or match.group(2) or ""
        reason = zh_map.get(matched, matched.lower().replace(" ", "_"))
        if reason not in seen:
            seen.append(reason)
    return seen


def grade_domain_fields(reasons: List[str]) -> JsonDict:
    """把降级原因名称转换为标准化领域字段。"""

    fields = {
        "risk_of_bias": "not_reported",
        "inconsistency": "not_reported",
        "indirectness": "not_reported",
        "imprecision": "not_reported",
        "publication_bias": "not_reported",
    }
    for reason in reasons:
        if reason in fields:
            fields[reason] = "serious_or_concern"
    return fields


def extraction_confidence(text: str, recommendation_candidate_id: str, association_quality: str = "missing") -> float:
    """根据分级信号和推荐关联情况计算置信度。"""

    score = 0.2
    if infer_grade_system(text) != "unknown":
        score += 0.2
    if infer_certainty(text) != "unclear":
        score += 0.25
    if infer_strength(text) != "unclear":
        score += 0.2
    if recommendation_candidate_id and association_quality == "strong":
        score += 0.1
    elif recommendation_candidate_id and association_quality == "medium":
        score += 0.07
    elif recommendation_candidate_id and association_quality == "weak":
        score += 0.03
    if downgrade_reasons(text):
        score += 0.05
    return round(min(score, 0.95), 4)


class RecommendationIndex:
    """为 grade 到 recommendation 的关联建立推荐候选索引。"""

    def __init__(self, candidates: Iterable[JsonDict]) -> None:
        self.by_block: Dict[str, List[JsonDict]] = {}
        self.by_record: Dict[str, List[JsonDict]] = {}
        for candidate in candidates:
            block_id = block_id_from_candidate(candidate)
            record_id = str(candidate.get("record_id") or "")
            if block_id:
                self.by_block.setdefault(block_id, []).append(candidate)
            if record_id:
                self.by_record.setdefault(record_id, []).append(candidate)
        for items in self.by_record.values():
            items.sort(key=order_from_candidate)

    def find(self, block: JsonDict) -> AssociationMatch:
        """为单个 grade block 寻找最合适的推荐候选。"""

        block_id = str(block.get("block_id") or "")
        same_block = self.by_block.get(block_id) or []
        if same_block:
            return AssociationMatch(str(same_block[0].get("candidate_id") or ""), "same_block", "strong", 0)

        record_id = str(block.get("record_id") or "")
        candidates = self.by_record.get(record_id) or []
        if not candidates:
            return AssociationMatch("", "not_found", "missing")

        try:
            block_order = int(block.get("order") or 0)
        except (TypeError, ValueError):
            block_order = 0
        nearest = min(candidates, key=lambda item: abs(order_from_candidate(item) - block_order))
        distance = abs(order_from_candidate(nearest) - block_order)
        if distance <= 5:
            quality = "strong" if distance <= 2 else "medium"
            return AssociationMatch(str(nearest.get("candidate_id") or ""), f"nearest_order_{distance}", quality, distance)
        if distance <= MAX_GRADE_RECOMMENDATION_DISTANCE and section_overlap(block, nearest):
            quality = "medium" if distance <= 8 else "weak"
            return AssociationMatch(
                str(nearest.get("candidate_id") or ""),
                f"nearest_same_section_order_{distance}",
                quality,
                distance,
            )
        return AssociationMatch("", "not_found", "missing")


def build_trace(block: JsonDict, parsed: JsonDict, elapsed_ms: int, success: bool = True, error: Optional[str] = None) -> ModelTrace:
    """为一次 GRADE 规则抽取生成 ModelTrace。"""

    trace_id = stable_id("model_trace", RULE_VERSION, block.get("block_id"), "grade_extraction")
    return ModelTrace(
        model_trace_id=trace_id,
        task_type="grade_extraction",
        method="rule",
        model_name=RULE_VERSION,
        input_entity_type="block",
        input_entity_id=str(block.get("block_id") or ""),
        input_text=str(block.get("text") or ""),
        model_version=RULE_VERSION,
        raw_output={"candidate_hints": block.get("candidate_hints", [])},
        parsed_output=parsed,
        confidence=float(parsed.get("confidence") or 0.0),
        parameters={
            "association": "same_block_then_nearest_order_or_same_section",
            "max_same_section_order_distance": MAX_GRADE_RECOMMENDATION_DISTANCE,
        },
        runtime_ms=elapsed_ms,
        success=success,
        error_message=error,
        target_table="grade_candidates",
        created_at=utc_now(),
    )


def _grade_candidate_fields(
    block: JsonDict,
    recommendation_candidate_id: str,
    association_reason: str,
    association_quality: str,
) -> JsonDict:
    """在构造 GradeCandidate 前推断标准化字段值。"""

    text = normalize_text(block.get("text"))
    certainty = infer_certainty(text)
    strength = infer_strength(text)
    reasons = downgrade_reasons(text)
    domains = grade_domain_fields(reasons)
    has_grade_signal = certainty != "unclear" or strength != "unclear"
    link_is_usable = association_quality in {"strong", "medium"}
    status = "pending" if recommendation_candidate_id and link_is_usable and has_grade_signal else "needs_review"
    review_note = None
    if status == "needs_review":
        review_note = (
            f"Needs review: association={association_reason}, association_quality={association_quality}, "
            f"certainty={certainty}, strength={strength}."
        )
    return {
        "text": text,
        "certainty": certainty,
        "strength": strength,
        "grade_system": infer_grade_system(text),
        "reasons": reasons,
        "domains": domains,
        "confidence": extraction_confidence(text, recommendation_candidate_id, association_quality),
        "status": status,
        "review_note": review_note,
    }


def _grade_candidate_payloads(
    block: JsonDict,
    association_reason: str,
    association_quality: str,
    association_distance: int | None,
) -> tuple[JsonDict, JsonDict]:
    """构造用于追溯和后续 QC 的 normalized/raw payload。"""

    normalized = {
        "block_id": block.get("block_id", ""),
        "source_order": block.get("order", 0),
        "association_reason": association_reason,
        "association_quality": association_quality,
        "association_distance": association_distance,
        "source_metadata": source_metadata(block),
        "route": block.get("route", {}),
    }
    raw = {
        "rule_version": RULE_VERSION,
        "block_id": block.get("block_id", ""),
        "candidate_hints": block.get("candidate_hints", []),
        "quality": block.get("quality", {}),
    }
    return normalized, raw


def build_grade_candidate(
    block: JsonDict,
    trace_id: str,
    recommendation_candidate_id: str,
    association_reason: str,
    association_quality: str = "missing",
    association_distance: int | None = None,
) -> GradeCandidate:
    """从已路由 grade block 构造 GradeCandidate 领域对象。"""

    fields = _grade_candidate_fields(block, recommendation_candidate_id, association_reason, association_quality)
    text = fields["text"]
    domains = fields["domains"]
    normalized_payload, raw_payload = _grade_candidate_payloads(
        block,
        association_reason,
        association_quality,
        association_distance,
    )
    grade_candidate_id = stable_id("grade_candidate", block.get("block_id"), recommendation_candidate_id, text[:120])
    return GradeCandidate(
        grade_candidate_id=grade_candidate_id,
        recommendation_candidate_id=recommendation_candidate_id,
        record_id=str(block.get("record_id") or ""),
        guideline_id=str(block.get("guideline_id") or "") or None,
        model_trace_id=trace_id,
        grade_system=fields["grade_system"],
        certainty=fields["certainty"],
        strength=fields["strength"],
        risk_of_bias=domains["risk_of_bias"],
        inconsistency=domains["inconsistency"],
        indirectness=domains["indirectness"],
        imprecision=domains["imprecision"],
        publication_bias=domains["publication_bias"],
        judgement_rationale=text,
        source_span=text,
        reasons_for_downgrade=fields["reasons"],
        source_text=text,
        source_section=source_section(block),
        source_url=source_url(block),
        extraction_method="rule",
        extraction_confidence=fields["confidence"],
        status=fields["status"],
        review_note=fields["review_note"],
        normalized_payload=normalized_payload,
        raw_payload=raw_payload,
        created_at=utc_now(),
        updated_at=utc_now(),
    )


def extract_from_block(block: JsonDict, rec_index: RecommendationIndex) -> Tuple[GradeCandidate, ModelTrace]:
    """从一个 grade block 抽取一条 GradeCandidate 及其 trace。"""

    start = time.perf_counter()
    association = rec_index.find(block)
    recommendation_candidate_id = association.candidate_id
    association_reason = association.reason
    text = normalize_text(block.get("text"))
    parsed = {
        "block_id": block.get("block_id", ""),
        "recommendation_candidate_id": recommendation_candidate_id,
        "association_reason": association_reason,
        "association_quality": association.quality,
        "association_distance": association.distance,
        "grade_system": infer_grade_system(text),
        "certainty": infer_certainty(text),
        "strength": infer_strength(text),
        "reasons_for_downgrade": downgrade_reasons(text),
    }
    parsed.update(grade_domain_fields(parsed["reasons_for_downgrade"]))
    parsed["confidence"] = extraction_confidence(text, recommendation_candidate_id, association.quality)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    trace = build_trace(block, parsed, elapsed_ms)
    candidate = build_grade_candidate(
        block,
        trace.model_trace_id,
        recommendation_candidate_id,
        association_reason,
        association.quality,
        association.distance,
    )
    return candidate, trace


def extract_grade_candidates(blocks: Iterable[JsonDict], recommendation_candidates: Iterable[JsonDict]) -> Tuple[List[JsonDict], List[JsonDict]]:
    """从已路由 grade block 批量抽取候选和 trace。"""

    rec_index = RecommendationIndex(recommendation_candidates)
    grade_candidates: List[JsonDict] = []
    traces: List[JsonDict] = []
    for block in blocks:
        try:
            candidate, trace = extract_from_block(block, rec_index)
        except Exception as exc:
            elapsed_ms = 0
            parsed = {"block_id": block.get("block_id", ""), "confidence": 0.0}
            trace = build_trace(block, parsed, elapsed_ms, success=False, error=str(exc))
            traces.append(trace.to_dict())
            continue
        grade_candidates.append(candidate.to_dict())
        traces.append(trace.to_dict())
    return grade_candidates, traces


def extract_file(
    grade_blocks_input: str | Path,
    recommendation_candidates_input: str | Path,
    grade_candidates_output: str | Path,
    traces_output: str | Path,
) -> JsonDict:
    """从流水线脚本使用的 JSONL 文件中抽取 GradeCandidate。"""

    candidates, traces = extract_grade_candidates(iter_jsonl(grade_blocks_input), iter_jsonl(recommendation_candidates_input))
    write_jsonl(grade_candidates_output, candidates)
    write_jsonl(traces_output, traces)
    return {
        "grade_blocks_input": str(grade_blocks_input),
        "recommendation_candidates_input": str(recommendation_candidates_input),
        "grade_candidates": len(candidates),
        "model_traces": len(traces),
        "successful_traces": sum(1 for trace in traces if trace.get("success")),
        "failed_traces": sum(1 for trace in traces if not trace.get("success")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract rule-based GradeCandidate JSONL from routed grade blocks.")
    parser.add_argument("--grade-blocks-input", required=True, help="Input *_grade_blocks.jsonl path.")
    parser.add_argument("--recommendation-candidates-input", required=True, help="Input RecommendationCandidate JSONL path.")
    parser.add_argument("--grade-candidates-output", required=True, help="Output GradeCandidate JSONL path.")
    parser.add_argument("--traces-output", required=True, help="Output grade ModelTrace JSONL path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = extract_file(
        args.grade_blocks_input,
        args.recommendation_candidates_input,
        args.grade_candidates_output,
        args.traces_output,
    )
    print("grade_candidates={grade_candidates} traces={model_traces} failed={failed_traces}".format(**summary))


if __name__ == "__main__":
    main()

