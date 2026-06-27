"""从已路由的 block 中抽取规则推荐候选。

本抽取器保留句子级来源信息，并记录质量备注供人工/LLM 复核使用。
它优先输出明确的临床动作句，避免把元数据、方法学、证据背景或版面污染文本
误写成正式推荐候选。
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from src.domain.candidate import ModelTrace, RecommendationCandidate
from src.domain.common import stable_id
from src.common.extraction_common import normalize_text, source_metadata, source_section, source_url, utc_now
from src.common.process_jsonl import iter_jsonl, write_jsonl
from src.pipeline.extraction.recommendation.patterns import ACTION_RECOMMENDATION_RE, RECOMMENDATION_SIGNAL_RE, action_patterns


JsonDict = Dict[str, Any]

RULE_VERSION = "recommendation_rules_v1"
NOISE_ONLY_RE = re.compile(
    r"\b("
    r"open\s+access|creative\s+commons|licen[cs]e|copyright|all\s+rights\s+reserved|"
    r"cataloguing-in-publication|author\s+contributions?|conflicts?\s+of\s+interest|"
    r"corresponding\s+author|trial\s+registration|references"
    r")\b",
    re.I,
)
METHOD_ONLY_RE = re.compile(
    r"\b("
    r"methodology|methods?|study\s+protocol|statistical\s+analysis|"
    r"randomi[sz]ed|cohort|retrospective|prospective|we\s+conducted|"
    r"systematic\s+review\s+was\s+performed"
    r")\b",
    re.I,
)
EVIDENCE_PSEUDO_RECOMMENDATION_RE = re.compile(
    r"\b("
    r"meta-analysis\s+suggests?|evidence\s+suggests?|data\s+suggests?|"
    r"stud(?:y|ies)\s+(?:suggest|suggested|recommend|recommended|show|showed|found|evaluated|assessed)|"
    r"recommendations?\s+(?:are|were)\s+based\s+on|recommendations?arebasedon|"
    r"provides?\s+recommendations?\s+for|"
    r"recommendations?\s+are\s+the\s+result\s+of|"
    r"guideline\s+recommendations?\s+are\s+intended|"
    r"recommendations?\s+in\s+this\s+guideline\s+were\s+formulated|"
    r"recommendations?\s+(?:were\s+)?developed|"
    r"the\s+purpose\s+of\s+(?:this\s+)?guideline|"
    r"(?:this\s+)?guideline\s+provides?\s+recommendations?"
    r")\b",
    re.I,
)
FIGURE_TABLE_RE = re.compile(r"\b(?:figure|fig\.|table)\s*[0-9A-Z]?\b|representative images", re.I)
LAYOUT_GLUE_RE = re.compile(r"(?:\.\.){2,}|[A-Za-z]{18,}|[a-z][A-Z][a-z]")
ADMINISTRATIVE_RE = re.compile(
    r"\b(?:fda|supplementary|contact|registry|copyright|conflicts?\s+of\s+interest|disclaimer|references)\b|"
    r"(参考文献|利益冲突|基金项目|通信作者|作者单位|版权|免责声明)",
    re.I,
)
INCOMPLETE_ACTION_RE = re.compile(
    r"\b(?:we\s+(?:recommend|suggest)(?:\s+that)?|clinicians?\s+should|patients?\s+should|it\s+is\s+(?:recommended|suggested)\s+that)\s*$|"
    r"\b(?:clinicians?\s+should|patients?\s+should|we\s+(?:recommend|suggest)\s+that\s+clinicians?)\s+(?:use|offer|receive|administer|provide|consider)\s*$",
    re.I,
)
RISK_SECTION_RE = re.compile(
    r"\b(?:references|author|disclosure|copyright|methods?|methodology|evidence|rationale|discussion)\b|"
    r"(参考文献|作者|利益冲突|方法|证据|讨论|背景)",
    re.I,
)
RECOMMENDATION_SECTION_RE = re.compile(r"\b(?:recommendations?|summary|key recommendations?)\b|(推荐|建议|指南|共识|治疗|诊断)", re.I)
NEGATIVE_DIRECTION_RE = re.compile(
    r"\b(should\s+not|must\s+not|do\s+not|avoid|not\s+recommended|contraindicat|not\s+indicated|suggests?\s+against|recommend\s+against)\b|"
    r"(不推荐|不建议|不宜|避免|禁用|不应|禁忌)",
    re.I,
)
STRONG_RE = re.compile(r"\b(strong recommendation|we recommend|must|should|standard|class\s+i|cor\s+i)\b|(强推荐|A级推荐|Ⅰ级推荐|应当|必须|首选)", re.I)
CONDITIONAL_RE = re.compile(r"\b(conditional recommendation|we suggest|suggest|may be considered|class\s+iia|cor\s+iia)\b|(建议|可考虑|可以考虑|B级推荐|Ⅱ级推荐)", re.I)
WEAK_RE = re.compile(r"\b(weak recommendation|option|class\s+iib|cor\s+iib)\b|(弱推荐|可选择|C级推荐|D级推荐|Ⅲ级推荐)", re.I)
CERTAINTY_PATTERNS = [
    ("very_low", re.compile(r"\b(very[-\s]+low\s+(?:certainty|quality)|(?:certainty|quality)(?:\s+of\s+evidence)?\s+(?:was|is|were|are)\s+very[-\s]+low)\b", re.I)),
    ("low", re.compile(r"\b(low\s+(?:certainty|quality)|(?:certainty|quality)(?:\s+of\s+evidence)?\s+(?:was|is|were|are)\s+low)\b", re.I)),
    ("moderate", re.compile(r"\b(moderate\s+(?:certainty|quality)|(?:certainty|quality)(?:\s+of\s+evidence)?\s+(?:was|is|were|are)\s+moderate)\b", re.I)),
    ("high", re.compile(r"\b(high\s+(?:certainty|quality)|(?:certainty|quality)(?:\s+of\s+evidence)?\s+(?:was|is|were|are)\s+high)\b", re.I)),
    ("very_low", re.compile(r"(极低质量证据|证据质量极低|极低证据质量|极低级别证据)")),
    ("low", re.compile(r"(低质量证据|证据质量低|低级别证据)")),
    ("moderate", re.compile(r"(中等质量证据|中等证据质量|中级别证据|中等强度证据)")),
    ("high", re.compile(r"(高质量证据|证据质量高|高级别证据|高强度证据)")),
]
REC_CODE_RE = re.compile(r"\b(?:recommendation|statement)\s*([0-9A-Za-z.-]+)|^\s*([0-9]+(?:\.[0-9]+)*)[.)]\s+", re.I)
SPLIT_RE = re.compile(
    r"(?=\b(?:Recommendation|Statement|Clinical question|Question)\s*\d+[A-Za-z]?(?:\.\d+)*\s*[:.)-]?\s+)|"
    r"(?=\b[0-9]+(?:\.[0-9]+)*[.)]\s+[A-Z])|"
    r"(?<=[.!?;。！？；])\s+(?=(?:We|The panel|Clinicians|Patients|Children|Adults|In|For)\b)|"
    r"(?=推荐意见\s*\d*[:：]?)|(?=推荐\s*\d+[:：.、]?)|(?=建议\s*\d+[:：.、]?)|"
    r"(?<=[。！？；])\s*(?=(?:推荐|建议|对于|对|成人|儿童|患者|应|宜|可|不推荐|不建议))"
)
SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s*(?=[A-Z0-9一二三四五六七八九十（(推荐建议对于对成人儿童患者应宜可不])")


def has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(text or "")))


def _section_text(block: JsonDict | None) -> str:
    if not block:
        return ""
    path = block.get("section_path")
    if isinstance(path, list):
        return " ".join(str(item) for item in path)
    return str(path or block.get("heading") or "")


# 给候选句打噪声标签，区分短残句、元说明、版面污染和风险 section。
def statement_noise_reasons(statement: str, block: JsonDict | None = None) -> List[str]:
    """给候选推荐句返回质量风险或噪声标签。"""

    clean = normalize_text(statement)
    section = _section_text(block)
    reasons: List[str] = []
    if not clean:
        return ["empty_statement"]
    min_len = 12 if has_cjk(clean) else 35
    if len(clean) < min_len:
        reasons.append("too_short_fragment")
    if re.match(r"^[,.;:)\]}]", clean):
        reasons.append("starts_with_punctuation")
    if INCOMPLETE_ACTION_RE.search(clean):
        reasons.append("incomplete_action_phrase")
    if EVIDENCE_PSEUDO_RECOMMENDATION_RE.search(clean):
        reasons.append("guideline_meta_statement")
    if ADMINISTRATIVE_RE.search(clean) or re.search(r"(?i)(?:https?://|www\.|doi\.org|doi:\s*10\.)", clean):
        reasons.append("administrative_statement")
    if re.search(r"\bdisclosure\b", section, re.I):
        reasons.append("disclosure_statement")
    if METHOD_ONLY_RE.search(clean) or re.search(r"\bmethods?|methodology\b", section, re.I):
        reasons.append("methodology_statement")
    if re.search(r"\b(?:evidence|rationale|discussion)\b", section, re.I):
        reasons.append("evidence_context_statement")
    if FIGURE_TABLE_RE.search(clean):
        reasons.append("table_or_figure_contaminated")
    if LAYOUT_GLUE_RE.search(clean) and len(clean) > 120:
        reasons.append("layout_glued_statement")
    if len(re.findall(r"\[\d+\]|\(\d+\)|\b\d{3,}\b", clean)) >= 4:
        reasons.append("citation_heavy")
    if ACTION_RECOMMENDATION_RE.search(clean) and not has_cjk(clean) and not re.search(r"\b(?:for|in|with|to|against|receive|use|offer|administer|treat|avoid|refer)\b", clean, re.I):
        reasons.append("missing_clinical_object")
    return reasons


# 对单条推荐句做最终质量判定，决定 accept/reject/needs_review/needs_layout_repair。
def evaluate_recommendation_statement(statement: str, block: JsonDict | None = None) -> JsonDict:
    """判断候选推荐句可直接抽取、需修复、需复核还是应拒绝。"""

    clean = normalize_text(statement)
    patterns = action_patterns(clean)
    reasons = statement_noise_reasons(clean, block)
    section = _section_text(block)
    route = block.get("route") if isinstance(block, dict) else {}
    route_action = route.get("route_action") if isinstance(route, dict) else ""

    # 判定顺序很重要：先排除非动作和元说明，再处理可修复的版面/残句风险。
    if not patterns:
        decision = "reject"
    elif "guideline_meta_statement" in reasons or "administrative_statement" in reasons:
        decision = "reject"
    elif route_action == "needs_layout_repair" or "layout_glued_statement" in reasons:
        decision = "needs_layout_repair"
    elif any(reason in reasons for reason in {"too_short_fragment", "incomplete_action_phrase", "starts_with_punctuation", "table_or_figure_contaminated"}):
        decision = "needs_layout_repair"
    elif RISK_SECTION_RE.search(section) and not RECOMMENDATION_SECTION_RE.search(section):
        decision = "needs_review"
    elif any(reason in reasons for reason in {"evidence_context_statement", "methodology_statement", "citation_heavy", "missing_clinical_object"}):
        decision = "needs_review"
    else:
        decision = "accept"

    score = 0.9 if decision == "accept" else 0.55 if decision == "needs_review" else 0.35 if decision == "needs_layout_repair" else 0.0
    return {
        "decision": decision,
        "quality_score": score,
        "noise_reasons": reasons,
        "action_patterns": patterns,
        "is_complete_statement": "incomplete_action_phrase" not in reasons and "too_short_fragment" not in reasons,
        "is_clinical_action": bool(patterns) and "missing_clinical_object" not in reasons,
        "is_meta_recommendation": "guideline_meta_statement" in reasons,
    }


# 将块文本拆成候选推荐句；拆不出明确动作句时不再回退输出整段。
def split_recommendation_statements(text: str) -> List[str]:
    """把 block 文本拆成候选推荐句。"""

    text = normalize_text(text)
    if not text:
        return []

    parts = [normalize_text(part) for part in SPLIT_RE.split(text) if normalize_text(part)]
    candidates: List[str] = []
    for part in parts:
        candidates.extend(recommendation_statement_candidates(part))
    if candidates:
        return candidates
    return []


# 从一个片段中提取候选句；长文本只允许句子级输出，禁止整段泄漏。
def recommendation_statement_candidates(text: str) -> List[str]:
    """从单个文本片段中抽取候选临床动作句。"""

    clean = normalize_text(text)
    if not clean or is_noise_statement(clean):
        return []
    if len(clean) <= 700:
        evaluation = evaluate_recommendation_statement(clean)
        return [clean] if evaluation["decision"] != "reject" else []

    sentences = [normalize_text(sentence) for sentence in SENTENCE_RE.split(clean) if normalize_text(sentence)]
    action_sentences = [
        sentence
        for sentence in sentences
        if len(sentence) <= 700 and evaluate_recommendation_statement(sentence)["decision"] != "reject" and not is_noise_statement(sentence)
    ]
    return action_sentences


# 快速过滤纯噪声文本，供拆句前后复用。
def is_noise_statement(text: str) -> bool:
    """判断文本是否明显属于行政信息、方法学说明或版面噪声。"""

    clean = normalize_text(text)
    if not clean:
        return True
    if LAYOUT_GLUE_RE.search(clean) and len(clean) > 300:
        return True
    if ADMINISTRATIVE_RE.search(clean) or re.search(r"(?i)(?:https?://|www\.|doi\.org|doi:\s*10\.)", clean):
        return True
    has_action = bool(ACTION_RECOMMENDATION_RE.search(clean))
    if NOISE_ONLY_RE.search(clean) and not has_action:
        return True
    if METHOD_ONLY_RE.search(clean) and not has_action:
        return True
    if EVIDENCE_PSEUDO_RECOMMENDATION_RE.search(clean):
        return True
    if FIGURE_TABLE_RE.search(clean) and not has_action:
        return True
    return False


def recommendation_code(text: str) -> str | None:
    """提取可见的 recommendation/statement 编号；没有时返回 None。"""

    match = REC_CODE_RE.search(text)
    if not match:
        return None
    code = next((group for group in match.groups() if group), None)
    if code and re.fullmatch(r"(?:19|20)\d{2}", code):
        return None
    return code


def infer_direction(text: str) -> str:
    """推断推荐方向是支持、反对还是不明确。"""

    if NEGATIVE_DIRECTION_RE.search(text):
        return "against"
    if RECOMMENDATION_SIGNAL_RE.search(text):
        return "for"
    return "unclear"


def infer_strength(text: str) -> str:
    """根据措辞和 COR 类标签推断推荐强度。"""

    if STRONG_RE.search(text):
        return "strong"
    if CONDITIONAL_RE.search(text):
        return "conditional"
    if WEAK_RE.search(text):
        return "weak"
    return "unclear"


def infer_certainty(text: str) -> str:
    """根据 GRADE 类措辞推断证据确定性或质量等级。"""

    for certainty, pattern in CERTAINTY_PATTERNS:
        if pattern.search(text):
            return certainty
    return "unclear"


# 将长度、片段形态和 statement evaluator 的噪声原因合并为候选质量备注。
def quality_notes(text: str, block: JsonDict | None = None) -> List[str]:
    """把句子形态和 evaluator 发现合并为候选质量备注。"""

    notes: List[str] = []
    clean = normalize_text(text)
    if len(clean) > 1000:
        notes.append("long_statement")
    if len(clean) < (12 if has_cjk(clean) else 35):
        notes.append("short_statement")
    if re.match(r"^[,.;:)\]}]", clean):
        notes.append("starts_with_fragment")
    if re.search(r"(?i)\b(last updated|posted online|www\.|https?://|corresponding author)\b", clean):
        notes.append("administrative_text")
    if is_noise_statement(clean):
        notes.append("noise_statement")
    notes.extend(reason for reason in statement_noise_reasons(clean, block) if reason not in notes)
    return notes


# 根据动作模式、强度/证据等级、路由优先级和质量风险生成规则置信度。
def extraction_confidence(text: str, block: JsonDict) -> float:
    """为规则抽取出的推荐句计算置信度。"""

    evaluation = evaluate_recommendation_statement(text, block)
    score = 0.25
    if RECOMMENDATION_SIGNAL_RE.search(text):
        score += 0.35
    if infer_strength(text) != "unclear":
        score += 0.15
    if infer_certainty(text) != "unclear":
        score += 0.1
    if 40 <= len(text) <= 1200:
        score += 0.1
    route = block.get("route")
    if isinstance(route, dict) and route.get("priority") == "high":
        score += 0.05
    if evaluation["decision"] == "needs_layout_repair":
        score = min(score, 0.5)
    if evaluation["decision"] == "needs_review":
        score = min(score, 0.65)
    notes = quality_notes(text, block)
    if "long_statement" in notes:
        score -= 0.2
    if "starts_with_fragment" in notes:
        score -= 0.15
    if "administrative_text" in notes:
        score -= 0.2
    if "short_statement" in notes:
        score -= 0.1
    return round(max(0.0, min(score, 0.95)), 4)


# 构造 RecommendationCandidate，并把句子级判定结果写入 normalized/raw payload 方便追溯。
def build_candidate(block: JsonDict, statement: str, model_trace_id: str, index: int) -> RecommendationCandidate:
    """构造带可追溯质量元数据的 RecommendationCandidate。"""

    evaluation = evaluate_recommendation_statement(statement, block)
    confidence = extraction_confidence(statement, block)
    notes = quality_notes(statement, block)
    status = "pending" if evaluation["decision"] == "accept" and confidence >= 0.55 and not notes else "needs_review"
    candidate_id = stable_id("rec_candidate", block.get("block_id"), index, statement[:120])
    normalized = {
        "block_id": block.get("block_id", ""),
        "source_block_id": block.get("block_id", ""),
        "statement_index": index,
        "sentence_index": index,
        "section_path": block.get("section_path", []),
        "extraction_reason": "clear_action_pattern",
        "source_order": block.get("order", 0),
        "source_metadata": source_metadata(block),
        "route": block.get("route", {}),
        "quality_notes": notes,
        "statement_evaluation": evaluation,
    }
    return RecommendationCandidate(
        candidate_id=candidate_id,
        record_id=str(block.get("record_id") or ""),
        guideline_id=str(block.get("guideline_id") or "") or None,
        model_trace_id=model_trace_id,
        source_text=str(block.get("text") or ""),
        recommendation_text=statement,
        recommendation_code=recommendation_code(statement),
        source_section=source_section(block),
        source_url=source_url(block),
        direction=infer_direction(statement),
        strength=infer_strength(statement),
        certainty=infer_certainty(statement),
        extraction_method="rule",
        extraction_confidence=confidence,
        status=status,
        review_note=None if status == "pending" else f"Needs manual or LLM review: {', '.join(notes) or 'low_rule_confidence'}.",
        normalized_payload=normalized,
        raw_payload={
            "rule_version": RULE_VERSION,
            "block_id": block.get("block_id", ""),
            "candidate_hints": block.get("candidate_hints", []),
            "quality": block.get("quality", {}),
            "statement_evaluation": evaluation,
        },
        created_at=utc_now(),
        updated_at=utc_now(),
    )


# 为规则抽取生成 ModelTrace，使规则链路和 LLM 链路使用同一套可审计轨迹。
def build_trace(block: JsonDict, statements: List[str], elapsed_ms: int, success: bool = True, error: str | None = None) -> ModelTrace:
    """为一次规则推荐抽取生成 ModelTrace。"""

    trace_id = stable_id("model_trace", RULE_VERSION, block.get("block_id"), "recommendation_extraction")
    parsed = {
        "block_id": block.get("block_id", ""),
        "statement_count": len(statements),
        "statements": statements,
    }
    return ModelTrace(
        model_trace_id=trace_id,
        task_type="recommendation_extraction",
        method="rule",
        model_name=RULE_VERSION,
        input_entity_type="block",
        input_entity_id=str(block.get("block_id") or ""),
        input_text=str(block.get("text") or ""),
        model_version=RULE_VERSION,
        raw_output={"matched": bool(statements), "candidate_hints": block.get("candidate_hints", [])},
        parsed_output=parsed,
        confidence=round(sum(extraction_confidence(item, block) for item in statements) / len(statements), 4) if statements else 0.0,
        parameters={"splitter": "numbered_and_sentence_rules"},
        runtime_ms=elapsed_ms,
        success=success,
        error_message=error,
        target_table="recommendation_candidates",
        created_at=utc_now(),
    )


# 单块抽取入口：repair/skip 块只产 trace 不产候选，避免污染 recommendation queue。
def extract_from_block(block: JsonDict) -> Tuple[List[RecommendationCandidate], ModelTrace]:
    """从一个已路由 block 中抽取零个或多个推荐候选，并返回 trace。"""

    start = time.perf_counter()
    try:
        route = block.get("route")
        if isinstance(route, dict) and route.get("route_action") in {"skip", "needs_layout_repair"}:
            statements = []
        else:
            statements = split_recommendation_statements(str(block.get("text") or ""))
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        trace = build_trace(block, statements, elapsed_ms)
        candidates = [build_candidate(block, statement, trace.model_trace_id, index) for index, statement in enumerate(statements)]
        return candidates, trace
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return [], build_trace(block, [], elapsed_ms, success=False, error=str(exc))


# 批量抽取候选和 trace，输出均转成 JSONL 友好的 dict。
def extract_candidates(blocks: Iterable[JsonDict]) -> Tuple[List[JsonDict], List[JsonDict]]:
    """批量抽取推荐候选和对应 trace，并转换为 JSONL 友好的对象。"""

    candidates: List[JsonDict] = []
    traces: List[JsonDict] = []
    for block in blocks:
        extracted, trace = extract_from_block(block)
        traces.append(trace.to_dict())
        candidates.extend(candidate.to_dict() for candidate in extracted)
    return candidates, traces


def extract_file(input_path: str | Path, candidates_output: str | Path, traces_output: str | Path) -> JsonDict:
    """从已路由推荐 block JSONL 中抽取 RecommendationCandidate。"""

    candidates, traces = extract_candidates(iter_jsonl(input_path))
    write_jsonl(candidates_output, candidates)
    write_jsonl(traces_output, traces)
    return {
        "input_path": str(input_path),
        "recommendation_candidates": len(candidates),
        "model_traces": len(traces),
        "successful_traces": sum(1 for trace in traces if trace.get("success")),
        "failed_traces": sum(1 for trace in traces if not trace.get("success")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract rule-based RecommendationCandidate JSONL from routed recommendation blocks.")
    parser.add_argument("--input", required=True, help="Input *_recommendation_blocks.jsonl path.")
    parser.add_argument("--candidates-output", required=True, help="Output RecommendationCandidate JSONL path.")
    parser.add_argument("--traces-output", required=True, help="Output ModelTrace JSONL path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = extract_file(args.input, args.candidates_output, args.traces_output)
    print(
        "candidates={recommendation_candidates} traces={model_traces} failed={failed_traces}".format(
            **summary
        )
    )


if __name__ == "__main__":
    main()
