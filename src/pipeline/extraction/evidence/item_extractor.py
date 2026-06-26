from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.common.extraction_common import normalize_text, source_metadata, source_section, source_url, utc_now
from src.common.process_jsonl import iter_jsonl, write_jsonl
from src.common.record_quality import block_record_quality_context
from src.domain.candidate import ModelTrace
from src.domain.common import stable_id
from src.domain.knowledge import EvidenceItem
from src.pipeline.extraction.common.extractor_common import RuleTraceInput, block_id_from_payload, build_rule_trace, compact, source_order

# 模块职责:
# 从已路由 block 中抽取结构化证据片段，并把证据关联到附近的 PICO 与推荐候选。
# 无法关联或结构较弱的证据不会进入 pending 知识，而是进入 association_review。


JsonDict = Dict[str, Any]

RULE_VERSION = "evidence_rules_v1"
MAX_PICO_ORDER_DISTANCE = 12
MAX_RECOMMENDATION_ORDER_DISTANCE = 8

STUDY_PATTERNS = [
    ("systematic_review", re.compile(r"\bsystematic\s+review\b", re.I)),
    ("meta_analysis", re.compile(r"\bmeta[-\s]analysis\b", re.I)),
    ("RCT", re.compile(r"\brandomi[sz]ed\b|\bRCTs?\b|\bclinical\s+trial\b", re.I)),
    ("cohort", re.compile(r"\bcohort\b", re.I)),
    ("case_control", re.compile(r"\bcase[-\s]control\b", re.I)),
    ("cross_sectional", re.compile(r"\bcross[-\s]sectional\b", re.I)),
    ("case_series", re.compile(r"\bcase\s+series\b", re.I)),
    ("guideline", re.compile(r"\bguidelines?\b|\bpractice\s+parameters?\b", re.I)),
]
SAMPLE_SIZE_RE = re.compile(r"\b(?:n\s*=\s*|sample\s+size\s+(?:of\s+)?)(?P<n>[0-9][0-9,]*)\b", re.I)
CI_RE = re.compile(r"\b(?:95%\s*)?CI\s*(?:=|:)?\s*(?P<ci>[0-9.]+\s*(?:-|to|,)\s*[0-9.]+)\b", re.I)
EFFECT_RE = re.compile(
    r"\b(?P<measure>RR|OR|HR|MD|SMD|risk\s+ratio|odds\s+ratio|hazard\s+ratio)\s*(?:=|of|:)?\s*(?P<value>[0-9.]+)",
    re.I,
)
BENEFIT_RE = re.compile(r"\b(reduc(?:ed|es|tion)|improv(?:ed|es|ement)|benefit|effective|lower risk|decrease)\b", re.I)
HARM_RE = re.compile(r"\b(increas(?:ed|es|e)|harm|adverse|higher risk|worse|mortality)\b", re.I)
NO_EFFECT_RE = re.compile(r"\b(no significant difference|no effect|not significant|similar)\b", re.I)
OUTCOME_RE = re.compile(
    r"\b(?:outcomes?|endpoint|risk\s+of|rate\s+of|incidence\s+of|mortality|symptoms?|quality\s+of\s+life)\s+"
    r"(?P<outcome>[^.;:]{0,160})",
    re.I,
)


class PicoIndex:
    """从 evidence block 到 PICO 问题的近邻查找索引。"""

    def __init__(self, rows: Iterable[JsonDict]) -> None:
        self.by_record: Dict[str, List[JsonDict]] = {}
        self.by_guideline: Dict[str, List[JsonDict]] = {}
        for row in rows:
            record_id = str(row.get("source_record_id") or row.get("record_id") or "")
            guideline_id = str(row.get("guideline_id") or "")
            if record_id:
                self.by_record.setdefault(record_id, []).append(row)
            if guideline_id:
                self.by_guideline.setdefault(guideline_id, []).append(row)
        for group in [*self.by_record.values(), *self.by_guideline.values()]:
            group.sort(key=source_order)

    def find(self, block: JsonDict) -> Tuple[str, str]:
        """返回最近的 PICO ID 以及本次关联理由。"""

        record_id = str(block.get("record_id") or "")
        guideline_id = str(block.get("guideline_id") or "")
        candidates = self.by_record.get(record_id) or self.by_guideline.get(guideline_id) or []
        if not candidates:
            return "", "no_pico_input_match"
        block_order = source_order(block)
        nearest = min(candidates, key=lambda row: abs(source_order(row) - block_order))
        distance = abs(source_order(nearest) - block_order)
        if distance <= MAX_PICO_ORDER_DISTANCE:
            return str(nearest.get("pico_id") or ""), f"nearest_pico_order_{distance}"
        return "", "no_nearby_pico"


class RecommendationIndex:
    """按同 block 或相近 source order 查找推荐候选。"""

    def __init__(self, rows: Iterable[JsonDict]) -> None:
        self.by_block: Dict[str, List[JsonDict]] = {}
        self.by_record: Dict[str, List[JsonDict]] = {}
        for row in rows:
            block_id = block_id_from_payload(row)
            record_id = str(row.get("record_id") or "")
            if block_id:
                self.by_block.setdefault(block_id, []).append(row)
            if record_id:
                self.by_record.setdefault(record_id, []).append(row)
        for items in self.by_record.values():
            items.sort(key=source_order)

    def find(self, block: JsonDict) -> Tuple[str, str]:
        """返回推荐候选 ID 以及本次关联理由。"""

        block_id = str(block.get("block_id") or "")
        same_block = self.by_block.get(block_id) or []
        if same_block:
            return str(same_block[0].get("candidate_id") or ""), "same_block"
        record_id = str(block.get("record_id") or "")
        candidates = self.by_record.get(record_id) or []
        if not candidates:
            return "", "no_recommendation_input_match"
        block_order = source_order(block)
        nearest = min(candidates, key=lambda row: abs(source_order(row) - block_order))
        distance = abs(source_order(nearest) - block_order)
        if distance <= MAX_RECOMMENDATION_ORDER_DISTANCE:
            return str(nearest.get("candidate_id") or ""), f"nearest_recommendation_order_{distance}"
        return "", "no_nearby_recommendation"


def infer_study_design(text: str) -> str:
    """根据常见证据描述短语推断研究设计。"""

    for design, pattern in STUDY_PATTERNS:
        if pattern.search(text):
            return design
    return "unclear"


def infer_sample_size(text: str) -> Optional[int]:
    """解析简单样本量表达，例如 ``n=123``。"""

    match = SAMPLE_SIZE_RE.search(text)
    if not match:
        return None
    try:
        return int(match.group("n").replace(",", ""))
    except ValueError:
        return None


def infer_effect_size(text: str) -> JsonDict:
    """文本中存在效应量时，提取紧凑的 effect-size payload。"""

    match = EFFECT_RE.search(text)
    if not match:
        return {}
    return {"measure": compact(match.group("measure"), 40), "value": match.group("value")}


def infer_confidence_interval(text: str) -> Optional[str]:
    """从证据文本中提取置信区间字符串。"""

    match = CI_RE.search(text)
    return compact(match.group("ci"), 80) if match else None


def infer_effect_direction(text: str) -> str:
    """判断证据方向是获益、伤害、混合、无效还是不确定。"""

    benefit = bool(BENEFIT_RE.search(text))
    harm = bool(HARM_RE.search(text))
    if NO_EFFECT_RE.search(text):
        return "no_effect"
    if benefit and harm:
        return "mixed"
    if benefit:
        return "benefit"
    if harm:
        return "harm"
    return "uncertain"


def infer_outcomes(text: str) -> List[JsonDict]:
    """从证据文本中提取 outcome 提及。"""

    outcomes: List[JsonDict] = []
    for match in OUTCOME_RE.finditer(text):
        value = compact(match.group("outcome"))
        if value and value.lower() not in {str(item.get("name", "")).lower() for item in outcomes}:
            outcomes.append({"name": value, "source": "rule"})
    return outcomes[:8]


def record_quality_context(block: JsonDict) -> JsonDict:
    """读取会影响证据置信度的记录级质量字段。"""

    context = block_record_quality_context(block)
    return {
        "record_type": str(context.get("record_type") or ""),
        "quality_flags": list(context.get("record_quality_flags") or []),
    }


def has_structured_evidence_signal(text: str) -> bool:
    """判断文本是否包含足够结构化证据信号，值得继续复核。"""

    study_design = infer_study_design(text)
    return any(
        [
            study_design not in {"unclear", "guideline"},
            infer_sample_size(text) is not None,
            bool(infer_effect_size(text)),
            bool(infer_confidence_interval(text)),
            infer_effect_direction(text) != "uncertain",
            bool(infer_outcomes(text)),
        ]
    )


def extraction_confidence(text: str, pico_id: str, rec_id: str, block: JsonDict | None = None) -> float:
    """在分配筛查状态前，按证据完整度和关联质量计算置信度。"""

    score = 0.2
    if infer_study_design(text) != "unclear":
        score += 0.2
    if infer_effect_size(text):
        score += 0.15
    if infer_confidence_interval(text):
        score += 0.1
    if infer_effect_direction(text) != "uncertain":
        score += 0.1
    if pico_id:
        score += 0.15
    if rec_id:
        score += 0.1
    if block is not None:
        context = record_quality_context(block)
        if "single_line_flattened_text" in context["quality_flags"]:
            score -= 0.05
        if "table_heavy_without_structured_tables" in context["quality_flags"] and not infer_effect_size(text):
            score -= 0.05
        if context["record_type"] == "guideline" and not pico_id and not rec_id:
            score = min(score, 0.6)
    return round(min(score, 0.95), 4)


def build_evidence_item(block: JsonDict, pico_index: PicoIndex, rec_index: RecommendationIndex, include_unlinked: bool = False) -> Optional[JsonDict]:
    """构造一条 EvidenceItem；信号或关联太弱时返回 None。"""

    text = normalize_text(block.get("text"))
    pico_id, pico_reason = pico_index.find(block)
    rec_id, rec_reason = rec_index.find(block)
    if not pico_id and not include_unlinked:
        return None
    structured_signal = has_structured_evidence_signal(text)
    if not pico_id and not rec_id and not structured_signal:
        return None
    confidence = extraction_confidence(text, pico_id, rec_id, block)
    screening_status = "pending" if confidence >= 0.55 and pico_id else "association_review" if structured_signal else "uncertain"
    evidence_id = stable_id("evidence", block.get("paper_id"), block.get("block_id"), pico_id, rec_id, text[:160])
    item = EvidenceItem(
        evidence_id=evidence_id,
        paper_id=str(block.get("paper_id") or "") or None,
        pico_id=pico_id,
        source_record_id=str(block.get("record_id") or "") or None,
        recommendation_candidate_id=rec_id or None,
        study_design=infer_study_design(text),
        sample_size=infer_sample_size(text),
        outcomes_extracted=infer_outcomes(text),
        effect_size=infer_effect_size(text),
        confidence_interval=infer_confidence_interval(text),
        effect_direction=infer_effect_direction(text),
        extraction_method="rule",
        extraction_confidence=confidence,
        screening_status=screening_status,
        source_span=str(block.get("text") or ""),
        created_at=utc_now(),
    ).to_dict()
    item["source_block_id"] = block.get("block_id", "")
    item["source_order"] = block.get("order", 0)
    item["source_text"] = block.get("text", "")
    item["source_section"] = source_section(block)
    item["source_url"] = source_url(block)
    item["normalized_payload"] = {
        "rule_version": RULE_VERSION,
        "block_id": block.get("block_id", ""),
        "source_order": block.get("order", 0),
        "source_metadata": source_metadata(block),
        "route": block.get("route", {}),
        "record_quality": record_quality_context(block),
        "has_structured_evidence_signal": structured_signal,
        "pico_association_reason": pico_reason,
        "recommendation_association_reason": rec_reason,
    }
    return item


def build_trace(block: JsonDict, item: Optional[JsonDict], elapsed_ms: int, success: bool = True, error: str | None = None) -> ModelTrace:
    """为一次 evidence 规则抽取生成 ModelTrace。"""

    skipped_reason = None
    if not item:
        text = normalize_text(block.get("text"))
        skipped_reason = "no_structured_evidence_signal" if not has_structured_evidence_signal(text) else "missing_pico_id"
    parsed = {
        "block_id": block.get("block_id", ""),
        "evidence_id": item.get("evidence_id") if item else None,
        "pico_id": item.get("pico_id") if item else None,
        "recommendation_candidate_id": item.get("recommendation_candidate_id") if item else None,
        "skipped_reason": skipped_reason,
    }
    return build_rule_trace(
        RuleTraceInput(
            rule_version=RULE_VERSION,
            task_type="evidence_extraction",
            block=block,
            parsed_output=parsed,
            confidence=float(item.get("extraction_confidence") or 0.0) if item else 0.0,
            parameters={
                "association": "same_record_nearest_order",
                "rule_version": RULE_VERSION,
                "max_pico_order_distance": MAX_PICO_ORDER_DISTANCE,
                "max_recommendation_order_distance": MAX_RECOMMENDATION_ORDER_DISTANCE,
            },
            elapsed_ms=elapsed_ms,
            success=success,
            error=error,
            target_table="evidence_items",
            target_entity_id=str(item.get("evidence_id") or "") if item else None,
        )
    )


def extract_from_block(block: JsonDict, pico_index: PicoIndex, rec_index: RecommendationIndex, include_unlinked: bool = False) -> Tuple[Optional[JsonDict], ModelTrace]:
    """从一个已路由 evidence block 中最多抽取一条 EvidenceItem。"""

    start = time.perf_counter()
    try:
        item = build_evidence_item(block, pico_index, rec_index, include_unlinked=include_unlinked)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return item, build_trace(block, item, elapsed_ms)
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return None, build_trace(block, None, elapsed_ms, success=False, error=str(exc))


def extract_evidence_items(
    blocks: Iterable[JsonDict],
    picos: Iterable[JsonDict] = (),
    recommendations: Iterable[JsonDict] = (),
    include_unlinked: bool = False,
) -> Tuple[List[JsonDict], List[JsonDict]]:
    """基于预构建的 PICO/推荐输入，批量抽取证据项和 trace。"""

    pico_index = PicoIndex(picos)
    rec_index = RecommendationIndex(recommendations)
    items: List[JsonDict] = []
    traces: List[JsonDict] = []
    for block in blocks:
        item, trace = extract_from_block(block, pico_index, rec_index, include_unlinked=include_unlinked)
        if item:
            items.append(item)
        traces.append(trace.to_dict())
    return items, traces


def summarize(items: List[JsonDict], traces: List[JsonDict]) -> JsonDict:
    """统计证据输出、关联质量和跳过原因。"""

    return {
        "evidence_items": len(items),
        "model_traces": len(traces),
        "linked_to_pico": sum(1 for item in items if item.get("pico_id")),
        "linked_to_recommendation": sum(1 for item in items if item.get("recommendation_candidate_id")),
        "pending_items": sum(1 for item in items if item.get("screening_status") == "pending"),
        "uncertain_items": sum(1 for item in items if item.get("screening_status") == "uncertain"),
        "skipped_missing_pico": sum(1 for trace in traces if (trace.get("parsed_output") or {}).get("skipped_reason") == "missing_pico_id"),
        "trace_success": sum(1 for trace in traces if trace.get("success")),
    }


def extract_file(
    input_path: str | Path,
    output_path: str | Path,
    traces_output: str | Path,
    picos_input: str | Path | None = None,
    recommendations_input: str | Path | None = None,
    include_unlinked: bool = False,
) -> JsonDict:
    """从流水线脚本使用的 JSONL 文件中抽取 EvidenceItem。"""

    picos = list(iter_jsonl(picos_input)) if picos_input else []
    recommendations = list(iter_jsonl(recommendations_input)) if recommendations_input else []
    items, traces = extract_evidence_items(
        iter_jsonl(input_path),
        picos=picos,
        recommendations=recommendations,
        include_unlinked=include_unlinked,
    )
    write_jsonl(output_path, items)
    write_jsonl(traces_output, traces)
    summary = summarize(items, traces)
    summary["input"] = str(input_path)
    summary["output"] = str(output_path)
    summary["traces_output"] = str(traces_output)
    summary["picos_input"] = str(picos_input or "")
    summary["recommendations_input"] = str(recommendations_input or "")
    summary["include_unlinked"] = include_unlinked
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract rule-based EvidenceItem JSONL from routed evidence blocks.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--traces-output", required=True)
    parser.add_argument("--picos-input", default=None)
    parser.add_argument("--recommendations-input", default=None)
    parser.add_argument("--include-unlinked", action="store_true", help="Keep evidence rows that could not be linked to a PICO.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = extract_file(
        input_path=args.input,
        output_path=args.output,
        traces_output=args.traces_output,
        picos_input=args.picos_input,
        recommendations_input=args.recommendations_input,
        include_unlinked=args.include_unlinked,
    )
    print(
        "evidence_items={evidence_items} linked_pico={linked_to_pico} linked_rec={linked_to_recommendation} uncertain={uncertain_items}".format(
            **summary
        )
    )


if __name__ == "__main__":
    main()
