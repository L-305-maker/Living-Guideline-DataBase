"""PICO 抽取文件：从文本块中抽取人群、干预、对照和结局等临床问题结构。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from src.common.extraction_common import normalize_text, source_metadata, source_section, utc_now
from src.common.process_jsonl import iter_jsonl, write_jsonl
from src.common.record_quality import block_record_quality_context
from src.domain.candidate import ModelTrace
from src.domain.common import stable_id
from src.domain.knowledge import PicoQuestion
from src.pipeline.extraction.common.extractor_common import RuleTraceInput, build_rule_trace, compact, mean_confidence

# 模块职责:
# 从已路由 block 中用规则抽取 PICO 问题。这里允许保守输出:
# 不完整的 PICO 行保留为 under_review，不直接提升为可用指南知识。


JsonDict = Dict[str, Any]

RULE_VERSION = "pico_rules_v1"

QUESTION_RE = re.compile(
    r"(?P<question>(?:clinical\s+question|key\s+question|pico(?:t)?\s+question)\s*[:\-]\s*[^.?!]+[.?!]?|"
    r"(?:临床问题|关键问题|PICO问题)\s*[:：]\s*[^。！？；]+[。！？]?)",
    re.I,
)
POPULATION_RE = re.compile(
    r"\b(?:in|for|among)\s+(?P<population>(?:adults?|children|patients?|people|individuals?|infants?|women|men)"
    r"[^.;:]{0,160}?(?:with|who\s+have|at\s+risk\s+of|undergoing|receiving)[^.;:]{0,180})",
    re.I,
)
SUBJECT_INTERVENTION_POPULATION_RE = re.compile(
    r"\b(?P<intervention>[A-Z][A-Za-z0-9()/ -]{3,140}?)\s+(?:in|for|among)\s+"
    r"(?P<population>(?:adults?|children|patients?|people|individuals?|infants?|women|men)[^.;:]{0,160}?)\s+"
    r"(?:should|is\s+indicated|are\s+indicated|may\s+benefit|is\s+recommended|are\s+recommended)\b",
    re.I,
)
INDICATED_FOR_POPULATION_RE = re.compile(
    r"\b(?P<intervention>[A-Z][A-Za-z0-9()/ -]{3,100}?)\s+"
    r"(?:is|are)\s+indicated\s+for\s+(?P<indication>[^.;:]{3,140}?)\s+"
    r"(?:in|for|among)\s+(?P<population>(?:adults?|children|patients?|people|individuals?|infants?|women|men)[^.;:]{0,160})",
    re.I,
)
LEADING_POPULATION_RE = re.compile(
    r"^\s*(?P<population>(?:adults?|children|patients?|people|individuals?|infants?|women|men)"
    r"[^.;:]{0,180}?(?:with|who\s+have|at\s+risk\s+of|undergoing|receiving|treated\s+with)[^.;:]{0,180})\s+"
    r"(?:should|may|must|is|are|benefit|require)\b",
    re.I,
)
PREFERRED_FOR_POPULATION_RE = re.compile(
    r"\b(?P<intervention>[A-Z0-9][^.;:]{2,180}?)\s+(?:are|is)\s+(?:preferred|alternative)?\s*"
    r"(?:treatment\s+)?options?\s+for\s+"
    r"(?P<population>(?:adults?|children|patients?|people|individuals?|infants?|women|men)\s+with[^.;:]{3,180})",
    re.I,
)
OPTIONS_INCLUDE_RE = re.compile(
    r"\b(?:preferred\s+)?(?:treatment\s+)?options?\s+for\s+"
    r"(?P<population>(?:adults?|children|patients?|people|individuals?|infants?|women|men)\s+with[^.;:]{3,180}?)\s+"
    r"(?:include|are)\s+(?P<intervention>[^.;:]{3,180})",
    re.I,
)
RANDOMIZED_TO_RE = re.compile(r"\brandomi[sz]ed\s+to\s+(?P<intervention>[^.;:]{3,180})", re.I)
INTERVENTION_RE = re.compile(
    r"\b(?:should\s+receive|should\s+be\s+offered|should\s+be\s+performed|is\s+indicated|are\s+indicated|"
    r"recommend(?:ed)?|suggest(?:ed)?|offer|consider|use|treated\s+with|treatment\s+with)\s+"
    r"(?P<intervention>[^.;:]{3,220})|"
    r"(?:推荐|建议|应当|应|应该|宜|可考虑|可以考虑|可予|可用|使用|采用|给予|接受|进行|首选|避免|不推荐|不建议)"
    r"(?P<zh_intervention>[^。；，,]{2,120})",
    re.I,
)
TREATED_WITH_RE = re.compile(r"\btreated\s+with\s+(?P<intervention>[^.;:]{3,160})", re.I)
NOISE_RE = re.compile(
    r"\b(last\s+updated|posted\s+online|https?://|pleasecheckwebsite|copyright|journal\s+of|"
    r"sleep,\s+vol\.|et\s+al|doi\.org)\b",
    re.I,
)
POPULATION_CLAUSE_BOUNDARY_RE = re.compile(
    r"\s+\b(?:if|assuming|based\s+on|compared\s+with|versus|vs\.?|have\s+included|were\s+randomi[sz]ed)\b.*$",
    re.I,
)
COMPARATOR_RE = re.compile(r"\b(?:compared\s+with|versus|vs\.?|rather\s+than)\s+(?P<comparator>[^.;:]{3,160})", re.I)
OUTCOME_RE = re.compile(
    r"\b(?:outcomes?|to\s+assess|to\s+determine|to\s+reduce|to\s+improve|to\s+prevent|for\s+the\s+prevention\s+of)\s+"
    r"(?P<outcome>[^.;:]{3,180})|"
    r"(?:降低|减少|改善|提高|预防|防治|控制|缓解)(?P<zh_outcome>[^。；，,]{2,120})",
    re.I,
)
NUMBERED_SPLIT_RE = re.compile(r"(?=\b\d+(?:\.\d+)*[.)]\s+[A-Z])")
ZH_POPULATION_RE = re.compile(
    r"(?P<population>(?:成人|儿童|青少年|老年|孕妇|患者|病人|病例|血液病患者|血液肿瘤患者|"
    r"淋巴瘤患者|白血病患者|血友病患者|多发性骨髓瘤患者|ITP患者|感染患者|移植患者)"
    r"[^。；，,]{0,80})"
)
ZH_POPULATION_WITH_CONDITION_RE = re.compile(
    r"(?:对于|对|在|针对)?(?P<population>[^。；，,]{0,80}?(?:患者|病人|儿童|成人|青少年|孕妇|老年人))"
)


def split_pico_units(text: str) -> List[str]:
    """把 block 拆成显式临床问题或编号问题单元。"""

    clean = normalize_text(text)
    if not clean:
        return []
    explicit_questions = [compact(match.group("question"), 600) for match in QUESTION_RE.finditer(clean)]
    if explicit_questions:
        return explicit_questions
    parts = [compact(part, 700) for part in NUMBERED_SPLIT_RE.split(clean) if compact(part)]
    useful = [
        part
        for part in parts
        if POPULATION_RE.search(part) or ZH_POPULATION_RE.search(part) or INTERVENTION_RE.search(part) or OUTCOME_RE.search(part)
    ]
    return useful or ([clean] if POPULATION_RE.search(clean) or ZH_POPULATION_RE.search(clean) else [])


def infer_population(text: str) -> str:
    """从常见指南句式中推断 PICO 的 population 短语。"""

    option_match = OPTIONS_INCLUDE_RE.search(text) or PREFERRED_FOR_POPULATION_RE.search(text)
    if option_match:
        return clean_population(option_match.group("population"))
    indicated_match = INDICATED_FOR_POPULATION_RE.search(text)
    if indicated_match:
        return clean_population(indicated_match.group("population"))
    subject_match = SUBJECT_INTERVENTION_POPULATION_RE.search(text)
    if subject_match:
        return clean_population(subject_match.group("population"))
    leading_match = LEADING_POPULATION_RE.search(text)
    if leading_match:
        return clean_population(leading_match.group("population"))
    match = POPULATION_RE.search(text)
    if match:
        return clean_population(match.group("population"))
    zh_match = ZH_POPULATION_RE.search(text) or ZH_POPULATION_WITH_CONDITION_RE.search(text)
    if zh_match:
        return clean_population(zh_match.group("population"))
    return "unclear"


def infer_intervention(text: str) -> str:
    """从推荐句和研究描述句式中推断 intervention 文本。"""

    option_match = OPTIONS_INCLUDE_RE.search(text) or PREFERRED_FOR_POPULATION_RE.search(text)
    if option_match:
        return compact(option_match.group("intervention"))
    indicated_match = INDICATED_FOR_POPULATION_RE.search(text)
    if indicated_match:
        intervention = compact(indicated_match.group("intervention"))
        indication = compact(indicated_match.group("indication"), 120)
        return compact(f"{intervention} for {indication}")
    subject_match = SUBJECT_INTERVENTION_POPULATION_RE.search(text)
    if subject_match:
        return compact(subject_match.group("intervention"))
    randomized_match = RANDOMIZED_TO_RE.search(text)
    if randomized_match:
        return compact(randomized_match.group("intervention"))
    treated_match = TREATED_WITH_RE.search(text)
    if treated_match:
        return compact(treated_match.group("intervention"))
    match = INTERVENTION_RE.search(text)
    if match:
        return compact(match.group("intervention") or match.group("zh_intervention"))
    return "unclear"


def infer_comparator(text: str) -> str | None:
    """当文本明确写出 comparator 时提取对照信息。"""

    match = COMPARATOR_RE.search(text)
    return compact(match.group("comparator")) if match else None


def infer_outcomes(text: str) -> List[JsonDict]:
    """把 outcome 提及抽成轻量结构化对象。"""

    outcomes: List[JsonDict] = []
    for match in OUTCOME_RE.finditer(text):
        value = compact(match.group("outcome") or match.group("zh_outcome"))
        if value and value.lower() not in {str(item.get("name", "")).lower() for item in outcomes}:
            outcomes.append({"name": value, "source": "rule"})
    return outcomes[:8]


def clinical_question(unit: str, population: str, intervention: str, comparator: str | None, outcomes: List[JsonDict]) -> str:
    """根据抽取出的 PICO 字段拼出可读临床问题。"""

    if QUESTION_RE.search(unit):
        return compact(unit, 700)
    parts = []
    if population != "unclear":
        parts.append(f"In {population}")
    if intervention != "unclear":
        parts.append(f"should {intervention}")
    if comparator:
        parts.append(f"compared with {comparator}")
    if outcomes:
        parts.append("for " + "; ".join(str(item.get("name")) for item in outcomes if item.get("name")))
    return compact(" ".join(parts) or unit, 700)


def extraction_confidence(population: str, intervention: str, outcomes: List[JsonDict], unit: str) -> float:
    """在质量惩罚前，按字段完整度计算规则抽取置信度。"""

    score = 0.25
    if population != "unclear":
        score += 0.25
    if intervention != "unclear":
        score += 0.25
    if outcomes:
        score += 0.1
    if QUESTION_RE.search(unit):
        score += 0.1
    return round(min(score, 0.95), 4)


def record_quality_context(block: JsonDict) -> JsonDict:
    """读取会影响 PICO 置信度的记录级质量字段。"""

    context = block_record_quality_context(block)
    return {
        "record_type": str(context.get("record_type") or ""),
        "quality_flags": list(context.get("record_quality_flags") or []),
    }


def adjusted_extraction_confidence(
    population: str,
    intervention: str,
    outcomes: List[JsonDict],
    unit: str,
    block: JsonDict,
) -> float:
    """针对字段不完整或记录质量较低的情况应用置信度上限/扣分。"""

    score = extraction_confidence(population, intervention, outcomes, unit)
    context = record_quality_context(block)
    if not outcomes:
        score = min(score, 0.64)
    if "single_line_flattened_text" in context["quality_flags"]:
        score -= 0.08
    if "single_section_large_document" in context["quality_flags"]:
        score -= 0.05
    if "table_heavy_without_structured_tables" in context["quality_flags"]:
        score -= 0.05
    if context["record_type"] == "paper":
        score = min(score, 0.7)
    return round(max(0.0, min(score, 0.95)), 4)


def clean_population(value: str) -> str:
    """裁掉 population 短语尾部混入的证据或比较子句。"""

    clean = POPULATION_CLAUSE_BOUNDARY_RE.sub("", compact(value))
    if not clean or NOISE_RE.search(clean):
        return "unclear"
    return clean


def should_emit_pico(unit: str, population: str, intervention: str) -> bool:
    """只有 population 和 intervention 都存在时才输出 PICO 行。"""

    return population != "unclear" and intervention != "unclear"


def build_pico(block: JsonDict, unit: str, index: int) -> JsonDict:
    """构造一条带来源追踪和质量上下文的 PicoQuestion。"""

    population = infer_population(unit)
    intervention = infer_intervention(unit)
    comparator = infer_comparator(unit)
    outcomes = infer_outcomes(unit)
    confidence = adjusted_extraction_confidence(population, intervention, outcomes, unit, block)
    status = "active" if confidence >= 0.8 else "under_review"
    pico_id = stable_id("pico", block.get("guideline_id"), block.get("record_id"), block.get("block_id"), index, unit[:160])
    row = PicoQuestion(
        pico_id=pico_id,
        guideline_id=str(block.get("guideline_id") or ""),
        clinical_question=clinical_question(unit, population, intervention, comparator, outcomes),
        population=population,
        intervention=intervention,
        comparator=comparator,
        outcomes=outcomes,
        priority="medium",
        status=status,
        source_record_id=str(block.get("record_id") or "") or None,
        source_text=str(block.get("text") or ""),
        source_span=str(block.get("text") or ""),
        extraction_method="rule",
        created_at=utc_now(),
        updated_at=utc_now(),
    ).to_dict()
    row["extraction_confidence"] = confidence
    row["source_block_id"] = block.get("block_id", "")
    row["source_order"] = block.get("order", 0)
    row["source_section"] = source_section(block)
    row["normalized_payload"] = {
        "rule_version": RULE_VERSION,
        "block_id": block.get("block_id", ""),
        "source_order": block.get("order", 0),
        "source_metadata": source_metadata(block),
        "route": block.get("route", {}),
        "record_quality": record_quality_context(block),
        "unit_index": index,
    }
    return row


def build_trace(block: JsonDict, picos: List[JsonDict], elapsed_ms: int, success: bool = True, error: str | None = None) -> ModelTrace:
    """为一次 PICO 规则抽取生成 ModelTrace。"""

    parsed = {
        "block_id": block.get("block_id", ""),
        "pico_count": len(picos),
        "pico_ids": [pico.get("pico_id") for pico in picos],
    }
    return build_rule_trace(
        RuleTraceInput(
            rule_version=RULE_VERSION,
            task_type="pico_extraction",
            block=block,
            parsed_output=parsed,
            confidence=mean_confidence(picos),
            parameters={"splitter": "question_or_numbered_units", "rule_version": RULE_VERSION},
            elapsed_ms=elapsed_ms,
            success=success,
            error=error,
            target_table="pico_questions",
            target_entity_id=str(picos[0].get("pico_id") or "") if len(picos) == 1 else None,
        )
    )


def extract_from_block(block: JsonDict) -> Tuple[List[JsonDict], ModelTrace]:
    """从一个已路由 block 中抽取零条或多条 PICO 行。"""

    start = time.perf_counter()
    try:
        units = split_pico_units(str(block.get("text") or ""))
        picos: List[JsonDict] = []
        for index, unit in enumerate(units):
            population = infer_population(unit)
            intervention = infer_intervention(unit)
            if should_emit_pico(unit, population, intervention):
                picos.append(build_pico(block, unit, index))
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return picos, build_trace(block, picos, elapsed_ms)
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return [], build_trace(block, [], elapsed_ms, success=False, error=str(exc))


def extract_picos(blocks: Iterable[JsonDict]) -> Tuple[List[JsonDict], List[JsonDict]]:
    """批量抽取 PICO 行及对应 trace。"""

    picos: List[JsonDict] = []
    traces: List[JsonDict] = []
    for block in blocks:
        extracted, trace = extract_from_block(block)
        picos.extend(extracted)
        traces.append(trace.to_dict())
    return picos, traces


def summarize(picos: List[JsonDict], traces: List[JsonDict]) -> JsonDict:
    """统计 PICO 抽取数量以及 active/under_review 分布。"""

    return {
        "pico_questions": len(picos),
        "model_traces": len(traces),
        "active_picos": sum(1 for pico in picos if pico.get("status") == "active"),
        "under_review_picos": sum(1 for pico in picos if pico.get("status") == "under_review"),
        "trace_success": sum(1 for trace in traces if trace.get("success")),
    }


def extract_file(input_path: str | Path, output_path: str | Path, traces_output: str | Path) -> JsonDict:
    """从已路由 PICO block JSONL 中抽取 PicoQuestion 行。"""

    picos, traces = extract_picos(iter_jsonl(input_path))
    write_jsonl(output_path, picos)
    write_jsonl(traces_output, traces)
    summary = summarize(picos, traces)
    summary["input"] = str(input_path)
    summary["output"] = str(output_path)
    summary["traces_output"] = str(traces_output)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract rule-based PicoQuestion JSONL from routed PICO blocks.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--traces-output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = extract_file(args.input, args.output, args.traces_output)
    print("pico_questions={pico_questions} active={active_picos} under_review={under_review_picos}".format(**summary))


if __name__ == "__main__":
    main()

