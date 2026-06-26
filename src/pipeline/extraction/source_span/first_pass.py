from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple

from src.common.extraction_common import utc_now
from src.domain.common import stable_id
from src.pipeline.cleaning.source_cleaner import (
    clean_source_record,
    extract_tables_from_text,
    light_clean_pdf_text,
    split_affiliations,
    split_references,
    split_sections,
)
from src.pipeline.cleaning.offset_mapping import span_offset_fields
from src.pipeline.extraction.common.enums import normalize_certainty, normalize_direction, normalize_strength
from src.pipeline.extraction.evidence.item_extractor import (
    infer_confidence_interval,
    infer_effect_direction,
    infer_effect_size,
    infer_outcomes as infer_evidence_outcomes,
    infer_sample_size,
    infer_study_design,
)
from src.pipeline.extraction.pico.question_extractor import infer_comparator, infer_intervention, infer_outcomes, infer_population
from src.pipeline.extraction.recommendation.candidate_extractor import (
    infer_certainty,
    infer_direction,
    infer_strength,
    recommendation_code,
)
from src.pipeline.extraction.common.validators import (
    validate_evidence_item,
    validate_grade_assessment,
    validate_pico,
    validate_recommendation,
)
from src.pipeline.extraction.common.extractor_common import RuleTraceInput, build_rule_trace


JsonDict = Dict[str, Any]
PIPELINE_VERSION = "source_span_first_v1"

RECOMMENDATION_ANCHOR_RE = re.compile(
    r"\b(?:"
    r"Recommendation\s+\d+[A-Za-z.-]*\s*:|"
    r"Conditional recommendation\s+(?:for|against)\s*:|"
    r"Strong recommendation\s+(?:for|against)\s*:|"
    r"We\s+(?:recommend|suggest)\b|"
    r"suggests?\s+against\b|"
    r"recommend\s+against\b"
    r")",
    re.I,
)
PRIMARY_RECOMMENDATION_ANCHOR_RE = re.compile(
    r"\b(?:Recommendation\s+\d+[A-Za-z.-]*\s*:|Conditional recommendation\s+(?:for|against)\s*:|Strong recommendation\s+(?:for|against)\s*:)",
    re.I,
)
NEXT_BOUNDARY_RE = re.compile(
    r"\n\s*(?:Recommendation\s+\d+[A-Za-z.-]*\s*:|References\b|Bibliography\b|Authors?\b|Affiliations\b|Funding\b|Conflict of Interest\b)",
    re.I,
)
EVIDENCE_SENTENCE_RE = re.compile(
    r"[^.!?]*(?:TF identified|evidence showed|desirable effects|undesirable effects|overall certainty of evidence|"
    r"no direct evidence|RCTs?|articles?|systematic review|meta-analysis)[^.!?]*[.!?]",
    re.I,
)
GRADE_SENTENCE_RE = re.compile(
    r"[^.!?]*(?:(?:very\s+low|low|moderate|high)\s+certainty(?:\s+of\s+evidence)?|"
    r"certainty\s+of\s+evidence\s+was\s+(?:very\s+low|low|moderate|high)|"
    r"due\s+to\s+(?:risk of bias|imprecision|inconsistency|indirectness|publication bias))[^.!?]*[.!?]",
    re.I,
)
GRADE_REASON_FIELDS = {
    "risk_of_bias": re.compile(r"\brisk\s+of\s+bias\b", re.I),
    "inconsistency": re.compile(r"\binconsistency\b", re.I),
    "indirectness": re.compile(r"\bindirectness\b", re.I),
    "imprecision": re.compile(r"\bimprecision\b", re.I),
}
PUBLICATION_BIAS_RE = re.compile(r"\bpublication\s+bias\b", re.I)
LOW_VALUE_TAIL_RE = re.compile(r"(?im)^\s*(?:references|bibliography|authors?\s+and\s+affiliations|affiliations|funding|conflicts?\s+of\s+interest)\b")
PICO_RECOMMENDATION_RE = re.compile(
    r"\bIn\s+(?P<population>[^,.;]+),\s+[^.]*?\bsuggests?\s+the\s+use\s+of\s+(?P<intervention>[^.;]+?)\s+over\s+(?P<comparator>[^.;]+?)(?:\.|$)",
    re.I,
)


def _section_for_offset(sections: Iterable[JsonDict], offset: int, text: str) -> str:
    cursor = 0
    for section in sections:
        section_text = str(section.get("text") or "")
        start = text.find(section_text[:80], cursor) if section_text else -1
        if start < 0:
            continue
        end = start + len(section_text)
        cursor = end
        if start <= offset <= end:
            return str(section.get("section_name") or "Document")
    return "Document"


def extract_recommendation_spans(
    text: str,
    source_record_id: str = "",
    sections: Iterable[JsonDict] = (),
    raw_text: str = "",
) -> List[JsonDict]:
    clean, _references = split_references(str(text or ""))
    clean, _affiliations = split_affiliations(clean)
    low_value = LOW_VALUE_TAIL_RE.search(clean)
    if low_value:
        clean = clean[: low_value.start()].rstrip()
    spans: List[JsonDict] = []
    primary_matches = list(PRIMARY_RECOMMENDATION_ANCHOR_RE.finditer(clean))
    matches = primary_matches or list(RECOMMENDATION_ANCHOR_RE.finditer(clean))
    for index, match in enumerate(matches, start=1):
        next_anchor = matches[index].start() if index < len(matches) else len(clean)
        boundary = NEXT_BOUNDARY_RE.search(clean, match.end(), next_anchor)
        end = boundary.start() if boundary else next_anchor
        source_span = clean[match.start() : end].strip()
        if not source_span:
            continue
        code = recommendation_code(source_span) or f"Recommendation {index}"
        offsets = span_offset_fields(raw_text or clean, clean, match.start(), end)
        spans.append(
            {
                "span_id": f"SPAN-REC-{index:03d}",
                "span_type": "recommendation",
                "recommendation_code": code if str(code).lower().startswith("recommendation") else f"Recommendation {code}",
                "source_section": _section_for_offset(sections, match.start(), clean),
                "source_record_id": source_record_id,
                "source_span": source_span,
                "start_char": match.start(),
                "end_char": end,
                **offsets,
            }
        )
    return spans


def extract_recommendations(spans: Iterable[JsonDict], guideline_id: str = "") -> List[JsonDict]:
    recommendations: List[JsonDict] = []
    for span in spans:
        source_span = str(span.get("source_span") or "")
        rec_id = stable_id("recommendation", guideline_id, span.get("source_record_id"), span.get("recommendation_code"), source_span[:120])
        item = {
            "recommendation_version_id": stable_id("recommendation_version", rec_id, "v1"),
            "recommendation_id": rec_id,
            "recommendation_code": span.get("recommendation_code"),
            "guideline_id": guideline_id,
            "recommendation_text": source_span,
            "population": infer_population(source_span),
            "intervention": infer_intervention(source_span),
            "comparator": infer_comparator(source_span),
            "direction": normalize_direction(infer_direction(source_span)),
            "strength": normalize_strength(infer_strength(source_span)),
            "certainty": normalize_certainty(infer_certainty(source_span)),
            "remark": None,
            "rationale": "",
            "source_span": source_span,
            "source_section": span.get("source_section"),
            "source_record_id": span.get("source_record_id"),
            "start_char": span.get("start_char"),
            "end_char": span.get("end_char"),
            "clean_start_char": span.get("clean_start_char"),
            "clean_end_char": span.get("clean_end_char"),
            "raw_start_char": span.get("raw_start_char"),
            "raw_end_char": span.get("raw_end_char"),
            "offset_mapping": span.get("offset_mapping", {}),
            "extraction_method": "rule",
            "confidence": 0.85,
        }
        result = validate_recommendation(item)
        item["validation"] = result.to_dict()
        item["quality_score"] = result.quality_score
        recommendations.append(item)
    return recommendations


def extract_pico_questions(recommendations: Iterable[JsonDict], guideline_id: str = "") -> List[JsonDict]:
    picos: List[JsonDict] = []
    for rec in recommendations:
        source_span = str(rec.get("source_span") or "")
        pico_match = PICO_RECOMMENDATION_RE.search(source_span)
        population = pico_match.group("population").strip() if pico_match else infer_population(source_span)
        intervention = pico_match.group("intervention").strip() if pico_match else infer_intervention(source_span)
        comparator = pico_match.group("comparator").strip() if pico_match else infer_comparator(source_span)
        outcomes = infer_outcomes(source_span)
        pico = {
            "pico_id": stable_id("pico", rec.get("recommendation_id"), source_span[:120]),
            "recommendation_id": rec.get("recommendation_id"),
            "guideline_id": guideline_id,
            "clinical_question": source_span,
            "population": population,
            "intervention": intervention,
            "comparator": comparator,
            "outcomes": [
                {**outcome, "importance": outcome.get("importance") or "not_reported", "source": "recommendation", "source_span": source_span}
                for outcome in outcomes
            ],
            "source_span": source_span,
            "source_section": rec.get("source_section"),
            "source_record_id": rec.get("source_record_id"),
            "extraction_method": "rule",
            "confidence": 0.75,
        }
        result = validate_pico(pico)
        pico["validation"] = result.to_dict()
        pico["quality_score"] = result.quality_score
        picos.append(pico)
    return picos


def _sentences(pattern: re.Pattern[str], text: str) -> List[Tuple[str, int, int]]:
    return [(match.group(0).strip(), match.start(), match.end()) for match in pattern.finditer(text)]


def extract_evidence_items(
    text: str,
    recommendations: Iterable[JsonDict],
    picos: Iterable[JsonDict],
    source_record_id: str = "",
    raw_text: str = "",
) -> List[JsonDict]:
    recs = list(recommendations)
    pico_by_rec = {str(pico.get("recommendation_id")): pico for pico in picos}
    evidence_items: List[JsonDict] = []
    for index, (sentence, start, end) in enumerate(_sentences(EVIDENCE_SENTENCE_RE, text), start=1):
        rec = recs[0] if recs else {}
        pico = pico_by_rec.get(str(rec.get("recommendation_id"))) or {}
        study_design = infer_study_design(sentence)
        evidence_type = "meta_analysis" if study_design == "meta_analysis" else "systematic_review" if study_design == "systematic_review" else "evidence_summary"
        item = {
            "evidence_id": stable_id("evidence", source_record_id, index, sentence[:120]),
            "source_record_id": source_record_id,
            "paper_id": None,
            "pico_id": pico.get("pico_id"),
            "recommendation_id": rec.get("recommendation_id"),
            "evidence_type": evidence_type,
            "study_design": study_design,
            "study_count": _count_before(sentence, "RCT"),
            "article_count": _count_before(sentence, "article"),
            "sample_size": infer_sample_size(sentence),
            "population_extracted": pico.get("population") or "",
            "intervention_extracted": pico.get("intervention") or "",
            "comparator_extracted": pico.get("comparator"),
            "outcomes_extracted": infer_evidence_outcomes(sentence),
            "effect_direction": infer_effect_direction(sentence),
            "benefits": sentence if infer_effect_direction(sentence) == "benefit" else "",
            "harms": sentence if infer_effect_direction(sentence) == "harm" else "",
            "effect_size": infer_effect_size(sentence) or None,
            "confidence_interval": infer_confidence_interval(sentence),
            "source_span": sentence,
            "source_section": "Evidence",
            "start_char": start,
            "end_char": end,
            **span_offset_fields(raw_text or text, text, start, end),
            "extraction_method": "rule",
            "confidence": 0.75,
        }
        result = validate_evidence_item(item)
        item["validation"] = result.to_dict()
        item["quality_score"] = result.quality_score
        evidence_items.append(item)
    return evidence_items


def _count_before(text: str, noun: str) -> int | None:
    numbers = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    pattern = re.compile(rf"\b(?P<n>\d+|{'|'.join(numbers)})\s+{re.escape(noun)}s?\b", re.I)
    match = pattern.search(text)
    if not match:
        return None
    value = match.group("n").lower()
    return int(value) if value.isdigit() else numbers.get(value)


def extract_grade_assessments(
    text: str,
    recommendations: Iterable[JsonDict],
    picos: Iterable[JsonDict],
    raw_text: str = "",
) -> List[JsonDict]:
    recs = list(recommendations)
    pico_by_rec = {str(pico.get("recommendation_id")): pico for pico in picos}
    grades: List[JsonDict] = []
    for index, (sentence, start, end) in enumerate(_sentences(GRADE_SENTENCE_RE, text), start=1):
        rec = recs[0] if recs else {}
        pico = pico_by_rec.get(str(rec.get("recommendation_id"))) or {}
        grade = {
            "grade_id": stable_id("grade", rec.get("recommendation_id"), index, sentence[:120]),
            "recommendation_id": rec.get("recommendation_id"),
            "pico_id": pico.get("pico_id"),
            "outcome_name": "overall",
            "risk_of_bias": "serious_or_concern" if GRADE_REASON_FIELDS["risk_of_bias"].search(sentence) else "not_reported",
            "inconsistency": "serious_or_concern" if GRADE_REASON_FIELDS["inconsistency"].search(sentence) else "not_reported",
            "indirectness": "serious_or_concern" if GRADE_REASON_FIELDS["indirectness"].search(sentence) else "not_reported",
            "imprecision": "serious_or_concern" if GRADE_REASON_FIELDS["imprecision"].search(sentence) else "not_reported",
            "publication_bias": "suspected" if PUBLICATION_BIAS_RE.search(sentence) else "not_reported",
            "final_certainty": normalize_certainty(infer_certainty(sentence)),
            "judgement_rationale": sentence,
            "source_span": sentence,
            "source_section": "Evidence",
            "start_char": start,
            "end_char": end,
            **span_offset_fields(raw_text or text, text, start, end),
            "extraction_method": "rule",
            "confidence": 0.8,
        }
        result = validate_grade_assessment(grade)
        grade["validation"] = result.to_dict()
        grade["quality_score"] = result.quality_score
        grades.append(grade)
    return grades


def build_model_trace(
    *,
    source_record_id: str,
    task_type: str,
    raw_input: Any,
    parsed_output: Any,
    target_table: str,
    target_entity_id: str = "",
) -> JsonDict:
    """为 source-span-first 抽取器构造一条合成规则 trace。

    这条兼容路径不调用模型；trace 只是为了用和后续抽取器一致的结构
    保留血缘信息和目标表元数据。
    """

    block = {"block_id": source_record_id, "text": raw_input, "candidate_hints": []}
    return build_rule_trace(
        RuleTraceInput(
            rule_version=PIPELINE_VERSION,
            task_type=task_type,
            block=block,
            parsed_output={"output": parsed_output},
            confidence=0.0,
            parameters={"temperature": 0},
            elapsed_ms=0,
            target_table=target_table,
            target_entity_id=target_entity_id,
        )
    ).to_dict()


def _guideline_id(cleaned: JsonDict) -> str:
    return str((cleaned.get("guideline_seed") or {}).get("guideline_id") or "")


def _model_traces(record_id: str, clean_content: str, recommendations: List[JsonDict], picos: List[JsonDict], evidence: List[JsonDict], grades: List[JsonDict]) -> List[JsonDict]:
    """为每一类抽取实体生成一条合成 trace。"""

    # source-span-first 是规则抽取流程，这里的 trace 用来保留血缘和目标表，不代表真实模型耗时。
    return [
        build_model_trace(source_record_id=record_id, task_type="recommendation_extraction", raw_input=clean_content, parsed_output=recommendations, target_table="recommendation_versions"),
        build_model_trace(source_record_id=record_id, task_type="pico_extraction", raw_input=[rec.get("source_span") for rec in recommendations], parsed_output=picos, target_table="pico_questions"),
        build_model_trace(source_record_id=record_id, task_type="evidence_extraction", raw_input=clean_content, parsed_output=evidence, target_table="evidence_items"),
        build_model_trace(source_record_id=record_id, task_type="grade_extraction", raw_input=clean_content, parsed_output=grades, target_table="grade_assessments"),
    ]


def _extract_entities(cleaned: JsonDict, clean_content: str, sections: Iterable[JsonDict]) -> tuple[JsonDict, List[JsonDict]]:
    record_id = str(cleaned.get("record_id") or "")
    guideline_id = _guideline_id(cleaned)
    raw_content = str(cleaned.get("raw_content") or clean_content)
    # 先定位原文 span，再派生 recommendation/PICO/evidence/GRADE，避免后续实体丢失原文坐标。
    spans = extract_recommendation_spans(clean_content, source_record_id=record_id, sections=sections, raw_text=raw_content)
    recommendations = extract_recommendations(spans, guideline_id=guideline_id)
    picos = extract_pico_questions(recommendations, guideline_id=guideline_id)
    evidence = extract_evidence_items(clean_content, recommendations, picos, source_record_id=record_id, raw_text=raw_content)
    grades = extract_grade_assessments(clean_content, recommendations, picos, raw_text=raw_content)
    return {
        "recommendation_versions": recommendations,
        "pico_questions": picos,
        "evidence_items": evidence,
        "grade_assessments": grades,
        "model_traces": _model_traces(record_id, clean_content, recommendations, picos, evidence, grades),
        "update_logs": [],
    }, spans


def _entity_id(row: JsonDict) -> object:
    return row.get("recommendation_id") or row.get("pico_id") or row.get("evidence_id") or row.get("grade_id")


def _validation_summary(entities: JsonDict) -> JsonDict:
    pending: List[JsonDict] = []
    valid: List[JsonDict] = []
    warnings: List[JsonDict] = []
    for entity_type, rows in entities.items():
        if entity_type == "model_traces":
            continue
        for row in rows:
            validation = row.get("validation") or {}
            # 校验失败的实体仍保留在输出中，交给人工或后续修复流程处理。
            bucket = valid if validation.get("is_valid") else pending
            bucket.append({"entity_type": entity_type, "entity_id": _entity_id(row)})
            if not validation.get("is_valid"):
                warnings.append({"entity_type": entity_type, "warnings": validation.get("warnings", [])})
    return {"valid_entities": valid, "pending_entities": pending, "warnings": warnings}


def _source_record_summary(cleaned: JsonDict, record_id: str) -> JsonDict:
    return {
        "record_id": record_id,
        "title": cleaned.get("title", ""),
        "url": cleaned.get("url", ""),
        "doi": cleaned.get("doi", ""),
        "source": cleaned.get("source", ""),
        "published_year": cleaned.get("published_year", ""),
        "raw_pdf_path": cleaned.get("raw_pdf_path", ""),
    }


def _cleaning_result(record: JsonDict, cleaned: JsonDict, clean_content: str, sections: Iterable[JsonDict]) -> JsonDict:
    return {
        "raw_content": cleaned.get("raw_content", ""),
        "clean_content": clean_content,
        "sections": sections,
        "tables": cleaned.get("tables") or extract_tables_from_text(light_clean_pdf_text(str(record.get("content") or ""))),
        "references_text": cleaned.get("references_text", ""),
        "affiliations_text": cleaned.get("affiliations_text", ""),
        "cleaning_log": cleaned.get("cleaning_log", {}),
        "cleaning_warnings": cleaned.get("cleaning_warnings", []),
    }


def _candidate_span_summary(spans: List[JsonDict], entities: JsonDict) -> JsonDict:
    recommendations = entities["recommendation_versions"]
    evidence = entities["evidence_items"]
    grades = entities["grade_assessments"]
    return {
        "recommendation_spans": spans,
        "pico_spans": [{"source": "recommendation", "source_span": rec.get("source_span"), "recommendation_id": rec.get("recommendation_id")} for rec in recommendations],
        "evidence_spans": [
            {
                "source_span": item.get("source_span"),
                "start_char": item.get("start_char"),
                "end_char": item.get("end_char"),
                "raw_start_char": item.get("raw_start_char"),
                "raw_end_char": item.get("raw_end_char"),
                "offset_mapping": item.get("offset_mapping", {}),
            }
            for item in evidence
        ],
        "grade_spans": [
            {
                "source_span": item.get("source_span"),
                "start_char": item.get("start_char"),
                "end_char": item.get("end_char"),
                "raw_start_char": item.get("raw_start_char"),
                "raw_end_char": item.get("raw_end_char"),
                "offset_mapping": item.get("offset_mapping", {}),
            }
            for item in grades
        ],
    }


def run_source_span_first_pipeline(record: JsonDict) -> JsonDict:
    cleaned = clean_source_record(record)
    record_id = str(cleaned.get("record_id") or "")
    clean_content = str(cleaned.get("clean_content") or cleaned.get("content") or "")
    sections = cleaned.get("sections") or split_sections(clean_content)
    entities, spans = _extract_entities(cleaned, clean_content, sections)
    return {
        "source_record": _source_record_summary(cleaned, record_id),
        "cleaning_result": _cleaning_result(record, cleaned, clean_content, sections),
        "candidate_spans": _candidate_span_summary(spans, entities),
        "entities": entities,
        "validation": _validation_summary(entities),
        "lineage": {
            "source_record_id": record_id,
            "pipeline_version": PIPELINE_VERSION,
            "processed_at": utc_now(),
            "processor": "src.pipeline.extraction.source_span.first_pass",
        },
    }
