from __future__ import annotations

import json

from src.domain.common import stable_id
from src.common.extraction_common import JsonDict, utc_now


DIRECTION_VALUES = {"for", "against", "neutral", "no_recommendation", "unclear"}
STRENGTH_VALUES = {"strong", "conditional", "weak", "good_practice", "none", "unclear"}
CERTAINTY_VALUES = {"high", "moderate", "low", "very_low", "none", "unclear"}
GRADE_SYSTEM_VALUES = {"GRADE", "COR_LOE", "LETTER_GRADE", "NUMERIC_LETTER_GRADE", "VERB_BASED", "unknown"}
GRADE_DOMAIN_VALUES = {"no_concern", "serious", "very_serious", "unclear", "not_extracted"}
PUBLICATION_BIAS_VALUES = {"undetected", "suspected", "strongly_suspected", "unclear", "not_extracted"}

RECOMMENDATION_PROMPT_POLICY = {
    "valid_source_roles": [
        "recommendations",
        "key_recommendations",
        "summary_of_recommendations",
        "guidance",
    ],
    "valid_block_types": ["paragraph", "list_item", "recommendation_box", "unknown"],
    "valid_quality_statuses": [
        "trusted_structured_html",
        "trusted_pdf_text",
        "trusted_layout_ocr",
        "unknown",
    ],
    "reject_text_types": [
        "navigation_or_toc_text",
        "methodology_or_wording_template",
        "research_recommendation",
        "grade_table_or_legend",
        "evidence_summary_or_study_result",
        "table_or_figure_text",
        "header_footer_page_number_reference",
        "ocr_corruption_or_column_interleaving",
    ],
    "reject_examples": [
        "Return to recommendations",
        "Grade for quality of evidence",
        "The committee made a recommendation for research",
        "We use the word recommend",
        "The evidence showed",
        "studies reported",
    ],
}

RECOMMENDATION_RESULT_SCHEMA = {
    "queue_id": "string",
    "recommendation_candidate_id": "string",
    "is_valid_recommendation": "boolean",
    "corrected_recommendation_text": "string|null",
    "direction": "for|against|neutral|no_recommendation|unclear",
    "strength": "strong|conditional|weak|good_practice|none|unclear",
    "certainty": "high|moderate|low|very_low|none|unclear",
    "population": "string|null",
    "intervention": "string|null",
    "comparator": "string|null",
    "outcomes": "list",
    "rationale": "string|null",
    "remarks": "string|null",
    "reject_reason": "string|null",
    "needs_human_review": "boolean",
    "confidence": "number between 0 and 1",
}

RECOMMENDATION_OUTPUT_TEMPLATE = {
    "queue_id": "<copy queue_id exactly>",
    "recommendation_candidate_id": "<copy recommendation_candidate_id exactly>",
    "is_valid_recommendation": "<boolean>",
    "corrected_recommendation_text": "<string or null>",
    "direction": "<for|against|neutral|no_recommendation|unclear>",
    "strength": "<strong|conditional|weak|good_practice|none|unclear>",
    "certainty": "<high|moderate|low|very_low|none|unclear>",
    "population": "<string or null>",
    "intervention": "<string or null>",
    "comparator": "<string or null>",
    "outcomes": "<list>",
    "rationale": "<string or null>",
    "remarks": "<string or null>",
    "reject_reason": "<string or null>",
    "needs_human_review": "<boolean>",
    "confidence": "<number between 0 and 1>",
}

GRADE_RESULT_SCHEMA = {
    "queue_id": "string",
    "grade_candidate_id": "string",
    "recommendation_candidate_id": "string|null",
    "is_valid_grade": "boolean",
    "grade_system": "GRADE|COR_LOE|LETTER_GRADE|NUMERIC_LETTER_GRADE|VERB_BASED|unknown",
    "certainty": "high|moderate|low|very_low|none|unclear",
    "strength": "strong|conditional|weak|good_practice|none|unclear",
    "risk_of_bias": "no_concern|serious|very_serious|unclear|not_extracted",
    "inconsistency": "no_concern|serious|very_serious|unclear|not_extracted",
    "indirectness": "no_concern|serious|very_serious|unclear|not_extracted",
    "imprecision": "no_concern|serious|very_serious|unclear|not_extracted",
    "publication_bias": "undetected|suspected|strongly_suspected|unclear|not_extracted",
    "reasons_for_downgrade": "list[str]",
    "reasons_for_upgrade": "list[str]",
    "reject_reason": "string|null",
    "needs_human_review": "boolean",
    "confidence": "number between 0 and 1",
}

GRADE_OUTPUT_TEMPLATE = {
    "queue_id": "<copy queue_id exactly>",
    "grade_candidate_id": "<copy grade_candidate_id exactly>",
    "recommendation_candidate_id": "<copy recommendation_candidate_id or null>",
    "is_valid_grade": "<boolean>",
    "grade_system": "<GRADE|COR_LOE|LETTER_GRADE|NUMERIC_LETTER_GRADE|VERB_BASED|unknown>",
    "certainty": "<high|moderate|low|very_low|none|unclear>",
    "strength": "<strong|conditional|weak|good_practice|none|unclear>",
    "risk_of_bias": "<no_concern|serious|very_serious|unclear|not_extracted>",
    "inconsistency": "<no_concern|serious|very_serious|unclear|not_extracted>",
    "indirectness": "<no_concern|serious|very_serious|unclear|not_extracted>",
    "imprecision": "<no_concern|serious|very_serious|unclear|not_extracted>",
    "publication_bias": "<undetected|suspected|strongly_suspected|unclear|not_extracted>",
    "reasons_for_downgrade": "<list[str]>",
    "reasons_for_upgrade": "<list[str]>",
    "reject_reason": "<string or null>",
    "needs_human_review": "<boolean>",
    "confidence": "<number between 0 and 1>",
}


def compact_json(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)


def _payload(row: JsonDict) -> JsonDict:
    payload = row.get("normalized_payload")
    return payload if isinstance(payload, dict) else {}


def _raw_payload(row: JsonDict) -> JsonDict:
    payload = row.get("raw_payload")
    return payload if isinstance(payload, dict) else {}


def compact_source_provenance(row: JsonDict) -> JsonDict:
    payload = _payload(row)
    raw_payload = _raw_payload(row)
    source_metadata = payload.get("source_metadata")
    source_metadata = source_metadata if isinstance(source_metadata, dict) else {}
    quality = raw_payload.get("quality")
    quality = quality if isinstance(quality, dict) else {}
    statement_evaluation = payload.get("statement_evaluation")
    statement_evaluation = statement_evaluation if isinstance(statement_evaluation, dict) else {}
    return {
        "record_id": row.get("record_id", ""),
        "guideline_id": row.get("guideline_id"),
        "source_url": row.get("source_url"),
        "source_section": row.get("source_section"),
        "block_id": payload.get("block_id") or payload.get("source_block_id") or raw_payload.get("block_id"),
        "section_path": payload.get("section_path", []),
        "source_order": payload.get("source_order"),
        "statement_index": payload.get("statement_index"),
        "page": source_metadata.get("page") or payload.get("page") or row.get("page"),
        "bbox": source_metadata.get("bbox") or payload.get("bbox") or row.get("bbox"),
        "section_role": payload.get("section_role") or source_metadata.get("section_role") or "unknown",
        "block_type": payload.get("block_type") or source_metadata.get("block_type") or "unknown",
        "quality_status": payload.get("quality_status") or quality.get("quality_status") or "unknown",
        "text_quality_score": payload.get("text_quality_score") or quality.get("text_quality_score"),
        "quality_notes": payload.get("quality_notes", []),
        "noise_reasons": statement_evaluation.get("noise_reasons", []),
    }


def compact_recommendation_state(rec: JsonDict) -> JsonDict:
    payload = _payload(rec)
    return {
        "candidate_id": rec.get("candidate_id", ""),
        "recommendation_text": rec.get("recommendation_text", ""),
        "recommendation_code": rec.get("recommendation_code"),
        "direction": rec.get("direction", "unclear"),
        "strength": rec.get("strength", "unclear"),
        "certainty": rec.get("certainty", "unclear"),
        "population": rec.get("population"),
        "intervention": rec.get("intervention"),
        "comparator": rec.get("comparator"),
        "outcomes": rec.get("outcomes", []),
        "rationale": rec.get("rationale"),
        "remarks": rec.get("remarks"),
        "status": rec.get("status", ""),
        "review_note": rec.get("review_note"),
        "source_url": rec.get("source_url"),
        "source_section": rec.get("source_section"),
        "quality_notes": payload.get("quality_notes", []),
        "source_provenance": compact_source_provenance(rec),
        "source_order": payload.get("source_order"),
        "statement_index": payload.get("statement_index"),
        "source_metadata": payload.get("source_metadata", {}),
    }


def compact_grade_state(grade: JsonDict) -> JsonDict:
    payload = _payload(grade)
    return {
        "grade_candidate_id": grade.get("grade_candidate_id", ""),
        "recommendation_candidate_id": grade.get("recommendation_candidate_id", ""),
        "grade_system": grade.get("grade_system", "unknown"),
        "certainty": grade.get("certainty", "unclear"),
        "strength": grade.get("strength", "unclear"),
        "risk_of_bias": grade.get("risk_of_bias", "not_extracted"),
        "inconsistency": grade.get("inconsistency", "not_extracted"),
        "indirectness": grade.get("indirectness", "not_extracted"),
        "imprecision": grade.get("imprecision", "not_extracted"),
        "publication_bias": grade.get("publication_bias", "not_extracted"),
        "reasons_for_downgrade": grade.get("reasons_for_downgrade", []),
        "reasons_for_upgrade": grade.get("reasons_for_upgrade", []),
        "status": grade.get("status", ""),
        "review_note": grade.get("review_note"),
        "source_section": grade.get("source_section"),
        "source_url": grade.get("source_url"),
        "association_reason": payload.get("association_reason"),
        "source_order": payload.get("source_order"),
        "source_metadata": payload.get("source_metadata", {}),
    }


def compact_candidate_state(item: JsonDict) -> JsonDict:
    state = item.get("current_candidate_state")
    state = state if isinstance(state, dict) else {}
    if item.get("task_type") == "recommendation_candidate_review":
        rec = state.get("recommendation")
        linked_grades = state.get("linked_grades")
        linked_grades = linked_grades if isinstance(linked_grades, list) else []
        return {
            "recommendation": compact_recommendation_state(rec if isinstance(rec, dict) else {}),
            "linked_grades": [compact_grade_state(grade) for grade in linked_grades if isinstance(grade, dict)],
        }
    if item.get("task_type") == "grade_candidate_review":
        grade = state.get("grade")
        return {
            "grade": compact_grade_state(grade if isinstance(grade, dict) else {}),
        }
    return state


def _queue_recommendation(item: JsonDict) -> JsonDict:
    state = item.get("current_candidate_state")
    state = state if isinstance(state, dict) else {}
    rec = state.get("recommendation")
    return rec if isinstance(rec, dict) else {}


def build_recommendation_prompt(item: JsonDict) -> str:
    output_template = dict(RECOMMENDATION_OUTPUT_TEMPLATE)
    output_template["queue_id"] = item.get("queue_id", "")
    output_template["recommendation_candidate_id"] = item.get("recommendation_candidate_id", "")
    current_candidate_state = compact_candidate_state(item)
    payload = {
        "task": "review_and_enhance_recommendation_candidate",
        "queue_id": item.get("queue_id", ""),
        "priority": item.get("priority", ""),
        "review_reasons": item.get("review_reasons", []),
        "review_policy": RECOMMENDATION_PROMPT_POLICY,
        "source_section": item.get("source_section", ""),
        "source_text": item.get("source_text", ""),
        "source_provenance": compact_source_provenance(_queue_recommendation(item)),
        "current_candidate_state": current_candidate_state,
        "required_output_schema": RECOMMENDATION_RESULT_SCHEMA,
        "output_template": output_template,
    }
    return (
        "You are reviewing a clinical guideline recommendation candidate.\n"
        "Use only the provided source text and candidate state. Do not invent evidence.\n"
        "A valid recommendation must be a formal clinical action statement from a trusted recommendation/guidance "
        "source span, with traceable provenance such as block_id, page/bbox, section_path, source_section, or source_url.\n"
        "Reject or require human review when provenance is missing, the source role is not a recommendation/guidance "
        "section, or quality notes indicate OCR corruption, column interleaving, table/figure contamination, or layout damage.\n"
        "Set is_valid_recommendation=false for navigation text, methodology or wording templates, research-only "
        "recommendations, GRADE legends/tables, evidence summaries, study results, headers, footers, page numbers, "
        "references, table cells/captions, and figure captions. Use reject_reason to name the specific source-quality "
        "or artifact issue.\n"
        "Do not treat the words recommend, suggest, evidence suggests, or recommendation for research as sufficient "
        "proof of a formal clinical recommendation.\n"
        "Return one strict JSON object only. Do not include markdown or explanatory text.\n"
        "The returned top-level JSON object must match output_template exactly by field names.\n"
        "Do not echo the input object. Do not return current_candidate_state, source_text, required_output_schema, "
        "review_reasons, priority, or task as top-level fields.\n"
        "Use null when text is absent, unclear when a categorical judgement cannot be made, "
        "and needs_human_review=true when the source is ambiguous.\n\n"
        f"INPUT_JSON:\n{compact_json(payload)}\n\n"
        "Now return OUTPUT_JSON only. Replace every placeholder in output_template with your judgement. "
        "Do not copy placeholder strings into the answer."
    )


def build_grade_prompt(item: JsonDict) -> str:
    grade_ids = item.get("grade_candidate_ids")
    expected_grade_id = grade_ids[0] if isinstance(grade_ids, list) and grade_ids else ""
    output_template = dict(GRADE_OUTPUT_TEMPLATE)
    output_template["queue_id"] = item.get("queue_id", "")
    output_template["grade_candidate_id"] = expected_grade_id
    output_template["recommendation_candidate_id"] = item.get("recommendation_candidate_id") or None
    payload = {
        "task": "review_and_enhance_grade_candidate",
        "queue_id": item.get("queue_id", ""),
        "priority": item.get("priority", ""),
        "review_reasons": item.get("review_reasons", []),
        "source_section": item.get("source_section", ""),
        "source_text": item.get("source_text", ""),
        "current_candidate_state": compact_candidate_state(item),
        "required_output_schema": GRADE_RESULT_SCHEMA,
        "output_template": output_template,
    }
    return (
        "You are reviewing a clinical guideline GRADE or evidence-certainty candidate.\n"
        "Use only the provided source text and candidate state. Do not infer unavailable GRADE domains.\n"
        "Return one strict JSON object only. Do not include markdown or explanatory text.\n"
        "The returned top-level JSON object must match output_template exactly by field names.\n"
        "Do not echo the input object. Do not return current_candidate_state, source_text, required_output_schema, "
        "review_reasons, priority, or task as top-level fields.\n"
        "Use not_extracted for GRADE domains that are not explicitly stated, unclear when a stated "
        "judgement is ambiguous, and needs_human_review=true when association is uncertain.\n\n"
        f"INPUT_JSON:\n{compact_json(payload)}\n\n"
        "Now return OUTPUT_JSON only. Replace every placeholder in output_template with your judgement. "
        "Do not copy placeholder strings into the answer."
    )


def build_prompt(item: JsonDict) -> str:
    task_type = item.get("task_type")
    if task_type == "recommendation_candidate_review":
        return build_recommendation_prompt(item)
    if task_type == "grade_candidate_review":
        return build_grade_prompt(item)
    raise ValueError(f"Unsupported LLM queue task_type: {task_type}")


def build_prompt_record(item: JsonDict, prompt_version: str) -> JsonDict:
    prompt = build_prompt(item)
    prompt_id = stable_id("llm_prompt", item.get("queue_id"), prompt_version)
    return {
        "prompt_id": prompt_id,
        "queue_id": item.get("queue_id", ""),
        "task_type": item.get("task_type", ""),
        "priority": item.get("priority", ""),
        "record_id": item.get("record_id", ""),
        "guideline_id": item.get("guideline_id", ""),
        "recommendation_candidate_id": item.get("recommendation_candidate_id", ""),
        "grade_candidate_ids": item.get("grade_candidate_ids", []),
        "model_trace_ids": item.get("model_trace_ids", []),
        "prompt_version": prompt_version,
        "prompt": prompt,
        "status": "pending_model_call",
        "created_at": utc_now(),
    }
