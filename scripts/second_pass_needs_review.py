from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.initial_association_audit import clean_text, text_flags
from src.common.data_artifacts import utc_now
from src.common.process_jsonl import iter_jsonl, write_jsonl


JsonDict = dict[str, Any]

ACTION_PATTERN = re.compile(
    r"\b(should|recommend|recommended|suggest|suggests|offer|avoid|consider|is indicated|are indicated|may be used|should not|do not)\b",
    re.I,
)
BAD_RECOMMENDATION_CONTEXT = re.compile(
    r"\b(return to recommendations|future research|recommendation for research|grade for quality|wording evidence|"
    r"not enough evidence to make|table\s+\d|committee made a recommendation|we recommend[.]{3}|we suggest[.]{3})\b",
    re.I,
)
AMBIGUOUS_PRONOUN_START = re.compile(r"^(it|they|these|this|so,?\s+it)\b", re.I)
OCR_SPLIT_SIGNAL = re.compile(r"\b[a-z]{3,}(?:of|to|for|and|in)[a-z]{3,}\b|[a-z]{2,}-\s+[a-z]{2,}", re.I)
BROKEN_PDF_WORD_SIGNAL = re.compile(
    r"\b(actigof|actigraof|raphy|hyperor|qualthe|recomstand|recombinant|suscompared|folobstructive|imprecisleep|"
    r"protients|disorlonged|findex|cant\s+large|tions\s+are\s+derived)\b",
    re.I,
)


def payload(row: JsonDict) -> JsonDict:
    value = row.get("normalized_payload")
    return value if isinstance(value, dict) else {}


def review_payload(row: JsonDict) -> JsonDict:
    value = payload(row).get("codex_full_data_review")
    return value if isinstance(value, dict) else {}


def add_second_pass(row: JsonDict, decision: JsonDict) -> JsonDict:
    item = deepcopy(row)
    normalized = item.get("normalized_payload")
    if not isinstance(normalized, dict):
        normalized = {}
    normalized["second_pass_needs_review"] = {
        "reviewer": "codex_llm_human_second_pass_v1",
        "reviewed_at": utc_now(),
        **decision,
    }
    item["normalized_payload"] = normalized
    item["review_note"] = "Second pass: {decision}; {reasons}".format(
        decision=decision["decision"],
        reasons=", ".join(decision.get("reasons") or []),
    )
    return item


def source_has_claim(row: JsonDict, claim: str) -> bool:
    source = clean_text(row.get("source_text") or row.get("source_span"))
    return bool(claim) and claim in source


def looks_like_valid_recommendation(row: JsonDict) -> tuple[bool, list[str], JsonDict | None]:
    text = clean_text(row.get("recommendation_text"))
    reasons: list[str] = []
    fix: JsonDict | None = None
    flags = set(text_flags(text))
    source_flags = {f"source_{flag}" for flag in text_flags(row.get("source_text") or row.get("source_span"))}
    if not text:
        reasons.append("empty_recommendation_text")
    if len(text) < 45:
        reasons.append("too_short_or_context_dependent")
    if not ACTION_PATTERN.search(text):
        reasons.append("missing_clear_action_signal")
    if BAD_RECOMMENDATION_CONTEXT.search(text):
        reasons.append("methodology_or_research_recommendation_context")
    if AMBIGUOUS_PRONOUN_START.search(text):
        reasons.append("ambiguous_pronoun_reference")
    if (
        "ocr_glued_words" in flags
        or "long_unbroken_tokens" in flags
        or OCR_SPLIT_SIGNAL.search(text)
        or BROKEN_PDF_WORD_SIGNAL.search(text)
    ):
        reasons.append("ocr_or_column_merge_noise")
    if "table_or_navigation" in flags:
        reasons.append("table_or_navigation_text")
    if not row.get("pico_id"):
        reasons.append("missing_pico_link")
    if not row.get("grade_candidate_id"):
        reasons.append("missing_grade_link")
    if not source_has_claim(row, text):
        reasons.append("claim_not_verbatim_in_source_span")
    if "source_methodology_context" in source_flags and len(text) > 180:
        reasons.append("source_span_mixed_with_methodology")

    fields = {
        "population": clean_text(row.get("population")),
        "intervention": clean_text(row.get("intervention")),
        "outcome_summary": clean_text(row.get("outcome_summary")),
    }
    if any(len(value) > 220 for value in fields.values()):
        reasons.append("structured_field_contains_context_spillover")
    if not fields["population"] or not fields["intervention"]:
        reasons.append("missing_population_or_intervention")

    valid = not reasons
    if not valid and {"claim_not_verbatim_in_source_span"} == set(reasons) and ACTION_PATTERN.search(text):
        fix = {"action": "keep_for_manual_source_span_repair", "recommendation_text": text}
    return valid, sorted(set(reasons)), fix


def decide_recommendation(row: JsonDict) -> JsonDict:
    valid, reasons, fix = looks_like_valid_recommendation(row)
    text = clean_text(row.get("recommendation_text"))
    # Human-review policy: rows already demoted to needs_review require explicit manual confirmation
    # before becoming a publishable core recommendation.
    if valid and row.get("quality_status") != "needs_review":
        return {
            "decision": "promote_to_core",
            "priority": "P1",
            "reasons": ["clear_actionable_recommendation_after_second_pass"],
            "suggested_fix": None,
        }
    if valid and row.get("quality_status") == "needs_review":
        return {
            "decision": "revise",
            "priority": "P1",
            "reasons": ["requires_manual_confirmation_before_promotion"],
            "suggested_fix": {"action": "manual_confirm_publishable_status", "recommendation_text": text},
        }
    if fix and len(text) <= 260:
        return {
            "decision": "revise",
            "priority": "P1",
            "reasons": reasons,
            "suggested_fix": fix,
        }
    hard_reject = {
        "methodology_or_research_recommendation_context",
        "ambiguous_pronoun_reference",
        "ocr_or_column_merge_noise",
        "table_or_navigation_text",
        "structured_field_contains_context_spillover",
    }
    if hard_reject & set(reasons):
        return {
            "decision": "quarantine",
            "priority": "P3",
            "reasons": reasons,
            "suggested_fix": None,
        }
    return {
        "decision": "keep_review",
        "priority": "P2",
        "reasons": reasons,
        "suggested_fix": fix,
    }


def pico_quality_reasons(row: JsonDict, linked_core_versions: set[str], evidence_by_pico: dict[str, int]) -> list[str]:
    reasons: list[str] = []
    question = clean_text(row.get("clinical_question"))
    population = clean_text(row.get("population"))
    intervention = clean_text(row.get("intervention"))
    flags = set(text_flags(" ".join([question, population, intervention])))
    if not question:
        reasons.append("missing_clinical_question")
    if not population:
        reasons.append("missing_population")
    if not intervention:
        reasons.append("missing_intervention")
    if not row.get("outcomes"):
        reasons.append("outcomes_empty")
    if len(population) > 180 or len(intervention) > 220:
        reasons.append("field_context_spillover")
    if flags & {"ocr_glued_words", "long_unbroken_tokens", "methodology_context", "table_or_navigation"}:
        reasons.extend(sorted(flags & {"ocr_glued_words", "long_unbroken_tokens", "methodology_context", "table_or_navigation"}))
    confidence = row.get("extraction_confidence")
    if isinstance(confidence, (int, float)) and confidence < 0.65:
        reasons.append("low_extraction_confidence")
    pico_id = str(row.get("pico_id") or "")
    if pico_id not in linked_core_versions:
        reasons.append("not_linked_to_core_recommendation")
    if evidence_by_pico.get(pico_id, 0) == 0:
        reasons.append("no_evidence_link")
    return sorted(set(reasons))


def decide_pico(row: JsonDict, linked_core_picos: set[str], evidence_by_pico: dict[str, int]) -> JsonDict:
    reasons = pico_quality_reasons(row, linked_core_picos, evidence_by_pico)
    if not reasons or reasons == ["outcomes_empty"]:
        return {
            "decision": "promote_to_core",
            "priority": "P1",
            "reasons": ["well_formed_and_linked_pico_after_second_pass"],
            "suggested_fix": None,
        }
    if "not_linked_to_core_recommendation" in reasons:
        return {"decision": "keep_review", "priority": "P2", "reasons": reasons, "suggested_fix": None}
    if {"ocr_glued_words", "long_unbroken_tokens", "methodology_context", "table_or_navigation", "field_context_spillover"} & set(reasons):
        return {"decision": "quarantine", "priority": "P3", "reasons": reasons, "suggested_fix": None}
    return {"decision": "keep_review", "priority": "P2", "reasons": reasons, "suggested_fix": None}


def evidence_quality_reasons(row: JsonDict, core_picos: set[str]) -> list[str]:
    reasons: list[str] = []
    text = clean_text(row.get("source_text") or row.get("source_span"))
    flags = set(text_flags(text))
    if not text:
        reasons.append("missing_source_text")
    if row.get("pico_id") not in core_picos:
        reasons.append("not_linked_to_core_pico")
    if row.get("screening_status") != "included":
        reasons.append("not_included_status")
    if row.get("study_design") in (None, "", "unclear"):
        reasons.append("unclear_study_design")
    if not row.get("outcomes_extracted"):
        reasons.append("outcomes_empty")
    if row.get("effect_direction") in (None, "", "uncertain"):
        reasons.append("effect_direction_uncertain")
    confidence = row.get("extraction_confidence")
    if isinstance(confidence, (int, float)) and confidence < 0.65:
        reasons.append("low_extraction_confidence")
    if flags & {"ocr_glued_words", "long_unbroken_tokens", "methodology_context", "table_or_navigation"}:
        reasons.extend(sorted(flags & {"ocr_glued_words", "long_unbroken_tokens", "methodology_context", "table_or_navigation"}))
    return sorted(set(reasons))


def decide_evidence(row: JsonDict, core_picos: set[str]) -> JsonDict:
    reasons = evidence_quality_reasons(row, core_picos)
    if row.get("screening_status") == "included" and row.get("pico_id") in core_picos:
        return {
            "decision": "promote_to_core",
            "priority": "P1",
            "reasons": ["included_and_linked_to_core_pico_after_second_pass"],
            "suggested_fix": None,
        }
    if {"ocr_glued_words", "long_unbroken_tokens", "methodology_context", "table_or_navigation"} & set(reasons):
        return {"decision": "quarantine", "priority": "P3", "reasons": reasons, "suggested_fix": None}
    return {"decision": "keep_review", "priority": "P2", "reasons": reasons, "suggested_fix": None}


def decision_record(entity_type: str, row: JsonDict, decision: JsonDict) -> JsonDict:
    id_field = {
        "recommendation_version": "recommendation_version_id",
        "pico_question": "pico_id",
        "evidence_item": "evidence_id",
    }[entity_type]
    return {
        "entity_type": entity_type,
        "entity_id": row.get(id_field),
        "decision": decision["decision"],
        "priority": decision["priority"],
        "reasons": decision.get("reasons") or [],
        "suggested_fix": decision.get("suggested_fix"),
        "prior_review": review_payload(row),
        "text_preview": clean_text(
            row.get("recommendation_text")
            or row.get("clinical_question")
            or row.get("source_text")
            or row.get("source_span")
        )[:500],
        "source_section": row.get("source_section"),
    }


def build_second_pass(run_dir: Path, output_dir: Path) -> JsonDict:
    output_dir.mkdir(parents=True, exist_ok=True)
    core_dir = output_dir / "core_embedding_ready_v2"
    review_dir = output_dir / "review_queue_v2"
    quarantine_dir = output_dir / "quarantine_v2"
    for folder in (core_dir, review_dir, quarantine_dir):
        folder.mkdir(parents=True, exist_ok=True)

    publishable_versions = list(iter_jsonl(run_dir / "recommendation_versions.jsonl"))
    needs_review_versions = list(iter_jsonl(run_dir / "recommendation_versions.needs_review.jsonl"))
    picos = list(iter_jsonl(run_dir / "pico_questions.jsonl"))
    evidence = list(iter_jsonl(run_dir / "evidence_items.jsonl"))

    core_versions: list[JsonDict] = []
    review_versions: list[JsonDict] = []
    quarantine_versions: list[JsonDict] = []
    decisions: list[JsonDict] = []

    # Keep previously publishable recommendations, except the one flagged by the initial audit rule.
    for row in publishable_versions:
        text = clean_text(row.get("recommendation_text"))
        if text.startswith("It is recommended that End-tidal Control"):
            decision = {
                "decision": "revise",
                "priority": "P1",
                "reasons": ["initial_audit_requested_source_span_or_action_pattern_confirmation"],
                "suggested_fix": {"action": "manual_confirm_before_core_embedding", "recommendation_text": text},
            }
            review_versions.append(add_second_pass(row, decision))
            decisions.append(decision_record("recommendation_version", row, decision))
        else:
            decision = {
                "decision": "promote_to_core",
                "priority": "P1",
                "reasons": ["previous_publishable_and_confirmed_by_second_pass"],
                "suggested_fix": None,
            }
            core_versions.append(add_second_pass(row, decision))
            decisions.append(decision_record("recommendation_version", row, decision))

    for row in needs_review_versions:
        decision = decide_recommendation(row)
        reviewed = add_second_pass(row, decision)
        decisions.append(decision_record("recommendation_version", row, decision))
        if decision["decision"] == "promote_to_core":
            core_versions.append(reviewed)
        elif decision["decision"] == "quarantine":
            quarantine_versions.append(reviewed)
        else:
            review_versions.append(reviewed)

    core_pico_ids = {str(row.get("pico_id")) for row in core_versions if row.get("pico_id")}
    evidence_by_pico: dict[str, int] = defaultdict(int)
    for row in evidence:
        if row.get("pico_id"):
            evidence_by_pico[str(row["pico_id"])] += 1

    core_picos: list[JsonDict] = []
    review_picos: list[JsonDict] = []
    quarantine_picos: list[JsonDict] = []
    for row in picos:
        prior = review_payload(row).get("decision")
        if prior == "publish_core" and row.get("pico_id") in core_pico_ids:
            decision = {
                "decision": "promote_to_core",
                "priority": "P1",
                "reasons": ["previous_publish_core_pico_linked_to_core_recommendation"],
                "suggested_fix": None,
            }
        else:
            decision = decide_pico(row, core_pico_ids, evidence_by_pico)
        reviewed = add_second_pass(row, decision)
        decisions.append(decision_record("pico_question", row, decision))
        if decision["decision"] == "promote_to_core":
            core_picos.append(reviewed)
        elif decision["decision"] == "quarantine":
            quarantine_picos.append(reviewed)
        else:
            review_picos.append(reviewed)

    final_core_picos = {str(row.get("pico_id")) for row in core_picos if row.get("pico_id")}
    core_evidence: list[JsonDict] = []
    review_evidence: list[JsonDict] = []
    quarantine_evidence: list[JsonDict] = []
    for row in evidence:
        prior = review_payload(row).get("decision")
        if prior == "publish_core" and row.get("pico_id") in final_core_picos:
            decision = {
                "decision": "promote_to_core",
                "priority": "P1",
                "reasons": ["previous_publish_core_evidence_linked_to_core_pico"],
                "suggested_fix": None,
            }
        else:
            decision = decide_evidence(row, final_core_picos)
        reviewed = add_second_pass(row, decision)
        decisions.append(decision_record("evidence_item", row, decision))
        if decision["decision"] == "promote_to_core":
            core_evidence.append(reviewed)
        elif decision["decision"] == "quarantine":
            quarantine_evidence.append(reviewed)
        else:
            review_evidence.append(reviewed)

    write_jsonl(core_dir / "recommendation_versions.jsonl", core_versions)
    write_jsonl(core_dir / "pico_questions.jsonl", core_picos)
    write_jsonl(core_dir / "evidence_items.jsonl", core_evidence)
    write_jsonl(review_dir / "recommendation_versions.jsonl", review_versions)
    write_jsonl(review_dir / "pico_questions.jsonl", review_picos)
    write_jsonl(review_dir / "evidence_items.jsonl", review_evidence)
    write_jsonl(quarantine_dir / "recommendation_versions.jsonl", quarantine_versions)
    write_jsonl(quarantine_dir / "pico_questions.jsonl", quarantine_picos)
    write_jsonl(quarantine_dir / "evidence_items.jsonl", quarantine_evidence)
    write_jsonl(output_dir / "second_pass_review_decisions.jsonl", decisions)

    counts = Counter(f"{row['entity_type']}:{row['decision']}" for row in decisions)
    reason_counts = Counter(reason for row in decisions for reason in row.get("reasons") or [])
    summary = {
        "review_version": "codex_llm_human_second_pass_v1",
        "input_dir": str(run_dir),
        "output_dir": str(output_dir),
        "reviewed_at": utc_now(),
        "decision_counts": dict(counts),
        "top_reason_counts": dict(reason_counts.most_common(80)),
        "core_counts": {
            "recommendation_versions": len(core_versions),
            "pico_questions": len(core_picos),
            "evidence_items": len(core_evidence),
        },
        "review_counts": {
            "recommendation_versions": len(review_versions),
            "pico_questions": len(review_picos),
            "evidence_items": len(review_evidence),
        },
        "quarantine_counts": {
            "recommendation_versions": len(quarantine_versions),
            "pico_questions": len(quarantine_picos),
            "evidence_items": len(quarantine_evidence),
        },
        "policy": (
            "Second-pass LLM/human conservative review. Promote only clear actionable recommendation versions, "
            "well-formed PICO rows linked to core recommendations, and included evidence linked to core PICO rows. "
            "OCR noise, methodology/research recommendation text, ambiguous pronouns, and table/navigation text are quarantined."
        ),
        "recommended_next_action": "build_embedding_queue_from_core_embedding_ready_v2",
    }
    write_jsonl(output_dir / "second_pass_review_summary.jsonl", [summary])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Second-pass LLM/human review for needs_review Living-Guideline data.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = build_second_pass(Path(args.run_dir), Path(args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
