from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.pipeline.extraction.enhancement import common as enhancement_common
from src.pipeline.llm_review.auto_qc.grade import output_grade_id
from src.common.extraction_common import utc_now
from src.common.process_jsonl import iter_jsonl, write_jsonl


JsonDict = Dict[str, Any]

ACCEPT_MERGE_FIELDS = {
    "grade_system",
    "certainty",
    "strength",
    "risk_of_bias",
    "inconsistency",
    "indirectness",
    "imprecision",
    "publication_bias",
    "reasons_for_downgrade",
    "reasons_for_upgrade",
}


def parsed_result(row: JsonDict) -> JsonDict:
    return enhancement_common.parsed_result(row)


def as_payload(row: JsonDict, field: str) -> JsonDict:
    return enhancement_common.as_payload(row, field)


def index_by_grade_id(rows: Iterable[JsonDict]) -> Dict[str, JsonDict]:
    indexed: Dict[str, JsonDict] = {}
    for row in rows:
        row_id = output_grade_id(row)
        if row_id:
            indexed[row_id] = row
    return indexed


def index_by_id(rows: Iterable[JsonDict], id_field: str) -> Dict[str, JsonDict]:
    return enhancement_common.index_by_id(rows, id_field)


def llm_suggestion(llm_output: JsonDict) -> JsonDict:
    result = parsed_result(llm_output)
    return {
        "grade_candidate_id": result.get("grade_candidate_id"),
        "recommendation_candidate_id": result.get("recommendation_candidate_id"),
        "is_valid_grade": result.get("is_valid_grade"),
        "grade_system": result.get("grade_system"),
        "certainty": result.get("certainty"),
        "strength": result.get("strength"),
        "risk_of_bias": result.get("risk_of_bias"),
        "inconsistency": result.get("inconsistency"),
        "indirectness": result.get("indirectness"),
        "imprecision": result.get("imprecision"),
        "publication_bias": result.get("publication_bias"),
        "reasons_for_downgrade": result.get("reasons_for_downgrade", []),
        "reasons_for_upgrade": result.get("reasons_for_upgrade", []),
        "reject_reason": result.get("reject_reason"),
        "needs_human_review": result.get("needs_human_review"),
        "confidence": result.get("confidence"),
    }


def build_enhancement_payload(llm_output: Optional[JsonDict], qc: Optional[JsonDict], decision: str) -> JsonDict:
    return enhancement_common.build_enhancement_payload(llm_output, qc, decision, "llm_grade_auto_qc")


def compact_review_note(prefix: str, qc: Optional[JsonDict], llm_output: Optional[JsonDict]) -> str:
    return enhancement_common.compact_review_note(prefix, qc, llm_output)


def apply_auto_accept(candidate: JsonDict, llm_output: JsonDict, qc: JsonDict) -> JsonDict:
    result = parsed_result(llm_output)
    enhanced = deepcopy(candidate)
    for field in ACCEPT_MERGE_FIELDS:
        if field in result:
            enhanced[field] = result[field]
    enhanced["recommendation_candidate_id"] = result.get("recommendation_candidate_id") or enhanced.get("recommendation_candidate_id", "")
    enhanced["model_trace_id"] = llm_output.get("model_trace_id") or enhanced.get("model_trace_id")
    enhanced["extraction_method"] = "hybrid"
    enhanced["extraction_confidence"] = result.get("confidence")
    enhanced["status"] = "accepted"
    enhanced["review_note"] = compact_review_note("Auto accepted by grade LLM auto QC.", qc, llm_output)
    enhanced["updated_at"] = utc_now()
    return enhanced


def apply_auto_reject(candidate: JsonDict, llm_output: JsonDict, qc: JsonDict) -> JsonDict:
    enhanced = deepcopy(candidate)
    enhanced["status"] = "rejected"
    enhanced["review_note"] = compact_review_note("Auto rejected by grade LLM auto QC.", qc, llm_output)
    enhanced["updated_at"] = utc_now()
    return enhanced


def apply_needs_review(candidate: JsonDict, llm_output: Optional[JsonDict], qc: Optional[JsonDict]) -> JsonDict:
    enhanced = deepcopy(candidate)
    enhanced["status"] = "needs_review"
    enhanced["review_note"] = compact_review_note("Needs review after grade LLM auto QC.", qc, llm_output)
    enhanced["updated_at"] = utc_now()
    return enhanced


def attach_payloads(candidate: JsonDict, llm_output: Optional[JsonDict], qc: Optional[JsonDict], decision: str) -> JsonDict:
    return enhancement_common.attach_payloads(candidate, llm_output, qc, decision, "llm_grade_auto_qc", llm_suggestion)


def enhance_candidate(candidate: JsonDict, llm_output: Optional[JsonDict], qc: Optional[JsonDict]) -> JsonDict:
    if not llm_output or not qc:
        enhanced = deepcopy(candidate)
        normalized = as_payload(enhanced, "normalized_payload")
        normalized["enhancement"] = build_enhancement_payload(None, None, "not_processed")
        enhanced["normalized_payload"] = normalized
        return enhanced

    decision = str(qc.get("auto_qc_decision") or "needs_review")
    if decision == "auto_accept":
        enhanced = apply_auto_accept(candidate, llm_output, qc)
    elif decision == "auto_reject":
        enhanced = apply_auto_reject(candidate, llm_output, qc)
    else:
        enhanced = apply_needs_review(candidate, llm_output, qc)
    return attach_payloads(enhanced, llm_output, qc, decision)


def summarize(rows: List[JsonDict]) -> JsonDict:
    status_counts, enhancement_counts, policy_counts = enhancement_common.summarize_enhancements(rows)
    grade_system_counts: Counter[str] = Counter()
    certainty_counts: Counter[str] = Counter()
    strength_counts: Counter[str] = Counter()
    for row in rows:
        grade_system_counts[str(row.get("grade_system") or "unknown")] += 1
        certainty_counts[str(row.get("certainty") or "unknown")] += 1
        strength_counts[str(row.get("strength") or "unknown")] += 1
    return {
        "enhanced_grade_candidates": len(rows),
        "status_counts": status_counts,
        "enhancement_decision_counts": enhancement_counts,
        "qc_policy_counts": policy_counts,
        "grade_system_counts": dict(grade_system_counts),
        "certainty_counts": dict(certainty_counts),
        "strength_counts": dict(strength_counts),
    }


def build_enhanced_file(
    candidates_input: str | Path,
    llm_outputs_input: str | Path,
    auto_qc_input: str | Path,
    enhanced_output: str | Path,
    summary_output: str | Path,
) -> JsonDict:
    """应用 GRADE 候选的 LLM/QC 决策，并写出增强后的分级候选记录。"""

    outputs_by_grade = index_by_grade_id(iter_jsonl(llm_outputs_input))
    qc_by_grade = index_by_id(iter_jsonl(auto_qc_input), "grade_candidate_id")

    enhanced_rows: List[JsonDict] = []
    for candidate in iter_jsonl(candidates_input):
        grade_id = str(candidate.get("grade_candidate_id") or "")
        enhanced_rows.append(enhance_candidate(candidate, outputs_by_grade.get(grade_id), qc_by_grade.get(grade_id)))

    summary = summarize(enhanced_rows)
    summary["candidates_input"] = str(candidates_input)
    summary["llm_outputs_input"] = str(llm_outputs_input)
    summary["auto_qc_input"] = str(auto_qc_input)
    write_jsonl(enhanced_output, enhanced_rows)
    write_jsonl(summary_output, [summary])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build enhanced grade candidates from LLM output and auto QC.")
    parser.add_argument("--candidates-input", required=True)
    parser.add_argument("--llm-outputs-input", required=True)
    parser.add_argument("--auto-qc-input", required=True)
    parser.add_argument("--enhanced-output", required=True)
    parser.add_argument("--summary-output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_enhanced_file(
        candidates_input=args.candidates_input,
        llm_outputs_input=args.llm_outputs_input,
        auto_qc_input=args.auto_qc_input,
        enhanced_output=args.enhanced_output,
        summary_output=args.summary_output,
    )
    print(
        "enhanced_grade_candidates={enhanced_grade_candidates} status={status_counts} decisions={enhancement_decision_counts}".format(
            **summary
        )
    )


if __name__ == "__main__":
    main()
