from __future__ import annotations

import argparse
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.pipeline.extraction.enhancement import common as enhancement_common
from src.common.extraction_common import utc_now
from src.common.process_jsonl import iter_jsonl, write_jsonl


JsonDict = Dict[str, Any]

ACCEPT_MERGE_FIELDS = {
    "direction",
    "strength",
    "certainty",
    "population",
    "intervention",
    "comparator",
    "outcomes",
    "rationale",
    "remarks",
}

INVALID_CODE_WORDS = {
    ".",
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "based",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "s",
    "s.",
    "should",
    "statements",
    "the",
    "to",
    "was",
}

VALID_CODE_RE = re.compile(
    r"^(?:"
    r"(?:recommendation|statement|strong recommendation|conditional recommendation)\s+[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*|"
    r"[0-9]+(?:\.[0-9]+){0,4}[A-Za-z]?|"
    r"[A-Z][0-9]+(?:\.[0-9]+)*|"
    r"[0-9]+[A-Z]"
    r")$",
    re.I,
)


def parsed_result(row: JsonDict) -> JsonDict:
    return enhancement_common.parsed_result(row)


def as_payload(row: JsonDict, field: str) -> JsonDict:
    return enhancement_common.as_payload(row, field)


def index_by_id(rows: Iterable[JsonDict], id_field: str) -> Dict[str, JsonDict]:
    return enhancement_common.index_by_id(rows, id_field)


def normalize_recommendation_code(value: Any) -> tuple[Optional[str], bool]:
    if value is None:
        return None, False
    code = " ".join(str(value).strip().split())
    if not code:
        return None, value is not None
    lower = code.lower()
    if lower in INVALID_CODE_WORDS:
        return None, True
    if len(code) == 1 and not code.isdigit():
        return None, True
    if len(code) > 40:
        return None, True
    if re.search(r"\s", code) and not re.match(r"(?i)^(recommendation|statement|strong recommendation|conditional recommendation)\s+", code):
        return None, True
    if VALID_CODE_RE.match(code):
        return code, code != value
    if code.isdigit():
        return code, code != value
    return None, True


def llm_suggestion(llm_output: JsonDict) -> JsonDict:
    result = parsed_result(llm_output)
    return {
        "corrected_recommendation_text": result.get("corrected_recommendation_text"),
        "direction": result.get("direction"),
        "strength": result.get("strength"),
        "certainty": result.get("certainty"),
        "population": result.get("population"),
        "intervention": result.get("intervention"),
        "comparator": result.get("comparator"),
        "outcomes": result.get("outcomes", []),
        "rationale": result.get("rationale"),
        "remarks": result.get("remarks"),
        "reject_reason": result.get("reject_reason"),
        "needs_human_review": result.get("needs_human_review"),
        "confidence": result.get("confidence"),
    }


def build_enhancement_payload(llm_output: Optional[JsonDict], qc: Optional[JsonDict], decision: str) -> JsonDict:
    return enhancement_common.build_enhancement_payload(llm_output, qc, decision, "llm_auto_qc")


def compact_review_note(prefix: str, qc: Optional[JsonDict], llm_output: Optional[JsonDict]) -> str:
    return enhancement_common.compact_review_note(prefix, qc, llm_output)


def apply_auto_accept(candidate: JsonDict, llm_output: JsonDict, qc: JsonDict) -> JsonDict:
    result = parsed_result(llm_output)
    enhanced = deepcopy(candidate)
    corrected_text = result.get("corrected_recommendation_text")
    if isinstance(corrected_text, str) and corrected_text.strip():
        enhanced["recommendation_text"] = corrected_text
    for field in ACCEPT_MERGE_FIELDS:
        if field in result:
            enhanced[field] = result[field]
    enhanced["model_trace_id"] = llm_output.get("model_trace_id") or enhanced.get("model_trace_id")
    enhanced["extraction_method"] = "hybrid"
    enhanced["extraction_confidence"] = result.get("confidence")
    enhanced["status"] = "accepted"
    enhanced["review_note"] = compact_review_note("Auto accepted by LLM auto QC.", qc, llm_output)
    enhanced["updated_at"] = utc_now()
    return enhanced


def apply_auto_reject(candidate: JsonDict, llm_output: JsonDict, qc: JsonDict) -> JsonDict:
    enhanced = deepcopy(candidate)
    enhanced["status"] = "rejected"
    enhanced["review_note"] = compact_review_note("Auto rejected by LLM auto QC.", qc, llm_output)
    enhanced["updated_at"] = utc_now()
    return enhanced


def apply_needs_review(candidate: JsonDict, llm_output: Optional[JsonDict], qc: Optional[JsonDict]) -> JsonDict:
    enhanced = deepcopy(candidate)
    enhanced["status"] = "needs_review"
    enhanced["review_note"] = compact_review_note("Needs review after LLM auto QC.", qc, llm_output)
    enhanced["updated_at"] = utc_now()
    return enhanced


def attach_payloads(candidate: JsonDict, llm_output: Optional[JsonDict], qc: Optional[JsonDict], decision: str) -> JsonDict:
    return enhancement_common.attach_payloads(candidate, llm_output, qc, decision, "llm_auto_qc", llm_suggestion)


def clean_recommendation_code(candidate: JsonDict) -> JsonDict:
    enhanced = deepcopy(candidate)
    original_code = enhanced.get("recommendation_code")
    normalized_code, changed = normalize_recommendation_code(original_code)
    if not changed:
        return enhanced
    normalized = as_payload(enhanced, "normalized_payload")
    normalized["recommendation_code_normalization"] = {
        "original_code": original_code,
        "normalized_code": normalized_code,
        "reason": "invalid_or_noise_code" if normalized_code is None else "normalized_code_format",
    }
    enhanced["recommendation_code"] = normalized_code
    enhanced["normalized_payload"] = normalized
    return enhanced


def enhance_candidate(candidate: JsonDict, llm_output: Optional[JsonDict], qc: Optional[JsonDict]) -> JsonDict:
    if not llm_output or not qc:
        enhanced = deepcopy(candidate)
        normalized = as_payload(enhanced, "normalized_payload")
        normalized["enhancement"] = build_enhancement_payload(None, None, "not_processed")
        enhanced["normalized_payload"] = normalized
        return clean_recommendation_code(enhanced)

    decision = str(qc.get("auto_qc_decision") or "needs_review")
    if decision == "auto_accept":
        enhanced = apply_auto_accept(candidate, llm_output, qc)
    elif decision == "auto_reject":
        enhanced = apply_auto_reject(candidate, llm_output, qc)
    else:
        enhanced = apply_needs_review(candidate, llm_output, qc)
    return clean_recommendation_code(attach_payloads(enhanced, llm_output, qc, decision))


def summarize(rows: List[JsonDict]) -> JsonDict:
    status_counts, enhancement_counts, policy_counts = enhancement_common.summarize_enhancements(rows)
    code_normalization_counts: Counter[str] = Counter()
    for row in rows:
        code_norm = row.get("normalized_payload", {}).get("recommendation_code_normalization")
        if isinstance(code_norm, dict):
            code_normalization_counts[str(code_norm.get("reason") or "changed")] += 1
    return {
        "enhanced_candidates": len(rows),
        "status_counts": status_counts,
        "enhancement_decision_counts": enhancement_counts,
        "qc_policy_counts": policy_counts,
        "recommendation_code_normalization_counts": dict(code_normalization_counts),
    }


def build_enhanced_file(
    candidates_input: str | Path,
    llm_outputs_input: str | Path,
    auto_qc_input: str | Path,
    enhanced_output: str | Path,
    summary_output: str | Path,
) -> JsonDict:
    """应用推荐候选的 LLM/QC 决策，并写出增强后的候选记录。"""

    outputs_by_candidate = index_by_id(iter_jsonl(llm_outputs_input), "recommendation_candidate_id")
    qc_by_candidate = index_by_id(iter_jsonl(auto_qc_input), "candidate_id")

    enhanced_rows: List[JsonDict] = []
    for candidate in iter_jsonl(candidates_input):
        candidate_id = str(candidate.get("candidate_id") or "")
        enhanced_rows.append(enhance_candidate(candidate, outputs_by_candidate.get(candidate_id), qc_by_candidate.get(candidate_id)))

    summary = summarize(enhanced_rows)
    summary["candidates_input"] = str(candidates_input)
    summary["llm_outputs_input"] = str(llm_outputs_input)
    summary["auto_qc_input"] = str(auto_qc_input)
    write_jsonl(enhanced_output, enhanced_rows)
    write_jsonl(summary_output, [summary])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build enhanced recommendation candidates from LLM output and auto QC.")
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
    print("enhanced_candidates={enhanced_candidates} status={status_counts} decisions={enhancement_decision_counts}".format(**summary))


if __name__ == "__main__":
    main()
