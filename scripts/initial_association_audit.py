"""本地脚本入口：把 src 中的项目能力包装成命令行工具，方便运行、审计或质量检查。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.process_jsonl import iter_jsonl, write_jsonl


JsonDict = dict[str, Any]


CORE_FILES = {
    "recommendation_versions": "recommendation_versions.jsonl",
    "recommendation_versions_needs_review": "recommendation_versions.needs_review.jsonl",
    "pico_questions": "pico_questions.jsonl",
    "evidence_items": "evidence_items.jsonl",
}

NOISE_PATTERNS = {
    "ocr_glued_words": re.compile(r"\b[A-Za-z]{8,}(?:tion|ment|ing|sis)[A-Za-z]{8,}\b"),
    "broken_hyphenation": re.compile(r"\b[A-Za-z]{2,}-\s+[A-Za-z]{2,}\b"),
    "citation_heavy": re.compile(r"(?:\[\d+\]|\(\d{4}\)|\bdoi:|\bet al\.)", re.I),
    "table_or_navigation": re.compile(r"\b(table|figure|appendix|annex|copyright|all rights reserved)\b", re.I),
    "methodology_context": re.compile(r"\b(methods?|systematic review of the literature|task force|committee|conflict of interest)\b", re.I),
}

RECOMMENDATION_ACTION = re.compile(
    r"\b(should|recommend|recommended|suggest|suggests|offer|avoid|consider|is indicated|are indicated|may be used|should not|do not)\b",
    re.I,
)


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def payload(row: JsonDict) -> JsonDict:
    value = row.get("normalized_payload")
    return value if isinstance(value, dict) else {}


def review_payload(row: JsonDict) -> JsonDict:
    value = payload(row).get("codex_full_data_review")
    return value if isinstance(value, dict) else {}


def source_metadata(row: JsonDict) -> JsonDict:
    value = payload(row).get("source_metadata")
    return value if isinstance(value, dict) else {}


def text_flags(text: str) -> list[str]:
    flags = []
    normalized = clean_text(text)
    if not normalized:
        return ["empty_text"]
    if len(normalized) < 40:
        flags.append("very_short_text")
    if len(normalized) > 2200:
        flags.append("very_long_text")
    for name, pattern in NOISE_PATTERNS.items():
        if pattern.search(normalized):
            flags.append(name)
    if len(re.findall(r"[A-Za-z]{20,}", normalized)) >= 2:
        flags.append("long_unbroken_tokens")
    if normalized.count("•") >= 4 or normalized.count(";") >= 8:
        flags.append("list_or_mixed_block")
    return flags


def field_missing(row: JsonDict, fields: list[str]) -> list[str]:
    return [field for field in fields if row.get(field) in (None, "", [], {})]


def entity_id(row: JsonDict, entity_type: str) -> str:
    id_field = {
        "recommendation_version": "recommendation_version_id",
        "pico_question": "pico_id",
        "evidence_item": "evidence_id",
    }[entity_type]
    return str(row.get(id_field) or "")


def status_from_review(row: JsonDict) -> str:
    review = review_payload(row)
    return str(review.get("decision") or row.get("quality_status") or row.get("status") or row.get("screening_status") or "")


def decide_recommendation_version(row: JsonDict, evidence_by_version: dict[str, int]) -> JsonDict:
    rec_text = clean_text(row.get("recommendation_text"))
    source_text = clean_text(row.get("source_text") or row.get("source_span"))
    flags = text_flags(rec_text) + [f"source_{flag}" for flag in text_flags(source_text) if flag != "very_long_text"]
    missing = field_missing(row, ["recommendation_version_id", "recommendation_text", "pico_id", "grade_candidate_id", "source_text"])
    version_id = str(row.get("recommendation_version_id") or "")
    evidence_count = evidence_by_version.get(version_id, 0)
    if not RECOMMENDATION_ACTION.search(rec_text):
        flags.append("missing_action_signal")
    if rec_text and source_text and rec_text not in source_text:
        flags.append("claim_not_verbatim_in_source_span")
    if evidence_count == 0:
        flags.append("no_direct_evidence_item_link")

    source_flags = set(flags)
    if row.get("quality_status") == "publishable" and not missing and source_flags.isdisjoint(
        {"empty_text", "very_short_text", "ocr_glued_words", "long_unbroken_tokens", "missing_action_signal"}
    ):
        decision = "accept"
    elif row.get("quality_status") == "publishable" and ("no_direct_evidence_item_link" in source_flags or "source_methodology_context" in source_flags):
        decision = "revise"
    elif row.get("quality_status") == "needs_review":
        decision = "needs_review"
    elif {"empty_text", "ocr_glued_words", "long_unbroken_tokens", "missing_action_signal"} & source_flags:
        decision = "quarantine"
    else:
        decision = "needs_review"

    return {
        "entity_type": "recommendation_version",
        "entity_id": version_id,
        "decision": decision,
        "reasons": sorted(set(missing + flags)),
        "linkage": {
            "pico_id": row.get("pico_id"),
            "grade_candidate_id": row.get("grade_candidate_id"),
            "direct_evidence_items": evidence_count,
        },
        "text_preview": rec_text[:360],
        "source_section": row.get("source_section"),
        "prior_status": status_from_review(row),
    }


def decide_pico(row: JsonDict, versions_by_pico: dict[str, int], evidence_by_pico: dict[str, int]) -> JsonDict:
    text = clean_text(row.get("clinical_question") or row.get("source_text") or row.get("source_span"))
    flags = text_flags(text)
    missing = field_missing(row, ["pico_id", "clinical_question", "population", "intervention"])
    if not row.get("outcomes"):
        flags.append("outcomes_empty")
    pico_id = str(row.get("pico_id") or "")
    version_count = versions_by_pico.get(pico_id, 0)
    evidence_count = evidence_by_pico.get(pico_id, 0)
    if version_count == 0:
        flags.append("no_recommendation_version_link")
    if evidence_count == 0:
        flags.append("no_evidence_link")
    confidence = row.get("extraction_confidence")
    if isinstance(confidence, (int, float)) and confidence < 0.6:
        flags.append("low_extraction_confidence")

    if row.get("status") == "active" and not missing and version_count > 0 and evidence_count > 0:
        decision = "accept"
    elif row.get("status") == "active" and not missing:
        decision = "revise"
    elif "very_short_text" in flags or missing:
        decision = "needs_review"
    else:
        decision = "needs_review"

    return {
        "entity_type": "pico_question",
        "entity_id": pico_id,
        "decision": decision,
        "reasons": sorted(set(missing + flags)),
        "linkage": {"recommendation_versions": version_count, "evidence_items": evidence_count},
        "text_preview": text[:360],
        "source_section": row.get("source_section"),
        "prior_status": status_from_review(row),
    }


def decide_evidence(row: JsonDict, versions_by_id: set[str], picos: set[str]) -> JsonDict:
    text = clean_text(row.get("source_text") or row.get("source_span"))
    flags = text_flags(text)
    missing = field_missing(row, ["evidence_id", "pico_id", "source_text"])
    evidence_id = str(row.get("evidence_id") or "")
    if not row.get("outcomes_extracted"):
        flags.append("outcomes_empty")
    if row.get("effect_direction") in (None, "", "uncertain"):
        flags.append("effect_direction_uncertain")
    if row.get("study_design") in (None, "", "unclear"):
        flags.append("study_design_unclear")
    if row.get("pico_id") not in picos:
        flags.append("pico_id_not_found")
    if row.get("recommendation_version_id") and row.get("recommendation_version_id") not in versions_by_id:
        flags.append("recommendation_version_id_not_found")
    if not row.get("recommendation_version_id"):
        flags.append("no_direct_recommendation_version_link")
    if row.get("screening_status") != "included":
        flags.append("not_included_status")
    confidence = row.get("extraction_confidence")
    if isinstance(confidence, (int, float)) and confidence < 0.6:
        flags.append("low_extraction_confidence")

    flag_set = set(flags)
    if row.get("screening_status") == "included" and "pico_id_not_found" not in flag_set and "empty_text" not in flag_set:
        decision = "accept"
    elif "methodology_context" in flag_set or "table_or_navigation" in flag_set:
        decision = "quarantine"
    elif row.get("screening_status") in {"association_review", "uncertain"}:
        decision = "needs_review"
    else:
        decision = "needs_review"

    return {
        "entity_type": "evidence_item",
        "entity_id": evidence_id,
        "decision": decision,
        "reasons": sorted(set(missing + flags)),
        "linkage": {
            "pico_id": row.get("pico_id"),
            "recommendation_version_id": row.get("recommendation_version_id"),
            "recommendation_candidate_id": row.get("recommendation_candidate_id"),
        },
        "text_preview": text[:360],
        "source_section": row.get("source_section"),
        "prior_status": status_from_review(row),
    }


def add_sample(samples: dict[tuple[str, str], list[JsonDict]], decision: JsonDict, limit: int) -> None:
    key = (decision["entity_type"], decision["decision"])
    if len(samples[key]) < limit:
        samples[key].append(decision)


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.open("r", encoding="utf-8") if line.strip())


def build_audit(run_dir: Path, output_dir: Path, sample_limit: int) -> tuple[JsonDict, list[JsonDict]]:
    versions = list(iter_jsonl(run_dir / "recommendation_versions.jsonl"))
    needs_review_versions = list(iter_jsonl(run_dir / "recommendation_versions.needs_review.jsonl"))
    all_versions = versions + needs_review_versions
    version_ids = {str(row.get("recommendation_version_id")) for row in all_versions if row.get("recommendation_version_id")}
    pico_ids = {str(row.get("pico_id")) for row in iter_jsonl(run_dir / "pico_questions.jsonl") if row.get("pico_id")}

    evidence_by_version: dict[str, int] = defaultdict(int)
    evidence_by_pico: dict[str, int] = defaultdict(int)
    evidence_counts = Counter()
    for row in iter_jsonl(run_dir / "evidence_items.jsonl"):
        evidence_counts["rows"] += 1
        evidence_counts[f"screening_status:{row.get('screening_status') or 'missing'}"] += 1
        if row.get("recommendation_version_id"):
            evidence_by_version[str(row["recommendation_version_id"])] += 1
        if row.get("pico_id"):
            evidence_by_pico[str(row["pico_id"])] += 1

    versions_by_pico: dict[str, int] = defaultdict(int)
    for row in all_versions:
        if row.get("pico_id"):
            versions_by_pico[str(row["pico_id"])] += 1

    summary: JsonDict = {
        "audit_version": "initial_association_audit_v1",
        "run_dir": str(run_dir),
        "input_counts": {name: count_jsonl(run_dir / filename) for name, filename in CORE_FILES.items()},
        "decision_counts": {},
        "reason_counts": {},
        "linkage_summary": {},
        "recommendations": [],
    }
    samples: dict[tuple[str, str], list[JsonDict]] = defaultdict(list)
    decision_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()

    for row in all_versions:
        decision = decide_recommendation_version(row, evidence_by_version)
        decision_counts[f"{decision['entity_type']}:{decision['decision']}"] += 1
        for reason in decision["reasons"]:
            reason_counts[f"{decision['entity_type']}:{reason}"] += 1
        add_sample(samples, decision, sample_limit)

    for row in iter_jsonl(run_dir / "pico_questions.jsonl"):
        decision = decide_pico(row, versions_by_pico, evidence_by_pico)
        decision_counts[f"{decision['entity_type']}:{decision['decision']}"] += 1
        for reason in decision["reasons"]:
            reason_counts[f"{decision['entity_type']}:{reason}"] += 1
        add_sample(samples, decision, sample_limit)

    for row in iter_jsonl(run_dir / "evidence_items.jsonl"):
        decision = decide_evidence(row, version_ids, pico_ids)
        decision_counts[f"{decision['entity_type']}:{decision['decision']}"] += 1
        for reason in decision["reasons"]:
            reason_counts[f"{decision['entity_type']}:{reason}"] += 1
        add_sample(samples, decision, sample_limit)

    summary["decision_counts"] = dict(decision_counts)
    summary["reason_counts"] = dict(reason_counts.most_common(80))
    summary["linkage_summary"] = {
        "recommendation_versions_total": len(all_versions),
        "recommendation_versions_with_direct_evidence": sum(1 for version_id in version_ids if evidence_by_version.get(version_id, 0) > 0),
        "recommendation_versions_without_direct_evidence": sum(1 for version_id in version_ids if evidence_by_version.get(version_id, 0) == 0),
        "picos_total": len(pico_ids),
        "picos_with_recommendation_version": sum(1 for pico_id in pico_ids if versions_by_pico.get(pico_id, 0) > 0),
        "picos_with_evidence": sum(1 for pico_id in pico_ids if evidence_by_pico.get(pico_id, 0) > 0),
        "evidence_screening_status_counts": dict(evidence_counts),
    }
    summary["recommendations"] = [
        "Regenerate the embedding queue from the reviewed package, not from the raw run root, so recommendation_version vectors are included.",
        "Keep source_block vectors out of the core embedding set unless a separate retrieval collection is explicitly needed.",
        "Before core embedding, accept only publishable recommendation versions and active/linked PICO/evidence subsets; keep weak associations in review queues.",
        "Direct evidence links to recommendation_version are sparse; use PICO-mediated retrieval or add a second association pass before treating evidence as claim support.",
    ]

    sample_rows = [item for group in samples.values() for item in group]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "initial_association_audit_summary.jsonl", [summary])
    write_jsonl(output_dir / "initial_association_audit_samples.jsonl", sample_rows)
    return summary, sample_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a read-only LLM/human-style association audit over reviewed JSONL data.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-limit", type=int, default=5)
    args = parser.parse_args()
    summary, samples = build_audit(Path(args.run_dir), Path(args.output_dir), args.sample_limit)
    print(
        json.dumps(
            {
                "summary_output": str(Path(args.output_dir) / "initial_association_audit_summary.jsonl"),
                "samples_output": str(Path(args.output_dir) / "initial_association_audit_samples.jsonl"),
                "decision_counts": summary["decision_counts"],
                "sample_rows": len(samples),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

