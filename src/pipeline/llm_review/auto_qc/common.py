"""LLM 复核文件：构建复核队列、prompt、响应解析和自动质检，让候选结果进入人工/模型辅助复核流程。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from src.common.extraction_common import JsonDict
from src.common.process_jsonl import iter_jsonl


NOISY_REJECT_RE = re.compile(
    r"\b(fragment|garbled|methods?|not (?:a )?(?:clear )?recommendation|"
    r"incomplete|long_statement|preamble|definition|development process|not specific)\b",
    re.I,
)

GENERIC_QC_POLICY: JsonDict = {
    "accepted_grade_systems": {
        "GRADE",
        "COR_LOE",
        "LETTER_GRADE",
        "NUMERIC_LETTER_GRADE",
        "VERB_BASED",
        "unknown",
    },
    "requires_certainty_for_auto_accept": True,
    "requires_strength_for_auto_accept": True,
    "auto_accept_confidence": 0.85,
    "notes": "Generic QC is conservative and does not vary by guideline source.",
}

PROFILE_TRUST_STATES = {"auto_med", "auto_high", "human_confirmed"}


def parsed(row: JsonDict) -> JsonDict:
    value = row.get("parsed_result")
    return value if isinstance(value, dict) else {}


def payload(row: JsonDict) -> JsonDict:
    value = row.get("normalized_payload")
    return value if isinstance(value, dict) else {}


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def normalized_source_metadata(candidate: JsonDict) -> JsonDict:
    metadata = payload(candidate).get("source_metadata")
    return metadata if isinstance(metadata, dict) else {}


def text_from_values(*values: Any) -> str:
    parts: list[str] = []
    for value in values:
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif isinstance(value, dict):
            parts.extend(str(item) for item in value.values())
        else:
            parts.append(str(value or ""))
    return normalize_text(" ".join(parts))


def index_by_id(rows: Iterable[JsonDict], id_field: str) -> dict[str, JsonDict]:
    indexed: dict[str, JsonDict] = {}
    for row in rows:
        row_id = str(row.get(id_field) or "")
        if row_id:
            indexed[row_id] = row
    return indexed


def load_profiles(path: str | Path | None) -> dict[str, JsonDict]:
    if not path:
        return {}
    profile_path = Path(path)
    if not profile_path.exists():
        return {}
    return index_by_id(iter_jsonl(profile_path), "record_id")


def profile_evidence_summary(profile: JsonDict, limit: int = 5) -> list[JsonDict]:
    evidence_items = profile.get("evidence_for_profile")
    if not isinstance(evidence_items, list):
        return []
    summary: list[JsonDict] = []
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        summary.append(
            {
                "kind": item.get("kind", ""),
                "field": item.get("field", ""),
                "target": item.get("target", ""),
                "weight": item.get("weight"),
            }
        )
        if len(summary) >= limit:
            break
    return summary


def profile_qc_context(profile: JsonDict) -> JsonDict:
    if not profile:
        return {
            "profile_id": "",
            "issuer": "unknown",
            "issuer_confidence": 0.0,
            "grading_system": "unknown",
            "grading_system_version": "unknown",
            "grading_confidence": 0.0,
            "resolution_state": "unresolved",
            "profile_evidence_summary": [],
        }
    return {
        "profile_id": profile.get("profile_id", ""),
        "issuer": profile.get("issuer", "unknown"),
        "issuer_confidence": profile.get("issuer_confidence", 0.0),
        "grading_system": profile.get("grading_system", "unknown"),
        "grading_system_version": profile.get("grading_system_version", "unknown"),
        "grading_confidence": profile.get("grading_confidence", 0.0),
        "resolution_state": profile.get("resolution_state", "unresolved"),
        "profile_evidence_summary": profile_evidence_summary(profile),
    }


def profile_reason_tokens(profile: JsonDict) -> list[str]:
    if not profile:
        return ["profile_missing"]
    return [
        f"profile_resolution:{profile.get('resolution_state', 'unresolved')}",
        f"profile_issuer:{profile.get('issuer', 'unknown')}",
        f"profile_grading_system:{profile.get('grading_system', 'unknown')}",
        f"profile_grading_confidence:{profile.get('grading_confidence', 0.0)}",
    ]


def policy_key_for_profile(profile: JsonDict) -> str:
    if not profile or profile.get("resolution_state") not in PROFILE_TRUST_STATES:
        return "generic"
    grading_system = str(profile.get("grading_system") or "unknown")
    return grading_system if grading_system != "unknown" else "generic"


def effective_policy_key(profile: JsonDict) -> str:
    return policy_key_for_profile(profile)


def qc_policy(profile: JsonDict) -> JsonDict:
    policy = dict(GENERIC_QC_POLICY)
    dimensions = profile.get("dimensions") if isinstance(profile, dict) else {}
    if isinstance(dimensions, dict):
        if dimensions.get("certainty") is False:
            policy["requires_certainty_for_auto_accept"] = False
        if dimensions.get("strength") is False:
            policy["requires_strength_for_auto_accept"] = False
    grading_system = str(profile.get("grading_system") or "unknown") if isinstance(profile, dict) else "unknown"
    if grading_system != "unknown":
        policy["accepted_grade_systems"] = set(policy["accepted_grade_systems"]) | {grading_system}
    return policy


def group_grades_by_recommendation(rows: Iterable[JsonDict]) -> dict[str, list[JsonDict]]:
    grouped: dict[str, list[JsonDict]] = {}
    for row in rows:
        rec_id = str(row.get("recommendation_candidate_id") or "")
        if rec_id:
            grouped.setdefault(rec_id, []).append(row)
    return grouped


def field_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip()) and value != "unclear"
    if isinstance(value, list):
        return bool(value)
    return True


def noisy_reject_reason(text: Any) -> bool:
    return bool(NOISY_REJECT_RE.search(str(text or "")))


def add_output_status_score(output: JsonDict, reasons: list[str]) -> float:
    if output.get("status") == "validated":
        reasons.append("validated_output")
        return 0.2
    reasons.append(f"output_status:{output.get('status')}")
    return -0.4


def add_confidence_score(result: JsonDict, reasons: list[str]) -> tuple[float, float]:
    confidence = result.get("confidence")
    if isinstance(confidence, (int, float)):
        confidence_value = max(0.0, min(float(confidence), 1.0))
        reasons.append(f"confidence:{round(float(confidence), 3)}")
        return confidence_value * 0.25, confidence_value
    reasons.append("missing_confidence")
    return -0.1, 0.0


def add_validity_score(result: JsonDict, field: str, valid_reason: str, rejected_reason: str, reasons: list[str]) -> float:
    """给布尔型 LLM 有效性字段评分，并追加对应原因 token。"""

    if result.get(field) is True:
        reasons.append(valid_reason)
        return 0.2
    if result.get(field) is False:
        reasons.append(rejected_reason)
        return -0.35
    return 0.0


def add_human_review_score(result: JsonDict, reasons: list[str]) -> float:
    if result.get("needs_human_review") is False:
        reasons.append("llm_no_human_review_needed")
        return 0.1
    if result.get("needs_human_review") is True:
        reasons.append("llm_requested_review")
        return -0.05
    return 0.0


def clamped_score(score: float) -> float:
    return round(max(0.0, min(score, 1.0)), 4)


def summarize_qc_records(records: list[JsonDict]) -> JsonDict:
    decision_counts: Counter[str] = Counter()
    policy_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    profile_state_counts: Counter[str] = Counter()
    profile_grading_counts: Counter[str] = Counter()
    for row in records:
        decision_counts[str(row.get("auto_qc_decision") or "unknown")] += 1
        policy_counts[str(row.get("qc_policy_key") or "generic")] += 1
        action_counts[str(row.get("recommended_action") or "unknown")] += 1
        profile_context = row.get("profile_context")
        if isinstance(profile_context, dict):
            profile_state_counts[str(profile_context.get("resolution_state") or "unknown")] += 1
            profile_grading_counts[str(profile_context.get("grading_system") or "unknown")] += 1
        for reason in row.get("auto_qc_reasons") or []:
            reason_counts[str(reason)] += 1
    return {
        "qc_items": len(records),
        "decision_counts": dict(decision_counts),
        "qc_policy_counts": dict(policy_counts),
        "action_counts": dict(action_counts),
        "profile_resolution_state_counts": dict(profile_state_counts),
        "profile_grading_system_counts": dict(profile_grading_counts),
        "reason_counts": dict(reason_counts),
    }

