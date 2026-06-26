from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.pipeline.llm_review.auto_qc.common import (
    add_confidence_score,
    add_human_review_score,
    add_output_status_score,
    add_validity_score,
    clamped_score,
    effective_policy_key,
    field_present,
    load_profiles,
    profile_qc_context,
    profile_reason_tokens,
    qc_policy,
    summarize_qc_records,
)
from src.common.process_jsonl import iter_jsonl, write_jsonl


JsonDict = Dict[str, Any]

GRADE_REQUIRES_STRENGTH_FOR_AUTO_ACCEPT = False

NOISY_GRADE_REJECT_RE = re.compile(
    r"\b(fragment|garbled|not (?:a )?(?:clear )?(?:grade|certainty|evidence)|"
    r"incomplete|administrative|reference|table header|not specific)\b",
    re.I,
)


def parsed(row: JsonDict) -> JsonDict:
    """读取已解析的 LLM 结果 payload；不存在时返回空对象。"""

    value = row.get("parsed_result")
    return value if isinstance(value, dict) else {}


def payload(row: JsonDict) -> JsonDict:
    """读取 normalized_payload；不存在时返回空对象。"""

    value = row.get("normalized_payload")
    return value if isinstance(value, dict) else {}


def index_by_id(rows: Iterable[JsonDict], id_field: str) -> Dict[str, JsonDict]:
    """按非空 ID 字段为行建立索引。"""

    indexed: Dict[str, JsonDict] = {}
    for row in rows:
        row_id = str(row.get(id_field) or "")
        if row_id:
            indexed[row_id] = row
    return indexed


def output_grade_id(output: JsonDict) -> str:
    """从 LLM 输出行中解析 grade_candidate_id。"""

    result = parsed(output)
    grade_id = result.get("grade_candidate_id")
    if isinstance(grade_id, str) and grade_id:
        return grade_id
    grade_ids = output.get("grade_candidate_ids")
    if isinstance(grade_ids, list) and grade_ids:
        return str(grade_ids[0] or "")
    return ""


def noisy_reject_reason(text: Any) -> bool:
    """判断 LLM 拒绝原因是否明确指向噪声。"""

    return bool(NOISY_GRADE_REJECT_RE.search(str(text or "")))


def association_reason(grade: JsonDict) -> str:
    """读取 GRADE 候选上保存的推荐关联原因。"""

    return str(payload(grade).get("association_reason") or "")


def _add_grade_field_scores(result: JsonDict, policy: JsonDict, reasons: List[str]) -> float:
    score = 0.0
    # GRADE 自动接受优先看 certainty；strength 在部分体系中不是必填，所以由开关控制。
    if field_present(result.get("certainty")):
        score += 0.08
        reasons.append("has_certainty")
    elif policy["requires_certainty_for_auto_accept"]:
        score -= 0.1
        reasons.append("qc_requires_certainty_but_unclear")

    if field_present(result.get("strength")):
        score += 0.08
        reasons.append("has_strength")
    elif GRADE_REQUIRES_STRENGTH_FOR_AUTO_ACCEPT and policy["requires_strength_for_auto_accept"]:
        score -= 0.1
        reasons.append("qc_requires_strength_but_unclear")
    return score


def _add_grade_system_score(result: JsonDict, policy: JsonDict, reasons: List[str]) -> float:
    grade_system = str(result.get("grade_system") or "unknown")
    if grade_system in policy["accepted_grade_systems"]:
        reasons.append(f"grade_system_allowed:{grade_system}")
        return 0.05
    reasons.append(f"grade_system_unexpected_for_profile:{grade_system}")
    return -0.12


def _add_grade_link_score(result: JsonDict, reasons: List[str]) -> float:
    if result.get("recommendation_candidate_id"):
        reasons.append("linked_recommendation_present")
        return 0.05
    reasons.append("missing_recommendation_link")
    return -0.08


def _add_grade_association_score(grade: JsonDict, reasons: List[str]) -> float:
    assoc = association_reason(grade)
    # same_block 关联最可靠；nearest_order 只是启发式匹配，需要降低自动合并信心。
    if assoc == "same_block":
        reasons.append("association_same_block")
        return 0.04
    if assoc.startswith("nearest_order"):
        reasons.append(f"association_{assoc}")
        return -0.04
    if assoc == "not_found":
        reasons.append("association_not_found")
        return -0.08
    return 0.0


def score_grade_output(output: JsonDict, grade: JsonDict, recommendation: JsonDict, profile: JsonDict) -> tuple[float, List[str]]:
    """评分判断 LLM GRADE 复核输出是否适合自动合并。"""

    result = parsed(output)
    policy = qc_policy(profile)
    reasons: List[str] = profile_reason_tokens(profile)
    score = 0.0

    score += add_output_status_score(output, reasons)
    confidence_score, _confidence_value = add_confidence_score(result, reasons)
    score += confidence_score
    score += add_validity_score(result, "is_valid_grade", "llm_valid_grade", "llm_rejected_grade", reasons)
    score += add_human_review_score(result, reasons)
    score += _add_grade_system_score(result, policy, reasons)
    score += _add_grade_field_scores(result, policy, reasons)
    score += _add_grade_link_score(result, reasons)
    score += _add_grade_association_score(grade, reasons)

    if noisy_reject_reason(result.get("reject_reason")):
        reasons.append("reject_reason_matches_grade_noise_rule")

    return clamped_score(score), reasons


def decide_grade(output: JsonDict, grade: JsonDict, profile: JsonDict, score: float) -> tuple[str, str, List[str]]:
    """把 GRADE QC 分数和校验信号转换为动作决策。"""

    result = parsed(output)
    policy = qc_policy(profile)
    reasons: List[str] = []

    if output.get("status") != "validated":
        return "needs_review", "fix_or_rerun_llm_output", ["llm_output_not_validated"]

    confidence = result.get("confidence")
    confidence_value = float(confidence) if isinstance(confidence, (int, float)) else 0.0

    if result.get("is_valid_grade") is False:
        # LLM 明确判定为噪声且置信低时可自动拒绝，其余 invalid 保留人工复核。
        if confidence_value <= 0.45 or noisy_reject_reason(result.get("reject_reason")):
            return "auto_reject", "mark_grade_candidate_rejected", ["invalid_grade_low_confidence_or_noise_reason"]
        return "needs_review", "manual_review_invalid_grade", ["llm_invalid_but_not_low_risk"]

    if result.get("is_valid_grade") is not True:
        return "needs_review", "manual_review_uncertain_grade_validity", ["missing_grade_validity_judgement"]

    if result.get("needs_human_review") is True:
        reasons.append("llm_requested_human_review")
    if confidence_value < float(policy["auto_accept_confidence"]):
        reasons.append("confidence_below_qc_threshold")
    if str(result.get("grade_system") or "unknown") not in policy["accepted_grade_systems"]:
        reasons.append("grade_system_not_allowed_for_profile")
    if policy["requires_certainty_for_auto_accept"] and not field_present(result.get("certainty")):
        reasons.append("qc_requires_certainty")
    if (
        GRADE_REQUIRES_STRENGTH_FOR_AUTO_ACCEPT
        and policy["requires_strength_for_auto_accept"]
        and not field_present(result.get("strength"))
    ):
        reasons.append("qc_requires_strength")
    if not result.get("recommendation_candidate_id"):
        reasons.append("missing_recommendation_link")

    if not reasons and score >= 0.75:
        return "auto_accept", "merge_llm_fields_into_enhanced_grade_candidate", ["passes_generic_grade_auto_accept"]

    return "needs_review", "manual_or_second_model_review", reasons or ["score_below_auto_accept_threshold"]


def build_qc_record(output: JsonDict, grade: JsonDict, recommendation: JsonDict, profile: JsonDict) -> JsonDict:
    """为一条 GRADE LLM 输出构造自动 QC 记录。"""

    policy_key = effective_policy_key(profile)
    policy = qc_policy(profile)
    score, score_reasons = score_grade_output(output, grade, recommendation, profile)
    decision, action, decision_reasons = decide_grade(output, grade, profile, score)
    result = parsed(output)
    grade_id = output_grade_id(output)
    return {
        "auto_qc_id": f"grade_auto_qc_{output.get('llm_output_id', output.get('queue_id', 'unknown'))}",
        "llm_output_id": output.get("llm_output_id", ""),
        "queue_id": output.get("queue_id", ""),
        "task_type": output.get("task_type", ""),
        "grade_candidate_id": grade_id,
        "recommendation_candidate_id": result.get("recommendation_candidate_id") or grade.get("recommendation_candidate_id", ""),
        "record_id": output.get("record_id", ""),
        "guideline_id": output.get("guideline_id", ""),
        "qc_policy_key": policy_key,
        "profile_context": profile_qc_context(profile),
        "source_url": grade.get("source_url", "") or recommendation.get("source_url", ""),
        "source_section": grade.get("source_section", "") or recommendation.get("source_section", ""),
        "qc_policy": {
            "accepted_grade_systems": sorted(policy["accepted_grade_systems"]),
            "requires_certainty_for_auto_accept": policy["requires_certainty_for_auto_accept"],
            "requires_strength_for_auto_accept": (
                GRADE_REQUIRES_STRENGTH_FOR_AUTO_ACCEPT and policy["requires_strength_for_auto_accept"]
            ),
            "auto_accept_confidence": policy["auto_accept_confidence"],
        },
        "association_reason": association_reason(grade),
        "llm_status": output.get("status"),
        "llm_is_valid_grade": result.get("is_valid_grade"),
        "llm_needs_human_review": result.get("needs_human_review"),
        "llm_confidence": result.get("confidence"),
        "llm_grade_system": result.get("grade_system"),
        "llm_strength": result.get("strength"),
        "llm_certainty": result.get("certainty"),
        "llm_reject_reason": result.get("reject_reason"),
        "auto_qc_score": score,
        "auto_qc_decision": decision,
        "auto_qc_reasons": score_reasons + decision_reasons,
        "recommended_action": action,
    }


def summarize_qc(records: List[JsonDict]) -> JsonDict:
    """汇总 GRADE 自动 QC 输出。"""

    return summarize_qc_records(records)


def build_auto_qc_file(
    outputs_input: str | Path,
    grades_input: str | Path,
    recommendations_input: str | Path,
    profiles_input: str | Path | None,
    qc_output: str | Path,
    summary_output: str | Path,
) -> JsonDict:
    """读取 LLM 输出和候选记录，写出 GRADE 自动 QC JSONL。"""

    grades = index_by_id(iter_jsonl(grades_input), "grade_candidate_id")
    recommendations = index_by_id(iter_jsonl(recommendations_input), "candidate_id")
    profiles = load_profiles(profiles_input)
    qc_records: List[JsonDict] = []
    for output in iter_jsonl(outputs_input):
        grade_id = output_grade_id(output)
        grade = grades.get(grade_id, {})
        rec_id = str(grade.get("recommendation_candidate_id") or parsed(output).get("recommendation_candidate_id") or "")
        recommendation = recommendations.get(rec_id, {})
        record_id = str(grade.get("record_id") or recommendation.get("record_id") or output.get("record_id") or "")
        qc_records.append(build_qc_record(output, grade, recommendation, profiles.get(record_id, {})))
    summary = summarize_qc(qc_records)
    summary["outputs_input"] = str(outputs_input)
    summary["grades_input"] = str(grades_input)
    summary["recommendations_input"] = str(recommendations_input)
    summary["profiles_input"] = str(profiles_input or "")
    write_jsonl(qc_output, qc_records)
    write_jsonl(summary_output, [summary])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply generic automatic QC to LLM grade review outputs.")
    parser.add_argument("--outputs-input", required=True)
    parser.add_argument("--grades-input", required=True)
    parser.add_argument("--recommendations-input", required=True)
    parser.add_argument("--profiles-input", default="data/processed/profiles/guideline_profiles.jsonl")
    parser.add_argument("--qc-output", required=True)
    parser.add_argument("--summary-output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_auto_qc_file(
        outputs_input=args.outputs_input,
        grades_input=args.grades_input,
        recommendations_input=args.recommendations_input,
        profiles_input=args.profiles_input,
        qc_output=args.qc_output,
        summary_output=args.summary_output,
    )
    print("qc_items={qc_items} decisions={decision_counts} policies={qc_policy_counts}".format(**summary))


if __name__ == "__main__":
    main()
