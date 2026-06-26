from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.pipeline.llm_review.auto_qc.common import (
    add_confidence_score,
    add_human_review_score,
    add_output_status_score,
    add_validity_score,
    clamped_score,
    effective_policy_key,
    field_present,
    group_grades_by_recommendation,
    index_by_id,
    load_profiles,
    noisy_reject_reason,
    parsed,
    payload,
    profile_qc_context,
    profile_reason_tokens,
    qc_policy,
    summarize_qc_records,
)
from src.common.extraction_common import JsonDict
from src.common.process_jsonl import iter_jsonl, write_jsonl


def best_grade_system(grades: list[JsonDict]) -> str:
    """返回第一个有信息量的关联 GRADE 体系；没有时返回 unknown。"""

    for grade in grades:
        system = str(grade.get("grade_system") or "unknown")
        if system != "unknown":
            return system
    return "unknown"


def has_informative_grade(grades: list[JsonDict]) -> bool:
    """判断关联 GRADE 是否能补充 certainty 或 strength。"""

    return any(grade.get("certainty") != "unclear" or grade.get("strength") != "unclear" for grade in grades)


def quality_notes(candidate: JsonDict) -> list[str]:
    """读取 recommendation 候选上的抽取质量备注。"""

    notes = payload(candidate).get("quality_notes")
    return [str(note) for note in notes] if isinstance(notes, list) else []


def _add_recommendation_field_scores(result: JsonDict, policy: JsonDict, reasons: list[str]) -> float:
    score = 0.0
    # 自动接受需要 LLM 给出可合并的临床核心字段；缺少 profile 要求的字段会降低置信。
    for field, value, reason in [
        ("corrected_recommendation_text", 0.08, "has_corrected_recommendation_text"),
        ("population", 0.04, "has_population"),
        ("intervention", 0.04, "has_intervention"),
    ]:
        if field_present(result.get(field)):
            score += value
            reasons.append(reason)
    for field, required_key in [("strength", "requires_strength_for_auto_accept"), ("certainty", "requires_certainty_for_auto_accept")]:
        if field_present(result.get(field)):
            score += 0.04
            reasons.append(f"has_{field}")
        elif policy[required_key]:
            score -= 0.08
            reasons.append(f"qc_requires_{field}_but_unclear")
    return score


def _add_grade_context_score(grades: list[JsonDict], policy: JsonDict, reasons: list[str]) -> float:
    score = 0.0
    grade_system = best_grade_system(grades)
    # 推荐候选的 QC 要参考已关联 GRADE；分级体系和 profile 不一致时不应自动合并。
    if grade_system in policy["accepted_grade_systems"]:
        score += 0.03
        reasons.append(f"grade_system_allowed:{grade_system}")
    else:
        score -= 0.08
        reasons.append(f"grade_system_unexpected_for_profile:{grade_system}")

    if has_informative_grade(grades):
        score += 0.04
        reasons.append("has_informative_linked_grade")
    return score


def _add_candidate_quality_penalty(candidate: JsonDict, reasons: list[str]) -> float:
    score = 0.0
    notes = quality_notes(candidate)
    if notes:
        reasons.extend(f"candidate_quality:{note}" for note in notes)
    if any(note in {"long_statement", "starts_with_fragment", "administrative_text"} for note in notes):
        score -= 0.1
    return score


def score_output(output: JsonDict, candidate: JsonDict, grades: list[JsonDict], profile: JsonDict) -> tuple[float, list[str]]:
    """评分判断 LLM recommendation 复核输出是否适合自动合并。"""

    result = parsed(output)
    policy = qc_policy(profile)
    reasons = profile_reason_tokens(profile)
    score = 0.0

    score += add_output_status_score(output, reasons)
    confidence_score, _confidence_value = add_confidence_score(result, reasons)
    score += confidence_score
    score += add_validity_score(result, "is_valid_recommendation", "llm_valid_recommendation", "llm_rejected_recommendation", reasons)
    score += add_human_review_score(result, reasons)
    score += _add_recommendation_field_scores(result, policy, reasons)
    score += _add_grade_context_score(grades, policy, reasons)
    score += _add_candidate_quality_penalty(candidate, reasons)

    if noisy_reject_reason(result.get("reject_reason")):
        reasons.append("reject_reason_matches_noise_rule")

    return clamped_score(score), reasons


def decide(output: JsonDict, candidate: JsonDict, grades: list[JsonDict], profile: JsonDict, score: float) -> tuple[str, str, list[str]]:
    """把 QC 分数和校验信号转换为动作决策。"""

    result = parsed(output)
    policy = qc_policy(profile)
    reasons: list[str] = []

    if output.get("status") != "validated":
        return "needs_review", "fix_or_rerun_llm_output", ["llm_output_not_validated"]

    confidence = result.get("confidence")
    confidence_value = float(confidence) if isinstance(confidence, (int, float)) else 0.0

    if result.get("is_valid_recommendation") is False:
        # 高置信的噪声拒绝可以自动落库，普通 invalid 仍留给人工避免误杀。
        if confidence_value <= 0.45 or noisy_reject_reason(result.get("reject_reason")):
            return "auto_reject", "mark_candidate_rejected", ["invalid_candidate_low_confidence_or_noise_reason"]
        return "needs_review", "manual_review_invalid_candidate", ["llm_invalid_but_not_low_risk"]

    if result.get("is_valid_recommendation") is not True:
        return "needs_review", "manual_review_uncertain_validity", ["missing_validity_judgement"]

    if result.get("needs_human_review") is True:
        reasons.append("llm_requested_human_review")
    if confidence_value < float(policy["auto_accept_confidence"]):
        reasons.append("confidence_below_qc_threshold")
    if policy["requires_strength_for_auto_accept"] and not field_present(result.get("strength")):
        reasons.append("qc_requires_strength")
    if policy["requires_certainty_for_auto_accept"] and not field_present(result.get("certainty")):
        reasons.append("qc_requires_certainty")
    if not field_present(result.get("population")) or not field_present(result.get("intervention")):
        reasons.append("missing_core_pico_fields")

    if not reasons and score >= 0.75:
        return "auto_accept", "merge_llm_fields_into_enhanced_candidate", ["passes_generic_auto_accept"]

    return "needs_review", "manual_or_second_model_review", reasons or ["score_below_auto_accept_threshold"]


def build_qc_record(output: JsonDict, candidate: JsonDict, grades: list[JsonDict], profile: JsonDict) -> JsonDict:
    """为一条 recommendation LLM 输出构造自动 QC 记录。"""

    policy_key = effective_policy_key(profile)
    policy = qc_policy(profile)
    score, score_reasons = score_output(output, candidate, grades, profile)
    decision, action, decision_reasons = decide(output, candidate, grades, profile, score)
    result = parsed(output)
    return {
        "auto_qc_id": f"auto_qc_{output.get('llm_output_id', output.get('queue_id', 'unknown'))}",
        "llm_output_id": output.get("llm_output_id", ""),
        "queue_id": output.get("queue_id", ""),
        "task_type": output.get("task_type", ""),
        "candidate_id": output.get("recommendation_candidate_id", ""),
        "record_id": output.get("record_id", ""),
        "guideline_id": output.get("guideline_id", ""),
        "qc_policy_key": policy_key,
        "profile_context": profile_qc_context(profile),
        "source_url": candidate.get("source_url", ""),
        "source_section": candidate.get("source_section", ""),
        "qc_policy": {
            "accepted_grade_systems": sorted(policy["accepted_grade_systems"]),
            "requires_certainty_for_auto_accept": policy["requires_certainty_for_auto_accept"],
            "requires_strength_for_auto_accept": policy["requires_strength_for_auto_accept"],
            "auto_accept_confidence": policy["auto_accept_confidence"],
        },
        "linked_grade_candidate_ids": [grade.get("grade_candidate_id", "") for grade in grades],
        "linked_grade_system": best_grade_system(grades),
        "llm_status": output.get("status"),
        "llm_is_valid_recommendation": result.get("is_valid_recommendation"),
        "llm_needs_human_review": result.get("needs_human_review"),
        "llm_confidence": result.get("confidence"),
        "llm_strength": result.get("strength"),
        "llm_certainty": result.get("certainty"),
        "llm_reject_reason": result.get("reject_reason"),
        "auto_qc_score": score,
        "auto_qc_decision": decision,
        "auto_qc_reasons": score_reasons + decision_reasons,
        "recommended_action": action,
    }


def summarize_qc(records: list[JsonDict]) -> JsonDict:
    """汇总 recommendation 自动 QC 输出。"""

    return summarize_qc_records(records)


def build_auto_qc_file(
    outputs_input: str | Path,
    recommendations_input: str | Path,
    grades_input: str | Path,
    profiles_input: str | Path | None,
    qc_output: str | Path,
    summary_output: str | Path,
) -> JsonDict:
    """读取 LLM 输出和候选记录，写出 recommendation 自动 QC JSONL。"""

    recommendations = index_by_id(iter_jsonl(recommendations_input), "candidate_id")
    grades_by_rec = group_grades_by_recommendation(iter_jsonl(grades_input))
    profiles = load_profiles(profiles_input)
    qc_records: list[JsonDict] = []
    for output in iter_jsonl(outputs_input):
        candidate_id = str(output.get("recommendation_candidate_id") or "")
        candidate = recommendations.get(candidate_id, {})
        grades = grades_by_rec.get(candidate_id, [])
        profile = profiles.get(str(candidate.get("record_id") or output.get("record_id") or ""), {})
        qc_records.append(build_qc_record(output, candidate, grades, profile))
    summary = summarize_qc(qc_records)
    summary["outputs_input"] = str(outputs_input)
    summary["recommendations_input"] = str(recommendations_input)
    summary["grades_input"] = str(grades_input)
    summary["profiles_input"] = str(profiles_input or "")
    write_jsonl(qc_output, qc_records)
    write_jsonl(summary_output, [summary])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply generic automatic QC to LLM review outputs.")
    parser.add_argument("--outputs-input", required=True)
    parser.add_argument("--recommendations-input", required=True)
    parser.add_argument("--grades-input", required=True)
    parser.add_argument("--profiles-input", default="data/processed/profiles/guideline_profiles.jsonl")
    parser.add_argument("--qc-output", required=True)
    parser.add_argument("--summary-output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_auto_qc_file(
        outputs_input=args.outputs_input,
        recommendations_input=args.recommendations_input,
        grades_input=args.grades_input,
        profiles_input=args.profiles_input,
        qc_output=args.qc_output,
        summary_output=args.summary_output,
    )
    print("qc_items={qc_items} decisions={decision_counts} policies={qc_policy_counts}".format(**summary))


if __name__ == "__main__":
    main()
