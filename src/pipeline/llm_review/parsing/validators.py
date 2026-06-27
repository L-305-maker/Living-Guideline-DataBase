"""LLM 复核文件：构建复核队列、prompt、响应解析和自动质检，让候选结果进入人工/模型辅助复核流程。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

from typing import Any

from src.pipeline.llm_review.prompts.builder import (
    CERTAINTY_VALUES,
    DIRECTION_VALUES,
    GRADE_DOMAIN_VALUES,
    GRADE_SYSTEM_VALUES,
    PUBLICATION_BIAS_VALUES,
    STRENGTH_VALUES,
)
from src.common.extraction_common import JsonDict


def _is_optional_string(value: Any) -> bool:
    return value is None or isinstance(value, str)


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _is_number_0_to_1(value: Any) -> bool:
    return isinstance(value, (int, float)) and 0 <= float(value) <= 1


def _missing_fields(result: JsonDict, required: list[str]) -> list[str]:
    return [f"missing_field:{field}" for field in required if field not in result]


def validate_recommendation_result(result: JsonDict) -> list[str]:
    errors = _missing_fields(
        result,
        [
            "queue_id",
            "recommendation_candidate_id",
            "is_valid_recommendation",
            "corrected_recommendation_text",
            "direction",
            "strength",
            "certainty",
            "population",
            "intervention",
            "comparator",
            "outcomes",
            "rationale",
            "remarks",
            "reject_reason",
            "needs_human_review",
            "confidence",
        ],
    )
    if errors:
        return errors
    if not isinstance(result["queue_id"], str):
        errors.append("queue_id_must_be_string")
    if not isinstance(result["recommendation_candidate_id"], str):
        errors.append("recommendation_candidate_id_must_be_string")
    if not isinstance(result["is_valid_recommendation"], bool):
        errors.append("is_valid_recommendation_must_be_boolean")
    if not _is_optional_string(result["corrected_recommendation_text"]):
        errors.append("corrected_recommendation_text_must_be_string_or_null")
    if result["direction"] not in DIRECTION_VALUES:
        errors.append("invalid_direction")
    if result["strength"] not in STRENGTH_VALUES:
        errors.append("invalid_strength")
    if result["certainty"] not in CERTAINTY_VALUES:
        errors.append("invalid_certainty")
    for field in ["population", "intervention", "comparator", "rationale", "remarks", "reject_reason"]:
        if not _is_optional_string(result[field]):
            errors.append(f"{field}_must_be_string_or_null")
    if not isinstance(result["outcomes"], list):
        errors.append("outcomes_must_be_list")
    if not isinstance(result["needs_human_review"], bool):
        errors.append("needs_human_review_must_be_boolean")
    if not _is_number_0_to_1(result["confidence"]):
        errors.append("confidence_must_be_number_between_0_and_1")
    return errors


def _validate_grade_identity_fields(result: JsonDict, errors: list[str]) -> None:
    if not isinstance(result["queue_id"], str):
        errors.append("queue_id_must_be_string")
    if not isinstance(result["grade_candidate_id"], str):
        errors.append("grade_candidate_id_must_be_string")
    if not _is_optional_string(result["recommendation_candidate_id"]):
        errors.append("recommendation_candidate_id_must_be_string_or_null")
    if not isinstance(result["is_valid_grade"], bool):
        errors.append("is_valid_grade_must_be_boolean")


def _validate_grade_enum_fields(result: JsonDict, errors: list[str]) -> None:
    if result["grade_system"] not in GRADE_SYSTEM_VALUES:
        errors.append("invalid_grade_system")
    if result["certainty"] not in CERTAINTY_VALUES:
        errors.append("invalid_certainty")
    if result["strength"] not in STRENGTH_VALUES:
        errors.append("invalid_strength")
    for field in ["risk_of_bias", "inconsistency", "indirectness", "imprecision"]:
        if result[field] not in GRADE_DOMAIN_VALUES:
            errors.append(f"invalid_{field}")
    if result["publication_bias"] not in PUBLICATION_BIAS_VALUES:
        errors.append("invalid_publication_bias")


def _validate_grade_list_and_review_fields(result: JsonDict, errors: list[str]) -> None:
    if not _is_string_list(result["reasons_for_downgrade"]):
        errors.append("reasons_for_downgrade_must_be_list_of_strings")
    if not _is_string_list(result["reasons_for_upgrade"]):
        errors.append("reasons_for_upgrade_must_be_list_of_strings")
    if not _is_optional_string(result["reject_reason"]):
        errors.append("reject_reason_must_be_string_or_null")
    if not isinstance(result["needs_human_review"], bool):
        errors.append("needs_human_review_must_be_boolean")
    if not _is_number_0_to_1(result["confidence"]):
        errors.append("confidence_must_be_number_between_0_and_1")


def validate_grade_result(result: JsonDict) -> list[str]:
    errors = _missing_fields(
        result,
        [
            "queue_id",
            "grade_candidate_id",
            "recommendation_candidate_id",
            "is_valid_grade",
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
            "reject_reason",
            "needs_human_review",
            "confidence",
        ],
    )
    if errors:
        return errors
    _validate_grade_identity_fields(result, errors)
    _validate_grade_enum_fields(result, errors)
    _validate_grade_list_and_review_fields(result, errors)
    return errors


def validate_llm_result(task_type: str, result: JsonDict) -> list[str]:
    return validate_llm_result_for_item(task_type, result)


def validate_llm_result_for_item(task_type: str, result: JsonDict, queue_item: JsonDict | None = None) -> list[str]:
    if task_type == "recommendation_candidate_review":
        errors = validate_recommendation_result(result)
        if queue_item:
            if result.get("queue_id") != queue_item.get("queue_id"):
                errors.append("queue_id_does_not_match_queue_item")
            if result.get("recommendation_candidate_id") != queue_item.get("recommendation_candidate_id"):
                errors.append("recommendation_candidate_id_does_not_match_queue_item")
        return errors
    if task_type == "grade_candidate_review":
        errors = validate_grade_result(result)
        if queue_item:
            grade_ids = queue_item.get("grade_candidate_ids")
            expected_grade_id = grade_ids[0] if isinstance(grade_ids, list) and grade_ids else ""
            if result.get("queue_id") != queue_item.get("queue_id"):
                errors.append("queue_id_does_not_match_queue_item")
            if result.get("grade_candidate_id") != expected_grade_id:
                errors.append("grade_candidate_id_does_not_match_queue_item")
            expected_rec_id = queue_item.get("recommendation_candidate_id") or None
            if result.get("recommendation_candidate_id") != expected_rec_id:
                errors.append("recommendation_candidate_id_does_not_match_queue_item")
        return errors
    return [f"unsupported_task_type:{task_type}"]

