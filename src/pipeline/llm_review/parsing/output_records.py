"""LLM 复核文件：构建复核队列、prompt、响应解析和自动质检，让候选结果进入人工/模型辅助复核流程。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.domain.common import stable_id
from src.common.extraction_common import JsonDict, utc_now


@dataclass(frozen=True)
class ModelTraceInput:
    # LLM trace 需要同时记录请求、响应、校验和运行时；用参数对象避免调用方传错顺序。
    prompt_record: JsonDict
    item: JsonDict
    model: str
    api_url: str
    api_format: str
    temperature: float
    timeout_seconds: int
    max_retries: int
    retry_base_seconds: float
    max_response_bytes: int
    raw_response: Any
    parsed_result: JsonDict | None
    validation_errors: list[str]
    runtime_ms: int
    error_message: str | None


@dataclass(frozen=True)
class LLMOutputInput:
    # 输出记录只关心一次调用的产物，和模型请求配置分开，便于后续替换 LLM client。
    prompt_record: JsonDict
    item: JsonDict
    raw_response: Any
    response_text: str
    parsed_result: JsonDict | None
    validation_errors: list[str]
    model_trace_id: str
    runtime_ms: int
    error_message: str | None


def input_entity_id_for_item(item: JsonDict) -> str:
    if item.get("task_type") == "recommendation_candidate_review":
        return str(item.get("recommendation_candidate_id") or "")
    grade_ids = item.get("grade_candidate_ids")
    if isinstance(grade_ids, list) and grade_ids:
        return str(grade_ids[0] or "")
    return str(item.get("block_id") or item.get("queue_id") or "")


def trace_task_type(task_type: str) -> str:
    if task_type == "grade_candidate_review":
        return "grade_extraction"
    return "recommendation_extraction"


def trace_input_entity_type(task_type: str) -> str:
    if task_type == "grade_candidate_review":
        return "block"
    return "recommendation_candidate"


def build_model_trace(data: ModelTraceInput) -> JsonDict:
    success = data.error_message is None and not data.validation_errors and data.parsed_result is not None
    response_json = data.raw_response if isinstance(data.raw_response, dict) else {}
    return {
        "model_trace_id": stable_id("model_trace", "llm_review", data.prompt_record.get("prompt_id"), data.model),
        "task_type": trace_task_type(str(data.item.get("task_type") or "")),
        "method": "model",
        "model_name": data.model,
        "input_entity_type": trace_input_entity_type(str(data.item.get("task_type") or "")),
        "input_entity_id": input_entity_id_for_item(data.item),
        "input_text": data.prompt_record.get("prompt", ""),
        "model_version": data.model,
        "prompt_version": data.prompt_record.get("prompt_version"),
        "raw_output": data.raw_response,
        "parsed_output": data.parsed_result,
        "confidence": data.parsed_result.get("confidence") if isinstance(data.parsed_result, dict) else None,
        "parameters": {
            "api_url": data.api_url,
            "api_format": data.api_format,
            "temperature": data.temperature,
            "timeout_seconds": data.timeout_seconds,
            "max_retries": data.max_retries,
            "retry_base_seconds": data.retry_base_seconds,
            "max_response_bytes": data.max_response_bytes,
            "queue_id": data.item.get("queue_id", ""),
            "queue_task_type": data.item.get("task_type", ""),
            "priority": data.item.get("priority", ""),
            "validation_errors": data.validation_errors,
        },
        "token_usage": response_json.get("usage", {}) if isinstance(response_json.get("usage"), dict) else {},
        "runtime_ms": data.runtime_ms,
        "code_version": None,
        "success": success,
        "error_message": data.error_message or (";".join(data.validation_errors) if data.validation_errors else None),
        "human_verified": False,
        "verified_by": None,
        "verified_at": None,
        "target_table": None,
        "target_entity_id": None,
        "created_at": utc_now(),
    }


def build_llm_output_record(data: LLMOutputInput) -> JsonDict:
    # 状态优先级按故障链路排序：API 失败 > 解析失败 > 校验失败 > 可用结果。
    if data.error_message:
        status = "api_failed"
    elif data.parsed_result is None:
        status = "parse_failed"
    elif data.validation_errors:
        status = "validation_failed"
    else:
        status = "validated"
    return {
        "llm_output_id": stable_id("llm_output", data.prompt_record.get("prompt_id"), data.model_trace_id),
        "queue_id": data.item.get("queue_id", ""),
        "prompt_id": data.prompt_record.get("prompt_id", ""),
        "model_trace_id": data.model_trace_id,
        "task_type": data.item.get("task_type", ""),
        "priority": data.item.get("priority", ""),
        "record_id": data.item.get("record_id", ""),
        "guideline_id": data.item.get("guideline_id", ""),
        "recommendation_candidate_id": data.item.get("recommendation_candidate_id", ""),
        "grade_candidate_ids": data.item.get("grade_candidate_ids", []),
        "raw_response": data.raw_response,
        "response_text": data.response_text,
        "parsed_result": data.parsed_result,
        "validation_errors": data.validation_errors,
        "status": status,
        "runtime_ms": data.runtime_ms,
        "error_message": data.error_message,
        "created_at": utc_now(),
    }

