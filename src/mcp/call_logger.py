"""JSONL call logging for MCP tool invocations."""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar


ResultT = TypeVar("ResultT")

SENSITIVE_KEYS = {"password", "passwd", "secret", "token", "api_key", "apikey", "authorization"}
DEFAULT_MAX_STRING_CHARS = 700
DEFAULT_MAX_RESULT_STRING_CHARS = 8000
DEFAULT_MAX_LIST_ITEMS = 20
DEFAULT_MAX_RESULT_LIST_ITEMS = 50


def helper_env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def helper_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def helper_env_int_from(names: list[str], default: int) -> int:
    for name in names:
        if os.environ.get(name) is not None:
            return helper_env_int(name, default)
    return default


def helper_log_path() -> Path:
    configured = os.environ.get("MCP_CALL_LOG_PATH")
    if configured:
        return Path(configured)
    data_dir = Path(os.environ.get("RAG_DATA_DIR", "data/evidence"))
    return data_dir / "logs" / "mcp_calls.jsonl"


def helper_truncate_string(value: str, limit: int, compact: bool = False) -> str:
    if compact:
        value = " ".join(value.split())
    if limit <= 0:
        return value
    if len(value) <= limit:
        return value
    omitted = len(value) - limit
    suffix = f"...[truncated {omitted} chars]"
    return value[: max(0, limit - len(suffix))].rstrip() + suffix


def helper_sanitize(value: Any, max_string_chars: int, max_list_items: int, compact_strings: bool) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.lower() in SENSITIVE_KEYS:
                sanitized[key_text] = "***"
            else:
                sanitized[key_text] = helper_sanitize(item, max_string_chars, max_list_items, compact_strings)
        return sanitized
    if isinstance(value, list):
        limited = [helper_sanitize(item, max_string_chars, max_list_items, compact_strings) for item in value[:max_list_items]]
        if len(value) > max_list_items:
            limited.append({"truncated_items": len(value) - max_list_items})
        return limited
    if isinstance(value, tuple):
        return helper_sanitize(list(value), max_string_chars, max_list_items, compact_strings)
    if isinstance(value, str):
        return helper_truncate_string(value, max_string_chars, compact=compact_strings)
    if isinstance(value, int | float | bool) or value is None:
        return value
    return helper_truncate_string(str(value), max_string_chars, compact=compact_strings)


def helper_result_summary(result: Any, max_list_items: int) -> dict[str, Any]:
    if isinstance(result, list):
        ids = []
        for item in result[:max_list_items]:
            if isinstance(item, dict):
                ids.append(item.get("doc_id") or item.get("chunk_id") or item.get("id"))
        return {
            "type": "list",
            "count": len(result),
            "sample_ids": [item_id for item_id in ids if item_id],
        }
    if isinstance(result, dict):
        summary: dict[str, Any] = {"type": "dict", "keys": sorted(str(key) for key in result.keys())[:max_list_items]}
        for id_key in ("doc_id", "chunk_id", "id"):
            if id_key in result:
                summary[id_key] = result[id_key]
                break
        return summary
    return {"type": type(result).__name__}


def helper_write_record(record: dict[str, Any]) -> None:
    if not helper_env_bool("MCP_CALL_LOG_ENABLED", True):
        return
    try:
        path = helper_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    except OSError:
        # Logging should never break the MCP tool call path.
        return


def log_mcp_call(tool_name: str, backend: str, payload: dict[str, Any], call: Callable[[], ResultT]) -> ResultT:
    """Execute an MCP tool call and append one JSONL audit record."""

    payload_max_string_chars = helper_env_int_from(
        ["MCP_CALL_LOG_MAX_PAYLOAD_STRING_CHARS", "MCP_CALL_LOG_MAX_STRING_CHARS"],
        DEFAULT_MAX_STRING_CHARS,
    )
    payload_max_list_items = helper_env_int_from(
        ["MCP_CALL_LOG_MAX_PAYLOAD_LIST_ITEMS", "MCP_CALL_LOG_MAX_LIST_ITEMS"],
        DEFAULT_MAX_LIST_ITEMS,
    )
    result_max_string_chars = helper_env_int_from(
        ["MCP_CALL_LOG_MAX_RESULT_STRING_CHARS", "MCP_CALL_LOG_MAX_STRING_CHARS"],
        DEFAULT_MAX_RESULT_STRING_CHARS,
    )
    result_max_list_items = helper_env_int_from(
        ["MCP_CALL_LOG_MAX_RESULT_LIST_ITEMS", "MCP_CALL_LOG_MAX_LIST_ITEMS"],
        DEFAULT_MAX_RESULT_LIST_ITEMS,
    )
    result_mode = os.environ.get("MCP_CALL_LOG_RESULT_MODE", "full").strip().lower()
    started = time.perf_counter()
    record: dict[str, Any] = {
        "call_id": uuid.uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool_name,
        "backend": backend,
        "transport": os.environ.get("MCP_TRANSPORT", "stdio"),
        "pid": os.getpid(),
        "payload": helper_sanitize(payload, payload_max_string_chars, payload_max_list_items, compact_strings=True),
    }
    try:
        result = call()
    except Exception as exc:
        record.update(
            {
                "status": "error",
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                "error_type": type(exc).__name__,
                "error": helper_truncate_string(str(exc), payload_max_string_chars, compact=True),
            }
        )
        helper_write_record(record)
        raise

    result_summary = helper_result_summary(result, result_max_list_items)
    record.update(
        {
            "status": "ok",
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            "result_summary": result_summary,
        }
    )
    if result_mode not in {"summary", "summary_only", "none", "off"}:
        record["result"] = helper_sanitize(result, result_max_string_chars, result_max_list_items, compact_strings=False)
    helper_write_record(record)
    return result
