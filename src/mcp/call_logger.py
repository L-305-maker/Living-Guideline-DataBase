# MCP 工具调用的 JSONL 审计日志。
#
# 关键设计：
# - 每条调用追加 1 行 JSON 到 logs/mcp_calls.jsonl（含 call_id / timestamp / tool / payload / result / duration）；
# - 敏感字段（password / token / api_key 等）写入前替换为 '***'；
# - 字符串按 MCP_CALL_LOG_MAX_*_STRING_CHARS 截断，列表按 *_LIST_ITEMS 截断；
# - result_mode = summary / full / none：full 写 result 全量，summary 只写 result_summary，none 不写 result；
# - 业务异常必须原样重新抛出，审计写入失败不能中断主调用链路。
"""JSONL call logging for MCP tool invocations."""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar
import queue, threading
# 守护线程并发 drain 时共享的日志队列；模块顶层定义, 命名统一为 _Q
# (此前同时出现 Q 与 _Q 两个不一致的名称, 导致 drainer 启动即 NameError)。
_Q: queue.Queue = queue.Queue(maxsize=100_000)


from src.mcp.server_common import (
    env_bool as helper_env_bool,
    env_int as helper_env_int,
)


ResultT = TypeVar("ResultT")

# 写入审计前会被替换为 '***' 的敏感字段名（lowercase 比较）。
SENSITIVE_KEYS = {"password", "passwd", "secret", "token", "api_key", "apikey", "authorization"}

# 默认日志截断阈值；可被同名环境变量覆盖（payload/result 各一套）。
DEFAULT_MAX_STRING_CHARS = 700
DEFAULT_MAX_RESULT_STRING_CHARS = 8000
DEFAULT_MAX_LIST_ITEMS = 20
DEFAULT_MAX_RESULT_LIST_ITEMS = 50


def helper_env_int_from(names: list[str], default: int) -> int:
    # 按顺序尝试读多个环境变量名，第一个非 None 即返回其 int 值。
    for name in names:
        if os.environ.get(name) is not None:
            return helper_env_int(name, default)
    return default


def helper_log_path() -> Path:
    # 计算审计日志路径：MCP_CALL_LOG_PATH > RAG_DATA_DIR/logs/mcp_calls.jsonl。
    configured = os.environ.get("MCP_CALL_LOG_PATH")
    if configured:
        return Path(configured)
    data_dir = Path(os.environ.get("RAG_DATA_DIR", "data/evidence"))
    return data_dir / "logs" / "mcp_calls.jsonl"


def helper_truncate_string(value: str, limit: int, compact: bool = False) -> str:
    # 字符串截断 + 标记省略长度（limit ≤ 0 时不截断）。

    # compact=True 时先压缩连续空白（避免行内换行被算字符）。
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
    # 递归脱敏 + 截断：dict 跳过敏感键、list 限制长度、str 截断、其它 str() 后截断。
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
    # 构造 result 的轻量摘要（避免在 result_mode=summary 时写全量）。
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


def drainer():
    while True:
        records = []
        try:
            while len(records) < 100:
                records.append(_Q.get_nowait())
        except queue.Empty:
            pass
        if not records:
            records.append(_Q.get())  # 阻塞等下一次（命名与定义一致, fix bug #1）
        try:
            with helper_log_path().open("a", encoding="utf-8", newline="\n") as h:
                for r in records:
                    h.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        except OSError:
            pass  # 与现状一致

threading.Thread(target=drainer, name="mcp-log-drain", daemon=True).start()


def flush(timeout: float = 2.0) -> None:
    """同步等待 drainer 把队列里现有 records 全部写盘（带 timeout）。

    设计动机：
    - 高并发热路径上异步 drain 是对的（fsync 抖动不会传到请求主链路）；
    - 但测试 / 进程退出 / 单元校验场景需要"看完队列再继续"。
    - 用 _Q.join() 等价于"等所有 task 完成"，drainer 的循环本身就是 task。

    用法（仅测试/进程退出用）：
        from src.mcp.call_logger import flush
        flush(timeout=2.0)
    """
    import time
    deadline = time.perf_counter() + timeout
    while _Q.qsize() > 0 and time.perf_counter() < deadline:
        time.sleep(0.02)  # drainer 醒着每 0ms 拉一次


def helper_write_record(record):
    if not helper_env_bool("MCP_CALL_LOG_ENABLED", True):
        return
    try:
        _Q.put_nowait(record)
    except queue.Full:
        pass  # 满则丢; 不影响主调用


def log_mcp_call(tool_name: str, backend: str, payload: dict[str, Any], call: Callable[[], ResultT]) -> ResultT:
    # 执行 MCP 工具调用并追加 1 条 JSONL 审计记录。

    # 行为：
    # - 自动加 call_id (uuid4) / timestamp (UTC) / duration_ms 等元字段；
    # - payload 与 result 都经 helper_sanitize 脱敏 + 截断；
    # - result_mode ∈ {full, summary, summary_only, none, off}：full 写 result；summary/ summary_only 写 result_summary；none/ off 不写 result；
    # - 业务异常会被记录（status=error, error_type, error）后原样 re-raise，不吞错。
    
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