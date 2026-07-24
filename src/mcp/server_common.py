"""MCP 服务入口共享的配置、请求载荷和启动逻辑。"""

from __future__ import annotations

import os
from typing import Any


def env_bool(name: str, default: bool = False) -> bool:
    """读取布尔环境变量，未配置时返回调用方默认值。"""

    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    """读取整数环境变量；非法值按可选配置处理并回退默认值。"""

    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def search_payload(
    query: str,
    source_institution: str | None,
    clinical_department: str | None,
    time_range: str | None,
    publication_date: str | None,
    recency_boost: bool,
    topk: int,
) -> dict[str, Any]:
    """构造文档搜索工具的稳定请求结构。"""

    return {
        "query": query,
        "source_institution": source_institution,
        "clinical_department": clinical_department,
        "time_range": time_range,
        "publication_date": publication_date,
        "recency_boost": recency_boost,
        "topk": topk,
    }


def read_payload(
    doc_id: str | None, title: str | None, max_chars: int | None
) -> dict[str, Any]:
    """构造全文读取工具的稳定请求结构。"""

    return {"doc_id": doc_id, "title": title, "max_chars": max_chars}


def retrieve_payload(
    query: str,
    source_institution: str | None,
    clinical_department: str | None,
    time_range: str | None,
    publication_date: str | None,
    topk: int,
) -> dict[str, Any]:
    """构造分块检索工具的稳定请求结构。"""

    return {
        "query": query,
        "source_institution": source_institution,
        "clinical_department": clinical_department,
        "time_range": time_range,
        "publication_date": publication_date,
        "topk": topk,
    }


def run_server(mcp: Any) -> None:
    """校验传输配置并启动 MCP 服务。"""

    if mcp is None:
        raise RuntimeError(
            "Install mcp>=1.9 to run the MCP server: python -m pip install mcp>=1.9"
        )
    transport = os.environ.get("MCP_TRANSPORT", "stdio").strip().lower()
    if transport not in {"stdio", "sse", "streamable-http"}:
        raise ValueError(
            "MCP_TRANSPORT must be one of: stdio, sse, streamable-http"
        )
    mcp.run(
        transport=transport,
        mount_path=os.environ.get("MCP_MOUNT_PATH"),
    )
