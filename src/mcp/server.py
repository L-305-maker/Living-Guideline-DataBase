# MCP server：暴露 search / read / retrieve 三个工具的 FastMCP 入口。
#
# 关键设计：
# - BACKEND = "postgres"：当前仅 PG 后端，避免误连旧 SQLite；
# - 三个 @mcp.tool() 装饰的函数全部经 helper_call_api → log_mcp_call 包装：
#   自动审计 + 异常原样抛出；
# - SEARCH_DEFAULT_TOPK = 10 / RETRIEVE_DEFAULT_TOPK = 5：合理上限，避免单次返回过大；
# - host / port / sse_path / message_path 全部从环境变量读，sse 与 streamable-http 模式可同进程并存。
"""MCP server exposing PostgreSQL Search, Read, and Retrieve tools."""

from __future__ import annotations

import os
from typing import Any

from src.mcp.call_logger import log_mcp_call
from src.mcp.api_pg import read_pg as read_api
from src.mcp.api_pg import retrieve_pg as retrieve_api
from src.mcp.api_pg import search_pg as search_api
from src.mcp.server_common import (
    env_bool as helper_env_bool,
    env_int as helper_env_int,
    read_payload,
    retrieve_payload,
    run_server,
    search_payload,
)

BACKEND = "postgres"
SEARCH_DEFAULT_TOPK = 10
RETRIEVE_DEFAULT_TOPK = 5

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover
    FastMCP = None  # type: ignore[assignment]


# 单一 FastMCP 实例：name 'pdf-markdown-rag-postgres' 与 systemd 单元中的 MCP 服务名一致。
# 缺 mcp>=1.9 时 mcp=None，工具装饰全部跳过，避免冷启动 import 失败。
mcp = (
    FastMCP(
        "pdf-markdown-rag-postgres",
        host=os.environ.get("MCP_HOST", "127.0.0.1"),
        port=helper_env_int("MCP_PORT", 8000),
        streamable_http_path=os.environ.get("MCP_STREAMABLE_HTTP_PATH", "/mcp"),
        sse_path=os.environ.get("MCP_SSE_PATH", "/sse"),
        message_path=os.environ.get("MCP_MESSAGE_PATH", "/messages/"),
        json_response=helper_env_bool("MCP_JSON_RESPONSE", False),
        stateless_http=helper_env_bool("MCP_STATELESS_HTTP", False),
    )
    if FastMCP
    else None
)


def helper_call_api(tool_name: str, api: Any, payload: dict[str, Any]) -> Any:
    """统一包装：所有 MCP 工具都走 log_mcp_call 审计 + 异常透传。"""
    return log_mcp_call(tool_name, BACKEND, payload, lambda: api(payload))


if mcp is not None:

    @mcp.tool()
    def search(
        query: str,
        source_institution: str | None = None,
        clinical_department: str | None = None,
        time_range: str | None = None,
        publication_date: str | None = None,
        recency_boost: bool = False,
        topk: int = SEARCH_DEFAULT_TOPK,
    ) -> list[dict[str, Any]]:
        """文档级检索：从 PostgreSQL BM25 + 向量索引召回候选指南文档。"""
        return helper_call_api(
            "search",
            search_api,
            search_payload(
                query,
                source_institution,
                clinical_department,
                time_range,
                publication_date,
                recency_boost,
                topk,
            ),
        )

    @mcp.tool()
    def read(doc_id: str | None = None, title: str | None = None, max_chars: int | None = None) -> dict[str, Any]:
        """按 doc_id 或 title 读取完整 clean Markdown 文档（max_chars 控制截断）。"""
        return helper_call_api("read", read_api, read_payload(doc_id, title, max_chars))

    @mcp.tool()
    def retrieve(
        query: str,
        source_institution: str | None = None,
        clinical_department: str | None = None,
        time_range: str | None = None,
        publication_date: str | None = None,
        topk: int = RETRIEVE_DEFAULT_TOPK,
    ) -> list[dict[str, Any]]:
        """分块级检索：从 PG 全文 + 向量索引召回 traceable chunks（含 consensus fallback）。"""
        return helper_call_api(
            "retrieve",
            retrieve_api,
            retrieve_payload(
                query,
                source_institution,
                clinical_department,
                time_range,
                publication_date,
                topk,
            ),
        )


def main() -> None:
    """CLI 入口：跑 run_server() 启动 MCP 服务（stdio / sse / streamable-http 三选一）。"""
    run_server(mcp)


if __name__ == "__main__":
    main()