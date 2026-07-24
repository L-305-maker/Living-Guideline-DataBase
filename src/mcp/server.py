"""MCP server exposing Search, Read, and Retrieve as first-class tools."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.mcp.server_common import (
    env_bool as helper_env_bool,
    env_int as helper_env_int,
    read_payload,
    retrieve_payload,
    run_server,
    search_payload,
)
from src.mcp.call_logger import log_mcp_call
from src.utils.io import DATA_DIR as DEFAULT_DATA_DIR

BACKEND = os.environ.get("RAG_BACKEND", "local").strip().lower()
IS_POSTGRES_BACKEND = BACKEND in {"postgres", "postgresql", "pg"}
SEARCH_DEFAULT_TOPK = 10 if IS_POSTGRES_BACKEND else 20
RETRIEVE_DEFAULT_TOPK = 5 if IS_POSTGRES_BACKEND else 30
DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", str(DEFAULT_DATA_DIR)))

if IS_POSTGRES_BACKEND:
    from src.mcp.api_pg import read_pg as read_api
    from src.mcp.api_pg import retrieve_pg as retrieve_api
    from src.mcp.api_pg import search_pg as search_api
else:
    from src.mcp.api_read import read as read_api
    from src.mcp.api_retrieve import retrieve as retrieve_api
    from src.mcp.api_search import search as search_api

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover
    FastMCP = None  # type: ignore[assignment]




mcp = (
    FastMCP(
        f"pdf-markdown-rag-{BACKEND}",
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
    def call() -> Any:
        if IS_POSTGRES_BACKEND:
            return api(payload)
        return api(payload, data_dir=DATA_DIR)

    return log_mcp_call(tool_name, BACKEND, payload, call)


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
        """Search candidate guideline documents by keywords, source institution, and time range."""

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
        """Read a full clean Markdown document by doc_id or title."""

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
        """Retrieve traceable chunks with BM25+Dense recall followed by one reranker pass."""

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
    run_server(mcp)


if __name__ == "__main__":
    main()
