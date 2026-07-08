"""PostgreSQL-only MCP server entry point."""

from __future__ import annotations

import os
from typing import Any

from src.mcp.api_pg import read_pg as read_api
from src.mcp.api_pg import retrieve_pg as retrieve_api
from src.mcp.api_pg import search_pg as search_api
from src.mcp.call_logger import log_mcp_call

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover
    FastMCP = None  # type: ignore[assignment]


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


mcp = (
    FastMCP(
        "pdf-markdown-rag-postgres",
        host=os.environ.get("MCP_HOST", "127.0.0.1"),
        port=_env_int("MCP_PORT", 8000),
        streamable_http_path=os.environ.get("MCP_STREAMABLE_HTTP_PATH", "/mcp"),
        sse_path=os.environ.get("MCP_SSE_PATH", "/sse"),
        message_path=os.environ.get("MCP_MESSAGE_PATH", "/messages/"),
        json_response=_env_bool("MCP_JSON_RESPONSE", False),
        stateless_http=_env_bool("MCP_STATELESS_HTTP", False),
    )
    if FastMCP
    else None
)


def _call_api(tool_name: str, api: Any, payload: dict[str, Any]) -> Any:
    return log_mcp_call(tool_name, "postgres", payload, lambda: api(payload))


if mcp is not None:

    @mcp.tool()
    def search(
        query: str,
        source_institution: str | None = None,
        clinical_department: str | None = None,
        time_range: str | None = None,
        publication_date: str | None = None,
        recency_boost: bool = False,
        topk: int = 10,
    ) -> list[dict[str, Any]]:
        """Search candidate guideline documents from PostgreSQL."""

        return _call_api(
            "search",
            search_api,
            {
                "query": query,
                "source_institution": source_institution,
                "clinical_department": clinical_department,
                "time_range": time_range,
                "publication_date": publication_date,
                "recency_boost": recency_boost,
                "topk": topk,
            },
        )

    @mcp.tool()
    def read(doc_id: str | None = None, title: str | None = None, max_chars: int | None = None) -> dict[str, Any]:
        """Read a full clean Markdown document from PostgreSQL by doc_id or title."""

        return _call_api("read", read_api, {"doc_id": doc_id, "title": title, "max_chars": max_chars})

    @mcp.tool()
    def retrieve(
        query: str,
        source_institution: str | None = None,
        clinical_department: str | None = None,
        time_range: str | None = None,
        publication_date: str | None = None,
        topk: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve traceable RAG chunks from PostgreSQL."""

        return _call_api(
            "retrieve",
            retrieve_api,
            {
                "query": query,
                "source_institution": source_institution,
                "clinical_department": clinical_department,
                "time_range": time_range,
                "publication_date": publication_date,
                "topk": topk,
            },
        )


def main() -> None:
    if mcp is None:
        raise RuntimeError("Install mcp>=1.9 to run the MCP server: python -m pip install mcp>=1.9")
    transport = os.environ.get("MCP_TRANSPORT", "stdio").strip().lower()
    if transport not in {"stdio", "sse", "streamable-http"}:
        raise ValueError("MCP_TRANSPORT must be one of: stdio, sse, streamable-http")
    mcp.run(transport=transport, mount_path=os.environ.get("MCP_MOUNT_PATH"))


if __name__ == "__main__":
    main()
