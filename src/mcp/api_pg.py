# PostgreSQL 后端的 search / read / retrieve JSON API。
#
# 三个函数对应 MCP 工具的入参与调用：search_pg / read_pg / retrieve_pg。
# 关键设计：payload 既接受 Pydantic 模型（已校验），也接受 dict（外部 JSON 调用），自动按需构造。
"""PostgreSQL-backed JSON APIs for Search, Read, and Retrieve."""

from __future__ import annotations

from typing import Any

from src.models.schemas import ReadInput, RetrieveInput, SearchInput
from src.storage.pg_hybrid_retrieval import (
    retrieve_chunks_with_consensus_fallback_pg,
    search_documents_with_consensus_fallback_pg,
)
from src.storage.postgres_store import read_document_pg


def search_pg(payload: dict[str, Any] | SearchInput) -> list[dict[str, Any]]:
    """MCP search 工具的 PostgreSQL 后端。

    输入兼容 dict / SearchInput 两种形式；自动按字段映射到 storage 的 hybrid retrieval。
    """
    request = payload if isinstance(payload, SearchInput) else SearchInput(**payload)
    return search_documents_with_consensus_fallback_pg(
        request.query,
        source_institution=request.source_institution,
        clinical_department=request.clinical_department,
        time_range=request.time_range,
        publication_date=request.publication_date,
        topk=request.topk,
    )


def read_pg(payload: dict[str, Any] | ReadInput) -> dict[str, Any]:
    """MCP read 工具的 PostgreSQL 后端（按 doc_id 或 title 定位文档）。"""
    request = payload if isinstance(payload, ReadInput) else ReadInput(**payload)
    return read_document_pg(request.doc_id, title=request.title, max_chars=request.max_chars)


def retrieve_pg(payload: dict[str, Any] | RetrieveInput) -> list[dict[str, Any]]:
    """MCP retrieve 工具的 PostgreSQL 后端（按块返回，含 consensus fallback）。"""
    request = payload if isinstance(payload, RetrieveInput) else RetrieveInput(**payload)
    return retrieve_chunks_with_consensus_fallback_pg(
        request.query,
        source_institution=request.source_institution,
        clinical_department=request.clinical_department,
        time_range=request.time_range,
        publication_date=request.publication_date,
        topk=request.topk,
    )