"""PostgreSQL-backed JSON APIs for Search, Read, and Retrieve."""

from __future__ import annotations

from typing import Any

from src.models.schemas import ReadInput, RetrieveInput, SearchInput
from src.storage.pg_hybrid_retrieval import retrieve_chunks_with_consensus_fallback_pg, search_documents_with_consensus_fallback_pg
from src.storage.postgres_store import read_document_pg


def search_pg(payload: dict[str, Any] | SearchInput) -> list[dict[str, Any]]:
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
    request = payload if isinstance(payload, ReadInput) else ReadInput(**payload)
    return read_document_pg(request.doc_id, title=request.title, max_chars=request.max_chars)


def retrieve_pg(payload: dict[str, Any] | RetrieveInput) -> list[dict[str, Any]]:
    request = payload if isinstance(payload, RetrieveInput) else RetrieveInput(**payload)
    return retrieve_chunks_with_consensus_fallback_pg(
        request.query,
        source_institution=request.source_institution,
        clinical_department=request.clinical_department,
        time_range=request.time_range,
        publication_date=request.publication_date,
        topk=request.topk,
    )
