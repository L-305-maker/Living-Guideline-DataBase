"""Document-level search API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.retrieval.document_multiview_search import has_document_representations_sqlite, search_documents_multiview
from src.retrieval.sqlite_store import DEFAULT_DB_PATH
from src.models.schemas import SearchInput
from src.utils.io import DATA_DIR


def _validate(payload: dict[str, Any] | SearchInput) -> SearchInput:
    return payload if isinstance(payload, SearchInput) else SearchInput(**payload)


def search(payload: dict[str, Any] | SearchInput, data_dir: str | Path = DATA_DIR) -> list[dict[str, Any]]:
    request = _validate(payload)
    index_dir = Path(data_dir) / "index"
    sqlite_path = index_dir / DEFAULT_DB_PATH.name
    if not sqlite_path.exists() or not has_document_representations_sqlite(sqlite_path):
        raise RuntimeError("Document search requires the new document_cards/document_views SQLite index; rebuild rag.sqlite.")
    return search_documents_multiview(
        request.query,
        sqlite_path,
        index_dir=index_dir,
        source_institution=request.source_institution,
        clinical_department=request.clinical_department,
        time_range=request.time_range,
        publication_date=request.publication_date,
        recency_boost=request.recency_boost,
        topk=request.topk,
    )
