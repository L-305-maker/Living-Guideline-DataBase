"""Chunk-level RAG retrieval API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.pipeline.cleaning.semantic_chunker import estimate_tokens, retrieval_text
from src.retrieval.bm25_store import load_store
from src.retrieval.document_repr.section_classifier import classify_chunk
from src.retrieval.hybrid import retrieve_chunks_with_consensus_fallback
from src.retrieval.rrf import rrf_fusion
from src.retrieval.sqlite_store import DEFAULT_DB_PATH
from src.retrieval.vector_store import vector_search
from src.models.schemas import RetrieveInput
from src.utils.io import DATA_DIR


def helper_validate(payload: dict[str, Any] | RetrieveInput) -> RetrieveInput:
    return payload if isinstance(payload, RetrieveInput) else RetrieveInput(**payload)


def helper_matches(record: dict[str, Any], request: RetrieveInput) -> bool:
    if request.source_institution and request.source_institution.lower() not in (record.get("source_institution") or "").lower():
        return False
    if request.clinical_department and request.clinical_department.lower() not in (record.get("clinical_department") or "").lower():
        return False
    return True


def retrieve(payload: dict[str, Any] | RetrieveInput, data_dir: str | Path = DATA_DIR) -> list[dict[str, Any]]:
    request = helper_validate(payload)
    index_dir = Path(data_dir) / "index"
    sqlite_path = index_dir / DEFAULT_DB_PATH.name
    if sqlite_path.exists():
        return retrieve_chunks_with_consensus_fallback(
            request.query,
            sqlite_path,
            index_dir=index_dir,
            source_institution=request.source_institution,
            clinical_department=request.clinical_department,
            time_range=request.time_range,
            publication_date=request.publication_date,
            topk=request.topk,
            exclude_reference_sections=True,
        )
    store = load_store(index_dir / "bm25_chunks.json")
    bm25 = store.search(
        request.query,
        top_n=50,
        source_institution=request.source_institution,
        clinical_department=request.clinical_department,
        time_range=request.time_range,
        exclude_reference_sections=True,
    )
    bm25_ids = [item_id for item_id, _score in bm25]
    vector_ids_raw = vector_search(
        request.query,
        index_dir / "faiss_chunks.index",
        index_dir / "faiss_chunks_mapping.jsonl",
        "chunk_id",
        top_n=50,
    )
    records = {record["chunk_id"]: record for record in store.records}
    vector_ids = [chunk_id for chunk_id in vector_ids_raw if chunk_id in records and helper_matches(records[chunk_id], request)]
    fused = rrf_fusion([bm25_ids, vector_ids])
    results: list[dict[str, Any]] = []
    for chunk_id, score in fused:
        record = records.get(chunk_id)
        if not record:
            continue
        chunk_type = str(record.get("chunk_type") or "") or classify_chunk(record)
        section_path = record.get("section_path", [])
        content = record.get("content", "")
        results.append(
            {
                "chunk_id": record["chunk_id"],
                "doc_id": record["doc_id"],
                "content": content,
                "title": record.get("title", ""),
                "publication_date": record.get("publication_date", "unknown"),
                "source_institution": record.get("source_institution", "Unknown"),
                "clinical_department": record.get("clinical_department", "未分类"),
                "section_path": section_path,
                "chunk_type": chunk_type,
                "token_count": int(record.get("token_count") or estimate_tokens(content)),
                "retrieval_text": record.get("retrieval_text")
                or retrieval_text(record.get("title", ""), section_path, chunk_type, content),
                "is_background": bool(record.get("is_background") or chunk_type == "background"),
                "score": score,
                "retrieval_key": record.get("retrieval_key", ""),
            }
        )
        if len(results) >= request.topk:
            break
    return results
