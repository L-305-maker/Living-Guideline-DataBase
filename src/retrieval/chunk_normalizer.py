"""Normalize chunk JSONL records for storage and retrieval indexes."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator

from src.utils.clinical_department import classify_chunk_departments
from src.utils.io import DATA_DIR, read_jsonl


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?|[\u4e00-\u9fff]")


def load_document_metadata(data_dir: str | Path = DATA_DIR) -> dict[str, dict[str, Any]]:
    return {record["doc_id"]: record for record in read_jsonl(Path(data_dir) / "documents.jsonl")}


def iter_normalized_chunks(
    data_dir: str | Path = DATA_DIR,
    chunks_path: str | Path | None = None,
) -> Iterator[dict[str, Any]]:
    data_path = Path(data_dir)
    docs = load_document_metadata(data_path)
    path = Path(chunks_path) if chunks_path else data_path / "chunks" / "all_chunks.jsonl"
    per_doc_index: dict[str, int] = {}
    for record in read_jsonl(path):
        doc_id = helper_doc_id(record)
        chunk_index = record.get("chunk_index")
        if chunk_index is None:
            chunk_index = per_doc_index.get(doc_id, 0)
        per_doc_index[doc_id] = int(chunk_index) + 1
        yield normalize_chunk_record(record, docs, int(chunk_index))


def normalize_chunk_record(
    record: dict[str, Any],
    documents: dict[str, dict[str, Any]] | None = None,
    chunk_index: int = 0,
) -> dict[str, Any]:
    doc_id = helper_doc_id(record)
    doc = (documents or {}).get(doc_id, {})
    section_path = as_list(record.get("section_path") or record.get("heading_path") or [])
    chunk_type = str(record.get("chunk_type") or "other")
    content = str(record.get("content") or record.get("text") or record.get("recommendation") or "")
    title = str(record.get("title") or doc.get("title") or (section_path[0] if section_path else ""))
    text_for_embedding = str(record.get("text_for_embedding") or helper_embedding_text(section_path, chunk_type, content))
    retrieval = str(record.get("retrieval_text") or text_for_embedding)
    parent_departments = doc.get("clinical_departments") or [doc.get("clinical_department") or "未分类"]
    department_result = classify_chunk_departments(section_path, content, parent_departments, chunk_type)

    normalized = dict(record)
    normalized.update(
        {
            "doc_id": doc_id,
            "title": title,
            "publication_date": record.get("publication_date") or doc.get("publication_date") or "unknown",
            "source_institution": record.get("source_institution") or doc.get("source_institution") or "Unknown",
            "clinical_department": department_result["clinical_department"],
            "clinical_departments": department_result["clinical_departments"],
            "department_scope": department_result["department_scope"],
            "document_kind": record.get("document_kind") or doc.get("document_kind") or "guideline",
            "section_path": section_path,
            "chunk_index": int(record.get("chunk_index") if record.get("chunk_index") is not None else chunk_index),
            "content": content,
            "retrieval_text": retrieval,
            "chunk_type": chunk_type,
            "token_count": int(record.get("token_count") or len(TOKEN_RE.findall(content))),
            "retrieval_key": record.get("retrieval_key") or f"{doc_id}#{chunk_index}",
            "source_file": record.get("source_file") or doc.get("source_file") or "",
            "markdown_clean_path": record.get("markdown_clean_path") or doc.get("markdown_clean_path") or "",
            "is_background": bool(record.get("is_background") or chunk_type == "background"),
            "is_reference_section": bool(record.get("is_reference_section")),
            "text_for_embedding": text_for_embedding,
            "recommendation": record.get("recommendation") or "",
            "evidence": as_list(record.get("evidence") or []),
            "metadata": record.get("metadata") or {},
        }
    )
    return normalized


def helper_doc_id(record: dict[str, Any]) -> str:
    doc_id = record.get("doc_id") or record.get("source_doc_id")
    if not doc_id:
        raise KeyError("chunk record missing doc_id/source_doc_id")
    return str(doc_id)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def helper_embedding_text(section_path: list[Any], chunk_type: str, content: str) -> str:
    return "\n".join(
        [
            "[GUIDELINE SECTION]: " + " > ".join(str(item) for item in section_path),
            "[CHUNK TYPE]: " + chunk_type,
            "[CONTENT]:",
            content,
        ]
    )
