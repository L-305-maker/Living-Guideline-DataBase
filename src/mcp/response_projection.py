"""Small public response contracts for MCP search and retrieval."""

from __future__ import annotations

import re
from typing import Any


def helper_clip(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def helper_departments(item: dict[str, Any]) -> list[str]:
    labels = item.get("clinical_departments") or []
    if isinstance(labels, str):
        labels = labels.split("|")
    if not labels:
        labels = str(item.get("clinical_department") or "未分类").split("|")
    return list(dict.fromkeys(str(label).strip() for label in labels if str(label).strip())) or ["未分类"]


def compact_document_result(item: dict[str, Any]) -> dict[str, Any]:
    views = item.get("matched_views") or []
    chunks = item.get("matched_chunks") or []
    snippet = item.get("abstract") or item.get("snippet") or (views[0].get("text") if views else "")
    return {
        "doc_id": item.get("doc_id", ""),
        "title": item.get("title", ""),
        "snippet": helper_clip(snippet, 400),
        "publication_date": item.get("publication_date") or "unknown",
        "source_institution": item.get("source_institution") or "Unknown",
        "clinical_departments": helper_departments(item),
        "document_kind": item.get("document_kind") or "guideline",
        "score": float(item.get("score") or 0.0),
        "matched_views": [
            {
                "view_type": view.get("view_type", ""),
                "text": helper_clip(view.get("text"), 500),
                "score": float(view.get("score") or 0.0),
            }
            for view in views[:2]
        ],
        "matched_chunks": [
            {
                "chunk_id": chunk.get("chunk_id", ""),
                "section_path": chunk.get("section_path") or [],
                "chunk_type": chunk.get("chunk_type") or "other",
                "content": helper_clip(chunk.get("content"), 1000),
                "score": float(chunk.get("score") or 0.0),
            }
            for chunk in chunks[:2]
        ],
    }


def compact_chunk_result(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": item.get("chunk_id", ""),
        "doc_id": item.get("doc_id", ""),
        "title": item.get("title", ""),
        "publication_date": item.get("publication_date") or "unknown",
        "source_institution": item.get("source_institution") or "Unknown",
        "clinical_departments": helper_departments(item),
        "document_kind": item.get("document_kind") or "guideline",
        "section_path": item.get("section_path") or [],
        "chunk_type": item.get("chunk_type") or "other",
        "content": helper_clip(item.get("content"), 1200),
        "score": float(item.get("score") or 0.0),
    }
