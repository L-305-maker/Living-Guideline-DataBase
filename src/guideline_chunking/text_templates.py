"""Embedding text templates that preserve original chunk text separately."""

from __future__ import annotations

from src.guideline_chunking.models import DocumentMeta


def build_text_for_embedding(
    *,
    meta: DocumentMeta,
    heading_path: list[str],
    chunk_type: str,
    text: str,
    page_start: int | None = None,
    page_end: int | None = None,
    table_title: str | None = None,
    columns_and_values: list[tuple[str, str]] | None = None,
    original_row: str | None = None,
) -> str:
    parts: list[str] = []
    add_field(parts, "Document", meta.title)
    add_field(parts, "Publisher", meta.publisher)
    add_field(parts, "Document type", meta.document_type)
    add_field(parts, "Section", " > ".join(heading_path))
    add_field(parts, "Chunk type", chunk_type)
    if page_start is not None:
        location = f"page {page_start}" if page_end in (None, page_start) else f"page {page_start}-{page_end}"
        add_field(parts, "Source location", location)
    add_field(parts, "Table title", table_title)
    if columns_and_values:
        add_field(parts, "Columns and values", "\n".join(f"{key}: {value}" for key, value in columns_and_values))
    add_field(parts, "Original row", original_row)
    add_field(parts, "Text", text)
    return "\n\n".join(parts)


def add_field(parts: list[str], label: str, value: object | None) -> None:
    if value in (None, "", [], {}):
        return
    parts.append(f"[{label}]\n{value}")
