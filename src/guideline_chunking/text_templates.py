# Embedding 文本模板：把 chunk 元信息拼成 LLM 友好的结构化文本。
#
# 关键设计：
# - 与 retrieval_text 的核心差异：本模板用 [Field]/value 形式，结构化更强；
# - 保留 original_row / columns_and_values 等表格专有字段；
# - build_text_for_embedding 是构造入口，add_field 是底层累加器（None/空值跳过）。
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
    """构造 embedding 输入文本：按 [Field]\\nvalue 顺序拼接，空值自动跳过。"""
    parts: list[str] = []
    add_field(parts, "Document", meta.title)
    add_field(parts, "Publisher", meta.publisher)
    add_field(parts, "Document type", meta.document_type)
    add_field(parts, "Section", " > ".join(heading_path))
    add_field(parts, "Chunk type", chunk_type)
    if page_start is not None:
        # page_end 缺失或与 start 相同 → 单页；否则输出 page_start-end 范围
        location = f"page {page_start}" if page_end in (None, page_start) else f"page {page_start}-{page_end}"
        add_field(parts, "Source location", location)
    add_field(parts, "Table title", table_title)
    if columns_and_values:
        add_field(parts, "Columns and values", "\n".join(f"{key}: {value}" for key, value in columns_and_values))
    add_field(parts, "Original row", original_row)
    add_field(parts, "Text", text)
    return "\n\n".join(parts)


def add_field(parts: list[str], label: str, value: object | None) -> None:
    """累加一个 [Field]/value 段；空值（None/"" / [] / {}）跳过，避免生成空 [Field] 标签。"""
    if value in (None, "", [], {}):
        return
    parts.append(f"[{label}]\n{value}")