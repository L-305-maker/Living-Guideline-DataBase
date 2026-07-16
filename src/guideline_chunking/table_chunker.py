"""Table-aware chunk construction."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict

from src.guideline_chunking.models import ChunkBuildWarning, DocumentMeta, ParsedBlock, SectionChunk, TableChunk
from src.guideline_chunking.section_chunker import section_for_block
from src.guideline_chunking.text_templates import build_text_for_embedding


def build_table_chunks(
    blocks: list[ParsedBlock],
    section_chunks: list[SectionChunk],
    meta: DocumentMeta,
    warnings: list[ChunkBuildWarning] | None = None,
) -> list[TableChunk]:
    chunks: list[TableChunk] = []
    table_index = 0
    table_blocks = [block for block in blocks if block.block_type == "table"]
    for block in table_blocks:
        table_index += 1
        table_id = f"{meta.doc_id}_table_{table_index:03d}"
        parent_section_id = section_for_block(section_chunks, block.block_id)
        try:
            # 表格解析失败只跳过当前表，并通过 warnings 保留可审计信息。
            headers, rows = parse_markdown_table(block.text)
        except ValueError as exc:
            if warnings is not None:
                warnings.append(
                    ChunkBuildWarning(
                        warning_type="table_parse_failed",
                        doc_id=block.doc_id,
                        block_id=block.block_id,
                        chunk_id=None,
                        message=str(exc),
                        severity="medium",
                    )
                )
            continue
        if not headers and warnings is not None:
            warnings.append(
                ChunkBuildWarning(
                    warning_type="table_header_missing",
                    doc_id=block.doc_id,
                    block_id=block.block_id,
                    chunk_id=None,
                    message="Markdown table has no parseable header.",
                    severity="medium",
                )
            )
        table_title = block.heading_path[-1] if block.heading_path else None
        parent_id = f"{table_id}_parent"
        # parent chunk 保存完整表格，row chunk 用于精确召回，二者通过 table_id 关联。
        chunks.append(
            TableChunk(
                chunk_id=parent_id,
                doc_id=meta.doc_id,
                parent_section_id=parent_section_id,
                table_id=table_id,
                chunk_type="table_parent",
                heading_path=list(block.heading_path),
                table_title=table_title,
                text=block.text,
                text_for_embedding=build_text_for_embedding(
                    meta=meta,
                    heading_path=block.heading_path,
                    chunk_type="table_parent",
                    text=block.text,
                    page_start=block.page_start,
                    page_end=block.page_end,
                    table_title=table_title,
                ),
                row_index=None,
                column_headers=headers,
                page_start=block.page_start,
                page_end=block.page_end,
                source_block_ids=[block.block_id],
                metadata={**asdict(meta), "table_index": table_index},
            )
        )
        for row_index, row in enumerate(rows, start=1):
            # 行列数不一致时按表头补空值，保持 embedding 模板的列语义稳定。
            values = [(headers[index], row[index] if index < len(row) else "") for index in range(len(headers))]
            row_text = "| " + " | ".join(row) + " |"
            chunks.append(
                TableChunk(
                    chunk_id=f"{table_id}_row_{row_index:03d}_{hash_text(row_text)}",
                    doc_id=meta.doc_id,
                    parent_section_id=parent_section_id,
                    table_id=table_id,
                    chunk_type="table_row",
                    heading_path=list(block.heading_path),
                    table_title=table_title,
                    text=row_text,
                    text_for_embedding=build_text_for_embedding(
                        meta=meta,
                        heading_path=block.heading_path,
                        chunk_type="table_row",
                        text=row_text,
                        page_start=block.page_start,
                        page_end=block.page_end,
                        table_title=table_title,
                        columns_and_values=values,
                        original_row=row_text,
                    ),
                    row_index=row_index,
                    column_headers=headers,
                    page_start=block.page_start,
                    page_end=block.page_end,
                    source_block_ids=[block.block_id],
                    metadata={**asdict(meta), "columns_and_values": dict(values)},
                )
            )
        # 紧随表格的说明文字单独建 note chunk，既保留语义又不污染行数据。
        note = helper_following_note(block, blocks)
        if note:
            chunks.append(
                TableChunk(
                    chunk_id=f"{table_id}_note_{hash_text(note.text)}",
                    doc_id=meta.doc_id,
                    parent_section_id=parent_section_id,
                    table_id=table_id,
                    chunk_type="table_note",
                    heading_path=list(note.heading_path),
                    table_title=table_title,
                    text=note.text,
                    text_for_embedding=build_text_for_embedding(
                        meta=meta,
                        heading_path=note.heading_path,
                        chunk_type="table_note",
                        text=note.text,
                        page_start=note.page_start,
                        page_end=note.page_end,
                        table_title=table_title,
                    ),
                    row_index=None,
                    column_headers=headers,
                    page_start=note.page_start,
                    page_end=note.page_end,
                    source_block_ids=[note.block_id],
                    metadata={**asdict(meta), "table_index": table_index},
                )
            )
    return chunks


def parse_markdown_table(text: str) -> tuple[list[str], list[list[str]]]:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("Failed to parse markdown table header.")
    headers = helper_cells(lines[0])
    separators = helper_cells(lines[1])
    if not headers or not separators or not all(helper_is_separator(cell) for cell in separators):
        raise ValueError("Failed to parse markdown table header.")
    rows = [helper_cells(line) for line in lines[2:] if line.startswith("|") and line.endswith("|")]
    return headers, rows


def helper_cells(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")]


def helper_is_separator(cell: str) -> bool:
    stripped = cell.strip()
    return len(stripped.replace(":", "")) >= 3 and set(stripped.replace(":", "")) == {"-"}


def hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def helper_following_note(table_block: ParsedBlock, blocks: list[ParsedBlock]) -> ParsedBlock | None:
    next_order = table_block.order_index + 1
    for block in blocks:
        if block.order_index != next_order:
            continue
        if block.heading_path == table_block.heading_path and re.match(r"^\s*notes?\s*[:：]", block.text, re.I):
            return block
    return None
