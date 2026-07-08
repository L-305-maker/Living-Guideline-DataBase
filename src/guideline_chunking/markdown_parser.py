"""Markdown block parser with heading path, page, and char span retention."""

from __future__ import annotations

import re
from dataclasses import asdict

from src.guideline_chunking.block_classifier import classify_block_type
from src.guideline_chunking.models import DocumentMeta, ParsedBlock


HEADING_RE = re.compile(r"^(#{1,5})\s+(.+?)\s*$")
PAGE_RE = re.compile(r"(?:<!--\s*page:\s*(\d+)\s*-->|^\[page\s+(\d+)\]$|^page\s+(\d+)$)", re.I)


def parse_markdown_document(md_text: str, meta: DocumentMeta) -> list[ParsedBlock]:
    blocks: list[ParsedBlock] = []
    heading_stack: list[str] = []
    current_page: int | None = None
    order_index = 0
    pending: list[tuple[str, int, int]] = []
    pending_is_table = False
    lines = md_text.splitlines(keepends=True)
    offset = 0

    def flush_pending() -> None:
        nonlocal order_index, pending, pending_is_table
        if not pending:
            return
        raw_text = "".join(line for line, _, _ in pending).strip("\n")
        text = raw_text.strip()
        if text:
            page_start, page_end = _page_span(text, current_page)
            block_type = "table" if pending_is_table else classify_block_type(text, heading_stack)
            blocks.append(
                ParsedBlock(
                    block_id=f"{meta.doc_id}_block_{order_index:06d}",
                    doc_id=meta.doc_id,
                    block_type=block_type,
                    heading_path=list(heading_stack),
                    heading_level=len(heading_stack) or None,
                    text=text,
                    page_start=page_start,
                    page_end=page_end,
                    char_start=pending[0][1],
                    char_end=pending[-1][2],
                    order_index=order_index,
                    metadata=_meta_payload(meta),
                )
            )
            order_index += 1
        pending = []
        pending_is_table = False

    for raw_line in lines:
        line_start = offset
        offset += len(raw_line)
        line_end = offset
        line = raw_line.rstrip("\r\n")
        stripped = line.strip()

        page_match = PAGE_RE.search(stripped)
        if page_match:
            flush_pending()
            current_page = int(next(group for group in page_match.groups() if group))
            continue

        heading_match = HEADING_RE.match(line)
        if heading_match:
            flush_pending()
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            heading_stack = heading_stack[: level - 1] + [title]
            block_type = classify_block_type(title, heading_stack)
            blocks.append(
                ParsedBlock(
                    block_id=f"{meta.doc_id}_block_{order_index:06d}",
                    doc_id=meta.doc_id,
                    block_type=block_type,
                    heading_path=list(heading_stack),
                    heading_level=level,
                    text=title,
                    page_start=current_page,
                    page_end=current_page,
                    char_start=line_start,
                    char_end=line_end,
                    order_index=order_index,
                    metadata={**_meta_payload(meta), "is_heading": True},
                )
            )
            order_index += 1
            continue

        if not stripped:
            flush_pending()
            continue

        is_table_line = stripped.startswith("|") and stripped.endswith("|")
        if pending and pending_is_table != is_table_line:
            flush_pending()
        pending.append((raw_line, line_start, line_end))
        pending_is_table = is_table_line

    flush_pending()
    return blocks


def _page_span(text: str, fallback: int | None) -> tuple[int | None, int | None]:
    pages = [int(group) for match in PAGE_RE.finditer(text or "") for group in match.groups() if group]
    if pages:
        return min(pages), max(pages)
    return fallback, fallback


def _meta_payload(meta: DocumentMeta) -> dict[str, str | None]:
    return asdict(meta)
