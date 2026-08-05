# Markdown block 解析器：保留 heading_path / page / char span 等溯源字段。
#
# 关键设计：
# - 逐行扫描，状态机：维持 heading_stack、current_page、pending 三组状态；
# - 页码标记（<!-- page: N -->）单独 flush pending 后只更新 current_page；
# - 标题层级维护：截断到当前 level-1 父路径再追加，避免 heading_stack 越积越长；
# - 表格与普通段落必须分块：pending_is_table 标志在跨类型时强制 flush。
"""Markdown block parser with heading path, page, and char span retention."""

from __future__ import annotations

import re
from dataclasses import asdict

from src.guideline_chunking.block_classifier import classify_block_type
from src.guideline_chunking.models import DocumentMeta, ParsedBlock


# 标题识别：1-5 个 # + 标题文本（不支持 6 级，与上游约定一致）
HEADING_RE = re.compile(r"^(#{1,5})\s+(.+?)\s*$")
# 页码标记：HTML 注释、方括号包裹、裸 page 三种写法兼容
PAGE_RE = re.compile(r"(?:<!--\s*page:\s*(\d+)\s*-->|^\[page\s+(\d+)\]$|^page\s+(\d+)$)", re.I)


def parse_markdown_document(md_text: str, meta: DocumentMeta) -> list[ParsedBlock]:
    """把 Markdown 文本解析为 ParsedBlock 列表，每个 block 携带完整溯源字段。

    流程：
    1. 维护 heading_stack（当前 heading 层级）和 current_page（最近的页码）；
    2. 遇到标题：flush 旧 pending，按 level 截断 heading_stack 父路径，追加新标题作为 block；
    3. 遇到页码标记：flush 旧 pending，更新 current_page；
    4. 遇到空行：flush pending；
    5. 其它行：append 到 pending；表格与普通段落在跨类型时强制 flush；
    6. 文档结束 flush 最后一段 pending。
    """
    blocks: list[ParsedBlock] = []
    heading_stack: list[str] = []
    current_page: int | None = None
    order_index = 0
    pending: list[tuple[str, int, int]] = []
    pending_is_table = False
    lines = md_text.splitlines(keepends=True)
    offset = 0

    # 普通文本按空行、标题、页标记和表格边界累计，flush 时一次生成可溯源 block。
    def flush_pending() -> None:
        nonlocal order_index, pending, pending_is_table
        if not pending:
            return
        raw_text = "".join(line for line, _, _ in pending).strip("\n")
        text = raw_text.strip()
        if text:
            page_start, page_end = helper_page_span(text, current_page)
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
                    metadata=helper_meta_payload(meta),
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

        # 页码标记本身不进入正文，只更新后续 block 的页范围。
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
            # 截断到当前层级的父路径，再追加新标题，保持 heading_path 层级正确。
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
                    metadata={**helper_meta_payload(meta), "is_heading": True},
                )
            )
            order_index += 1
            continue

        if not stripped:
            flush_pending()
            continue

        is_table_line = stripped.startswith("|") and stripped.endswith("|")
        # 表格与普通段落必须分块，否则后续表格解析会收到混合文本。
        if pending and pending_is_table != is_table_line:
            flush_pending()
        pending.append((raw_line, line_start, line_end))
        pending_is_table = is_table_line

    flush_pending()
    return blocks


def helper_page_span(text: str, fallback: int | None) -> tuple[int | None, int | None]:
    """从 text 中提取页码标记的最小 / 最大值；无标记时回退 fallback。"""
    pages = [int(group) for match in PAGE_RE.finditer(text or "") for group in match.groups() if group]
    if pages:
        return min(pages), max(pages)
    return fallback, fallback


def helper_meta_payload(meta: DocumentMeta) -> dict[str, str | None]:
    """把 DocumentMeta 转为 dict 作为 ParsedBlock.metadata，便于序列化。"""
    return asdict(meta)