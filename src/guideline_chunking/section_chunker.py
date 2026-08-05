# Section 级 chunk 构造（heading_path 维度的父章节）。
#
# 关键设计：
# - MAX_SECTION_TOKENS = 3000：单 section 上限，超长按 token 预算分片；
# - section_id 含 heading_path + order_index + part_index，同名章节也唯一；
# - 只在块边界拆分，避免截断表格、列表或推荐语句的内部结构。
"""Parent section chunk construction."""

from __future__ import annotations

import hashlib

from src.guideline_chunking.models import DocumentMeta, ParsedBlock, SectionChunk
from src.utils.text import estimate_tokens


# 单 section token 上限。超长按此阈值分片（按 token 累计在块边界切）。
MAX_SECTION_TOKENS = 3000


def build_section_chunks(blocks: list[ParsedBlock], meta: DocumentMeta) -> list[SectionChunk]:
    """把 ParsedBlock 列表归并成 SectionChunk 列表（按 heading_path 分组 + token 预算分片）。

    行为：
    1. 按 order_index 排序确保按文档顺序处理；
    2. heading_path 变化时触发 flush；
    3. 同一 heading_path 下按 token 累计分片，超 MAX_SECTION_TOKENS 在块边界 flush。
    """
    sections: list[SectionChunk] = []
    current_key: tuple[str, ...] | None = None
    current_blocks: list[ParsedBlock] = []

    def flush() -> None:
        nonlocal current_blocks
        if not current_blocks:
            return
        sections.extend(helper_section_parts(current_blocks, meta))
        current_blocks = []

    # 先按原文顺序恢复块序列；heading_path 变化才代表进入新的父章节。
    for block in sorted(blocks, key=lambda item: item.order_index):
        key = tuple(block.heading_path or [helper_untitled(meta)])
        if current_key is not None and key != current_key:
            flush()
        current_key = key
        current_blocks.append(block)
    flush()
    return sections


def section_for_block(section_chunks: list[SectionChunk], block_id: str) -> str | None:
    """按 block_id 反查所属 section_id；找不到返回 None。"""
    for section in section_chunks:
        if block_id in section.child_block_ids:
            return section.section_id
    return None


def helper_section_parts(blocks: list[ParsedBlock], meta: DocumentMeta) -> list[SectionChunk]:
    """对一组同 heading_path 块按 token 预算分片，每片构造一个 SectionChunk。

    关键设计：
    - 只在块边界拆分，避免截断表格 / 列表 / 推荐语句；
    - title 来自 path 最后一项；metadata 含 title / publisher / part_index；
    - 分片 page_start = 最小页，分片 page_end = 最大页。
    """
    parts: list[list[ParsedBlock]] = []
    current: list[ParsedBlock] = []
    current_tokens = 0
    for block in blocks:
        block_tokens = estimate_tokens(block.text)
        # 只在块边界拆分，避免截断表格、列表或推荐语句的内部结构。
        if current and current_tokens + block_tokens > MAX_SECTION_TOKENS:
            parts.append(current)
            current = []
            current_tokens = 0
        current.append(block)
        current_tokens += block_tokens
    if current:
        parts.append(current)

    section_chunks: list[SectionChunk] = []
    for part_index, part in enumerate(parts):
        path = part[0].heading_path or [helper_untitled(meta)]
        title = path[-1]
        # ID 同时包含标题路径、起始顺序和分片号，使同名章节及超长章节分片保持唯一。
        section_id = helper_section_id(meta.doc_id, path, part[0].order_index, part_index)
        pages = [page for block in part for page in (block.page_start, block.page_end) if page is not None]
        section_chunks.append(
            SectionChunk(
                section_id=section_id,
                doc_id=meta.doc_id,
                heading_path=list(path),
                heading_level=part[0].heading_level or len(path) or 1,
                title=title,
                text="\n\n".join(block.text for block in part if not block.metadata.get("is_heading")),
                child_block_ids=[block.block_id for block in part],
                child_chunk_ids=[],
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
                order_start=part[0].order_index,
                order_end=part[-1].order_index,
                metadata={"title": meta.title, "publisher": meta.publisher, "part_index": part_index},
            )
        )
    return section_chunks


def helper_section_id(doc_id: str, path: list[str], order_index: int, part_index: int) -> str:
    """section_id = SHA1(path:order:part)[:10]，保证同名章节 + 超长分片都唯一。"""
    digest = hashlib.sha1((" > ".join(path) + f":{order_index}:{part_index}").encode("utf-8")).hexdigest()[:10]
    return f"{doc_id}_sec_{digest}"


def helper_untitled(meta: DocumentMeta) -> str:
    """缺失标题时用文档 title / 'Untitled' 兜底。"""
    return meta.title or "Untitled"