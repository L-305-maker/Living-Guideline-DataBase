"""Chunk relationship construction."""

from __future__ import annotations

import hashlib

from src.guideline_chunking.models import AtomicChunk, ChunkLink, SectionChunk, TableChunk


def build_chunk_links(
    section_chunks: list[SectionChunk],
    atomic_chunks: list[AtomicChunk],
    table_chunks: list[TableChunk],
) -> list[ChunkLink]:
    links: list[ChunkLink] = []
    all_child_chunks = [*atomic_chunks, *table_chunks]
    for chunk in all_child_chunks:
        parent = getattr(chunk, "parent_section_id", None)
        if parent:
            links.append(make_link(chunk.doc_id, parent, chunk.chunk_id, "parent_section", {}))

    for chunk in atomic_chunks:
        if chunk.prev_chunk_id:
            links.append(make_link(chunk.doc_id, chunk.chunk_id, chunk.prev_chunk_id, "previous_chunk", {}))
        if chunk.next_chunk_id:
            links.append(make_link(chunk.doc_id, chunk.chunk_id, chunk.next_chunk_id, "next_chunk", {}))

    # 表格行同时建立父到子和子到父链接，支持整表扩展与行级回溯。
    table_parents = {chunk.table_id: chunk for chunk in table_chunks if chunk.chunk_type == "table_parent"}
    for chunk in table_chunks:
        if chunk.chunk_type != "table_row":
            continue
        parent = table_parents.get(chunk.table_id)
        if parent:
            links.append(make_link(chunk.doc_id, parent.chunk_id, chunk.chunk_id, "table_parent_to_row", {}))
            links.append(make_link(chunk.doc_id, chunk.chunk_id, parent.chunk_id, "row_to_table_parent", {}))

    by_section: dict[str, list[str]] = {}
    for chunk in all_child_chunks:
        parent = getattr(chunk, "parent_section_id", None)
        if parent:
            by_section.setdefault(parent, []).append(chunk.chunk_id)
    for section_id, chunk_ids in by_section.items():
        for index, source in enumerate(chunk_ids):
            # 同章节只连接后续五个邻居，避免链接数量随章节长度平方增长。
            for target in chunk_ids[index + 1 : min(index + 6, len(chunk_ids))]:
                links.append(make_link(helper_doc_id_for_section(section_chunks, section_id), source, target, "same_section", {}))
    return links


def make_link(doc_id: str, source: str, target: str, link_type: str, metadata: dict) -> ChunkLink:
    digest = hashlib.sha1(f"{source}:{target}:{link_type}".encode("utf-8")).hexdigest()[:10]
    return ChunkLink(
        link_id=f"{doc_id}_link_{digest}",
        doc_id=doc_id,
        source_chunk_id=source,
        target_chunk_id=target,
        link_type=link_type,
        metadata=metadata,
    )


def helper_doc_id_for_section(sections: list[SectionChunk], section_id: str) -> str:
    for section in sections:
        if section.section_id == section_id:
            return section.doc_id
    return "unknown"
