"""Atomic retrieval chunk construction."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict

from src.guideline_chunking.models import AtomicChunk, ChunkBuildWarning, DocumentMeta, ParsedBlock, SectionChunk
from src.guideline_chunking.section_chunker import estimate_tokens, section_for_block
from src.guideline_chunking.text_templates import build_text_for_embedding


TARGET_MIN = 300
TARGET_MAX = 900
OVERLAP = 80
NUMBERED_RECOMMENDATION_RE = re.compile(
    r"(?m)^\s*(?=(?:\d+(?:\.\d+)+|Recommendation\s+\d+[A-Za-z]?)\b)"
)
RECOMMENDATION_LINE_RE = re.compile(
    r"^\s*(?:[-*+]\s*)?(?:we recommend|we suggest|do not offer|offer|consider|should|should not|"
    r"is recommended|is not recommended)\b",
    re.I,
)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?|[\u4e00-\u9fff]|\S")


def build_atomic_chunks(
    blocks: list[ParsedBlock],
    section_chunks: list[SectionChunk],
    meta: DocumentMeta,
    warnings: list[ChunkBuildWarning] | None = None,
) -> list[AtomicChunk]:
    chunks: list[AtomicChunk] = []
    for block in sorted(blocks, key=lambda item: item.order_index):
        if block.metadata.get("is_heading") or block.block_type == "table" or not block.text.strip():
            continue
        if block.block_type == "recommendation_candidate":
            parts = _split_recommendations(block.text)
        elif block.block_type in {"evidence_candidate", "rationale_candidate", "background", "unknown"}:
            parts = _split_semantic(block.text, TARGET_MIN, TARGET_MAX, OVERLAP)
        else:
            parts = [block.text]
        for part_index, part in enumerate(parts):
            chunk_type = block.block_type
            chunk_id = _chunk_id(meta.doc_id, block.block_id, part_index, part)
            parent_section_id = section_for_block(section_chunks, block.block_id)
            text_for_embedding = build_text_for_embedding(
                meta=meta,
                heading_path=block.heading_path,
                chunk_type=chunk_type,
                text=part,
                page_start=block.page_start,
                page_end=block.page_end,
            )
            chunk = AtomicChunk(
                chunk_id=chunk_id,
                doc_id=meta.doc_id,
                parent_section_id=parent_section_id,
                chunk_type=chunk_type,
                heading_path=list(block.heading_path),
                text=part,
                text_for_embedding=text_for_embedding,
                source_block_ids=[block.block_id],
                page_start=block.page_start,
                page_end=block.page_end,
                order_start=block.order_index,
                order_end=block.order_index,
                prev_chunk_id=None,
                next_chunk_id=None,
                metadata={**asdict(meta), "source_block_type": block.block_type, "part_index": part_index},
            )
            _warn_for_chunk(chunk, block, warnings)
            chunks.append(chunk)
            if parent_section_id:
                _attach_child_chunk(section_chunks, parent_section_id, chunk_id)
    _wire_prev_next(chunks)
    return chunks


def _split_recommendations(text: str) -> list[str]:
    starts = [match.start() for match in NUMBERED_RECOMMENDATION_RE.finditer(text)]
    if len(starts) < 2:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) > 1 and sum(1 for line in lines if RECOMMENDATION_LINE_RE.search(line)) > 1:
            return lines
        return [text.strip()]
    starts.append(len(text))
    return [text[starts[i] : starts[i + 1]].strip() for i in range(len(starts) - 1) if text[starts[i] : starts[i + 1]].strip()]


def _split_semantic(text: str, min_tokens: int, max_tokens: int, overlap: int) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text or "") if part.strip()]
    if not paragraphs:
        return []
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for paragraph in paragraphs:
        tokens = estimate_tokens(paragraph)
        if tokens > max_tokens:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_tokens = 0
            chunks.extend(_split_long_text(paragraph, max_tokens, overlap))
            continue
        if current and current_tokens + tokens > max_tokens and current_tokens >= min_tokens:
            chunks.append("\n\n".join(current))
            current = []
            current_tokens = 0
        current.append(paragraph)
        current_tokens += tokens
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _split_long_text(text: str, max_tokens: int, overlap: int) -> list[str]:
    tokens = TOKEN_RE.findall(text)
    parts: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        parts.append(_join_tokens(tokens[start:end]))
        if end >= len(tokens):
            break
        start = max(start + 1, end - overlap)
    return parts


def _join_tokens(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?，。；：！？)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    return text.strip()


def _chunk_id(doc_id: str, block_id: str, part_index: int, text: str) -> str:
    digest = hashlib.sha1(f"{block_id}:{part_index}:{text[:80]}".encode("utf-8")).hexdigest()[:10]
    return f"{doc_id}_chunk_{digest}"


def _wire_prev_next(chunks: list[AtomicChunk]) -> None:
    by_doc: dict[str, list[AtomicChunk]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk)
    for doc_chunks in by_doc.values():
        doc_chunks.sort(key=lambda item: (item.order_start, item.chunk_id))
        for index, chunk in enumerate(doc_chunks):
            chunk.prev_chunk_id = doc_chunks[index - 1].chunk_id if index else None
            chunk.next_chunk_id = doc_chunks[index + 1].chunk_id if index + 1 < len(doc_chunks) else None


def _attach_child_chunk(section_chunks: list[SectionChunk], section_id: str, chunk_id: str) -> None:
    for section in section_chunks:
        if section.section_id == section_id and chunk_id not in section.child_chunk_ids:
            section.child_chunk_ids.append(chunk_id)
            return


def _warn_for_chunk(chunk: AtomicChunk, block: ParsedBlock, warnings: list[ChunkBuildWarning] | None) -> None:
    if warnings is None:
        return
    token_count = estimate_tokens(chunk.text)
    if token_count > TARGET_MAX:
        warnings.append(_warning("chunk_too_long", block, chunk.chunk_id, f"Chunk has {token_count} tokens.", "medium"))
    if token_count < 10:
        warnings.append(_warning("chunk_too_short", block, chunk.chunk_id, "Chunk has fewer than 10 tokens.", "low"))
    if not chunk.heading_path:
        warnings.append(_warning("heading_missing", block, chunk.chunk_id, "Chunk has no heading path.", "medium"))
    if chunk.page_start is None:
        warnings.append(_warning("missing_page_span", block, chunk.chunk_id, "Chunk has no page span.", "low"))
    if chunk.chunk_type == "reference":
        warnings.append(_warning("reference_chunk_detected", block, chunk.chunk_id, "Reference chunk detected.", "low"))
    if chunk.chunk_type == "method":
        warnings.append(_warning("method_chunk_detected", block, chunk.chunk_id, "Method chunk detected.", "low"))
    if chunk.chunk_type == "unknown":
        warnings.append(_warning("unknown_block_type", block, chunk.chunk_id, "Unknown block type.", "low"))


def _warning(
    warning_type: str, block: ParsedBlock, chunk_id: str | None, message: str, severity: str
) -> ChunkBuildWarning:
    return ChunkBuildWarning(
        warning_type=warning_type,
        doc_id=block.doc_id,
        block_id=block.block_id,
        chunk_id=chunk_id,
        message=message,
        severity=severity,
    )
