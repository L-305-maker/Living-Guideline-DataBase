"""Dataclasses for guideline chunk construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, TypeVar


T = TypeVar("T")


@dataclass
class DocumentMeta:
    doc_id: str
    title: str | None = None
    publisher: str | None = None
    language: str | None = None
    document_type: str | None = None
    publication_date: str | None = None
    last_updated_date: str | None = None
    source_url: str | None = None


@dataclass
class ParsedBlock:
    block_id: str
    doc_id: str
    block_type: str
    heading_path: list[str]
    heading_level: int | None
    text: str
    page_start: int | None
    page_end: int | None
    char_start: int | None
    char_end: int | None
    order_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SectionChunk:
    section_id: str
    doc_id: str
    heading_path: list[str]
    heading_level: int
    title: str
    text: str
    child_block_ids: list[str]
    child_chunk_ids: list[str]
    page_start: int | None
    page_end: int | None
    order_start: int
    order_end: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AtomicChunk:
    chunk_id: str
    doc_id: str
    parent_section_id: str | None
    chunk_type: str
    heading_path: list[str]
    text: str
    text_for_embedding: str
    source_block_ids: list[str]
    page_start: int | None
    page_end: int | None
    order_start: int
    order_end: int
    prev_chunk_id: str | None
    next_chunk_id: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TableChunk:
    chunk_id: str
    doc_id: str
    parent_section_id: str | None
    table_id: str
    chunk_type: str
    heading_path: list[str]
    table_title: str | None
    text: str
    text_for_embedding: str
    row_index: int | None
    column_headers: list[str]
    page_start: int | None
    page_end: int | None
    source_block_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkLink:
    link_id: str
    doc_id: str
    source_chunk_id: str
    target_chunk_id: str
    link_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkBuildWarning:
    warning_type: str
    doc_id: str
    block_id: str | None
    chunk_id: str | None
    message: str
    severity: str


@dataclass
class ChunkRetrieveResult:
    chunk_id: str
    doc_id: str
    chunk_type: str
    score: float
    title: str | None
    publisher: str | None
    heading_path: list[str]
    text: str
    source_span: dict[str, Any]
    debug: dict[str, Any]


def to_dict(obj: Any) -> dict[str, Any]:
    return asdict(obj)


def from_dict(cls: type[T], data: dict[str, Any]) -> T:
    return cls(**data)


def meta_from_block(block: ParsedBlock) -> DocumentMeta:
    metadata = block.metadata or {}
    return DocumentMeta(
        doc_id=block.doc_id,
        title=metadata.get("title"),
        publisher=metadata.get("publisher"),
        language=metadata.get("language"),
        document_type=metadata.get("document_type"),
        publication_date=metadata.get("publication_date"),
        last_updated_date=metadata.get("last_updated_date"),
        source_url=metadata.get("source_url"),
    )
