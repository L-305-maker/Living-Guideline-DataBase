# Guideline chunk 构建的 dataclass 集合。
#
# 主要 dataclass：
# - DocumentMeta：文档级元数据；
# - ParsedBlock：解析后的单个 block（含 heading_path / char span / page）；
# - SectionChunk / AtomicChunk / TableChunk：三种 chunk 类型；
# - ChunkLink：chunk 间关系（前/后/父/同章节邻居）；
# - ChunkBuildWarning：构建期 warning；
# - ChunkRetrieveResult：检索服务返回结构（含 debug 字段）。
"""Dataclasses for guideline chunk construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, TypeVar


T = TypeVar("T")


@dataclass
class DocumentMeta:
    """文档级元数据（从 metadata 目录或 markdown 路径派生）。"""
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
    """Markdown 解析后的单个 block（含 heading_path / char span / page）。"""
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
    """section 级 chunk：含 child_block_ids 与 child_chunk_ids（双向溯源）。"""
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
    """atomic 级 chunk（不可再分）；含 prev/next 指针支持顺序遍历。"""
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
    """表格 chunk：含 table_id（关联 table_parent 与 table_row）、row_index 与 column_headers。"""
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
    """chunk 间关系链接：parent_section / previous_chunk / next_chunk / table_* / same_section。"""
    link_id: str
    doc_id: str
    source_chunk_id: str
    target_chunk_id: str
    link_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkBuildWarning:
    """构建期 warning（type / severity / message 便于 review 工具聚合）。"""
    warning_type: str
    doc_id: str
    block_id: str | None
    chunk_id: str | None
    message: str
    severity: str


@dataclass
class ChunkRetrieveResult:
    """ChunkRetrieveService 返回结构；debug 字段含 raw_score / matched_terms 便于排查。"""
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
    """dataclass → dict 序列化（依赖 dataclasses.asdict）。"""
    return asdict(obj)


def from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """dict → dataclass 反序列化。"""
    return cls(**data)


def meta_from_block(block: ParsedBlock) -> DocumentMeta:
    """从 ParsedBlock.metadata 反向构造 DocumentMeta。"""
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