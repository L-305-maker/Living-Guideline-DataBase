# 跨层共享的 Pydantic 数据模型：pipeline 产出、MCP 入参与出参。
#
# 三个核心模型：
# - DocumentRecord：documents.jsonl 中每行记录的强类型定义；
# - SectionRecord：sections/*.jsonl 中每行记录；
# - ChunkRecord：chunks/*.jsonl 中每行记录。
#
# 三个 MCP 入参模型：SearchInput / ReadInput / RetrieveInput，用于 Pydantic 输入校验。
#
# 容错：当 pydantic 不可用时降级为简单 BaseModel（字段直接 setattr），
# 避免在精简环境中导入失败。
"""Strict JSON-facing schemas used by pipeline and MCP APIs."""

from __future__ import annotations

from typing import Any

try:
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    class BaseModel:
        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

        def model_dump(self) -> dict[str, Any]:
            return dict(self.__dict__)

    def Field(default: Any = None, default_factory: Any | None = None) -> Any:  # noqa: N802
        return default_factory() if default_factory else default


class DocumentRecord(BaseModel):
    """documents.jsonl 每行记录的强类型契约。

    关键字段分组：
    - 身份：doc_id（必填唯一）
    - 元数据：title / publication_date / source_institution / clinical_department(s) / document_kind
    - 路径：source_file / markdown_raw_path / markdown_clean_path
    - 内容：abstract / content_sha256 / content_md（实际只写前两个）
    - 清洗与 OCR 审计：cleaning_quality / cleaning_flags / pdf_* / ocr_*
    """
    doc_id: str
    title: str
    publication_date: str
    source_institution: str
    clinical_department: str = "未分类"
    clinical_departments: list[str] = Field(default_factory=list)
    department_scope: str = "single"
    document_kind: str = "guideline"
    source_file: str
    markdown_raw_path: str
    markdown_clean_path: str
    abstract: str = ""
    content_sha256: str = ""
    cleaning_quality: str = ""
    cleaning_flags: str = ""
    source_pdf_text_quality: str = ""
    source_pdf_needs_ocr: bool = False
    source_pdf_is_scanned: bool = False
    pdf_text_quality: str = ""
    pdf_needs_ocr: bool = False
    pdf_is_scanned: bool = False
    ocr_engine: str = ""
    ocr_applied: bool = False
    ocr_status: str = ""
    ocr_error: str = ""


class SectionRecord(BaseModel):
    """sections/*.jsonl 每行记录的强类型契约。

    section_path 是 heading 层级路径（如 ["2. 治疗", "2.1 药物"]）；
    char_start / char_end 是 section 在 markdown_clean 中的字符偏移，便于对齐 chunk。
    """
    doc_id: str
    title: str
    publication_date: str
    source_institution: str
    clinical_department: str = "未分类"
    clinical_departments: list[str] = Field(default_factory=list)
    department_scope: str = "single"
    document_kind: str = "guideline"
    section_path: list[str] = Field(default_factory=list)
    heading: str | None = None
    heading_level: int | None = None
    char_start: int = 0
    char_end: int = 0
    content: str = ""
    is_reference_section: bool = False


class ChunkRecord(BaseModel):
    """chunks/*.jsonl 每行记录的强类型契约。

    chunk_id / retrieval_key 是稳定主键（跨 JSONL / PG / MCP）；
    retrieval_text 是 BM25 检索口径（构造见 semantic_chunker.retrieval_text）；
    text_for_embedding 是向量化口径（构造见 chunk_normalizer.normalize_chunk_record）。
    """
    chunk_id: str
    doc_id: str
    title: str
    publication_date: str
    source_institution: str
    clinical_department: str = "未分类"
    clinical_departments: list[str] = Field(default_factory=list)
    department_scope: str = "single"
    document_kind: str = "guideline"
    section_path: list[str] = Field(default_factory=list)
    chunk_index: int
    content: str
    retrieval_text: str = ""
    chunk_type: str = "other"
    token_count: int = 0
    retrieval_key: str
    source_file: str = ""
    markdown_clean_path: str = ""
    is_background: bool = False
    is_reference_section: bool = False


class SearchInput(BaseModel):
    """MCP search 工具入参契约。

    过滤项：source_institution / clinical_department / time_range / publication_date 全部可选；
    recency_boost 用于在排序阶段对新近文献加权；
    debug=True 时 MCP 返回中间调试字段（如 rank_maps）。
    """
    query: str
    source_institution: str | None = None
    clinical_department: str | None = None
    time_range: str | dict[str, str] | None = None
    publication_date: str | None = None
    recency_boost: bool = False
    topk: int = 20
    debug: bool = False


class ReadInput(BaseModel):
    """MCP read 工具入参契约。

    doc_id 与 title 互斥（至少传一个）；max_chars 控制返回正文长度，避免大文档撑爆 MCP 响应。
    """
    doc_id: str | None = None
    title: str | None = None
    max_chars: int | None = None


class RetrieveInput(BaseModel):
    """MCP retrieve 工具入参契约。结构与 SearchInput 类似，但 topk 默认 30（chunk 更小）。"""
    query: str
    topk: int = 30
    source_institution: str | None = None
    clinical_department: str | None = None
    time_range: str | dict[str, str] | None = None
    publication_date: str | None = None
    debug: bool = False


def dump_model(model: BaseModel) -> dict[str, Any]:
    """统一 dump 入口：pydantic v2 用 model_dump，pydantic v1 用 dict() 兜底。"""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()