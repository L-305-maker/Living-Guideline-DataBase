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
    query: str
    source_institution: str | None = None
    clinical_department: str | None = None
    time_range: str | dict[str, str] | None = None
    publication_date: str | None = None
    recency_boost: bool = False
    topk: int = 20
    debug: bool = False


class ReadInput(BaseModel):
    doc_id: str | None = None
    title: str | None = None
    max_chars: int | None = None


class RetrieveInput(BaseModel):
    query: str
    topk: int = 30
    source_institution: str | None = None
    clinical_department: str | None = None
    time_range: str | dict[str, str] | None = None
    publication_date: str | None = None
    debug: bool = False


def dump_model(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()
