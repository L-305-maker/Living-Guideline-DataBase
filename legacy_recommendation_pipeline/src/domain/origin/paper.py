"""领域模型文件：定义 Living-Guideline 项目中的核心数据结构，只描述数据形状，不负责文件读写或数据库操作。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from dataclasses import dataclass, field
from typing import Optional

from src.domain.common import ArticleType, JsonDict, SerializableMixin


@dataclass
class Paper(SerializableMixin):
    paper_id: str  # 文献唯一 ID，用于关联 EvidenceItem.paper_id。
    title: str  # 文献标题，入库时应尽量保留原始大小写和标点。
    source: str  # 文献来源或数据库来源，例如 pubmed、guideline_reference、origin。
    article_type: ArticleType = "unclear"  # 文献类型，例如 RCT、cohort、meta_analysis、guideline。
    authors: list[str] = field(default_factory=list)  # 作者列表，保持来源顺序。
    journal: Optional[str] = None  # 期刊名称。
    publication_year: Optional[str] = None  # 发表年份，来源只能提供年份时使用。
    publication_date: Optional[str] = None  # 精确发表日期，建议使用 ISO 日期字符串。
    doi: Optional[str] = None  # DOI。
    pmid: Optional[str] = None  # PubMed ID。
    url: Optional[str] = None  # 文献网页、DOI 或 PDF URL。
    abstract: Optional[str] = None  # 摘要文本。
    full_text: Optional[str] = None  # 全文文本；只有合法获取全文后才填入。
    raw_pdf_path: Optional[str] = None  # 本地原始 PDF 路径，用于追溯和重新解析。
    guideline_id: Optional[str] = None  # 若文献来自某指南或作为指南引用，关联 Guideline.guideline_id。
    source_record_id: Optional[str] = None  # 来源清洗记录 ID，关联 SourceRecord.record_id。
    metadata: JsonDict = field(default_factory=dict)  # 其他未结构化文献元数据。
    created_at: str = ""  # 入库创建时间。
    updated_at: str = ""  # 入库更新时间。

