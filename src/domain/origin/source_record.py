from dataclasses import dataclass, field
from typing import Optional

from src.domain.common import JsonDict, SerializableMixin


@dataclass
class SourceRecord(SerializableMixin):
    record_id: str  # 清洗后来源记录唯一ID，连接原始输入、结构解析和候选抽取
    content: str  # 清洗后的正文全文，后续 structure_parser 基于它分割章节、block 或 chunk
    title: str  # 来源文档标题，通常来自 origin title 或 source_cleaner 修正结果
    source: str  # 数据来源简称，例如 nice、who、kdigo、esc
    guideline_id: Optional[str] = None  # 指向 Guideline.guideline_id，表示该来源记录归属的指南实体
    paper_id: Optional[str] = None  # 指向 Paper.paper_id，当前可作为来源文献或后续证据文献入口
    issuer: Optional[str] = None  # 发布机构，例如 NICE、WHO、KDIGO
    url: Optional[str] = None  # 来源网页、DOI 或 PDF URL
    published_year: Optional[str] = None  # 来源记录可直接得到的年份，通常来自 origin published_year
    raw_pdf_path: Optional[str] = None  # 本地原始 PDF 路径，用于回查和重新解析
    url_provenance: Optional[str] = None  # URL 来源说明，例如 metadata、doi_from_text、not_found
    tables: list[list[list[str]]] = field(default_factory=list)  # 第一轮保留的原始二维表格结构，后续统一处理
    table_count: int = 0  # 表格数量统计，便于后续判断是否需要表格解析
    table_row_count: int = 0  # 所有表格的行数统计
    table_cell_count: int = 0  # 所有表格的单元格统计
    guideline_seed: JsonDict = field(default_factory=dict)  # 可直接生成 Guideline 的 seed 数据
    paper_seed: JsonDict = field(default_factory=dict)  # 可直接生成 Paper 的 seed 数据
    direct_extraction: JsonDict = field(default_factory=dict)  # source_cleaner 直接提取字段汇总
    raw_record: JsonDict = field(default_factory=dict)  # 原始或清洗后完整 record 快照，避免溯源信息丢失
    created_at: str = ""  # 入库创建时间，建议由存储层统一填充
