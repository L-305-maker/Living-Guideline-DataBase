"""领域模型文件：定义 Living-Guideline 项目中的核心数据结构，只描述数据形状，不负责文件读写或数据库操作。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from dataclasses import dataclass, field
from typing import Optional

from src.domain.common import GuidelineStatus, GuidelineType, JsonDict, SerializableMixin


@dataclass
class Guideline(SerializableMixin):
    guideline_id: str  # 指南唯一ID，是后续推荐、PICO、证据和更新日志的主线ID
    title: str  # 指南标题，优先使用发布方正式标题
    issuer: str  # 发布机构，例如 NICE、WHO、KDIGO
    source: str  # 数据来源或站点简称，例如 nice、who、kdigo、esc
    guideline_type: GuidelineType = "standard"  # 指南类型：living、standard、rapid、consensus
    status: GuidelineStatus = "active"  # 指南状态：draft、active、archived、retired
    version: Optional[str] = None  # 指南版本号或版本标签
    publication_year: Optional[str] = None  # 发布年份，第一轮清洗可从原始数据直接提取
    publication_date: Optional[str] = None  # 精确发布日期，能从数据中获得时填写
    update_date: Optional[str] = None  # 最近更新日期，适合 living guideline 追踪
    url: Optional[str] = None  # 指南网页 URL 或入口 URL
    doi: Optional[str] = None  # 指南 DOI，如有则填写
    raw_pdf_path: Optional[str] = None  # 本地原始 PDF 路径，用于回溯和重新解析
    language: Optional[str] = None  # 指南语言，例如 en、zh
    country_or_region: Optional[str] = None  # 发布国家或地区
    topic: Optional[str] = None  # 指南主题或疾病领域
    source_record_ids: list[str] = field(default_factory=list)  # 组成该指南的来源记录ID列表
    metadata: JsonDict = field(default_factory=dict)  # 其他来源元数据，保留不稳定或站点特异字段
    created_at: str = ""  # 入库创建时间
    updated_at: str = ""  # 入库更新时间

