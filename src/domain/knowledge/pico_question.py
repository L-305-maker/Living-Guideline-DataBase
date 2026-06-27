"""领域模型文件：定义 Living-Guideline 项目中的核心数据结构，只描述数据形状，不负责文件读写或数据库操作。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from dataclasses import dataclass, field
from typing import Optional

from src.domain.common import ExtractionMethod, JsonDict, PicoStatus, Priority, SerializableMixin


@dataclass
class PicoQuestion(SerializableMixin):
    pico_id: str
    guideline_id: str
    clinical_question: str
    population: str
    intervention: str
    comparator: Optional[str] = None
    outcomes: list[JsonDict] = field(default_factory=list)
    priority: Optional[Priority] = None
    status: PicoStatus = "active"
    source_record_id: Optional[str] = None
    source_text: Optional[str] = None
    source_span: Optional[str] = None
    extraction_method: ExtractionMethod = "manual"
    created_at: str = ""
    updated_at: str = ""

