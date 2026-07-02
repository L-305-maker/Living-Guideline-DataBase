"""领域模型文件：定义 Living-Guideline 项目中的核心数据结构，只描述数据形状，不负责文件读写或数据库操作。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from dataclasses import dataclass, field
from typing import Optional

from src.domain.common import EffectDirection, ExtractionMethod, JsonDict, ScreeningStatus, SerializableMixin, StudyDesign


@dataclass
class EvidenceItem(SerializableMixin):
    evidence_id: str
    paper_id: Optional[str]
    pico_id: str
    source_record_id: Optional[str] = None
    model_trace_id: Optional[str] = None
    recommendation_version_id: Optional[str] = None
    recommendation_candidate_id: Optional[str] = None
    study_design: Optional[StudyDesign] = None
    sample_size: Optional[int] = None
    population_extracted: Optional[str] = None
    intervention_extracted: Optional[str] = None
    comparator_extracted: Optional[str] = None
    outcomes_extracted: list[JsonDict] = field(default_factory=list)
    effect_size: JsonDict = field(default_factory=dict)
    confidence_interval: Optional[str] = None
    effect_direction: EffectDirection = "uncertain"
    extraction_method: ExtractionMethod = "manual"
    extraction_confidence: Optional[float] = None
    screening_status: ScreeningStatus = "uncertain"
    exclusion_reason: Optional[str] = None
    source_span: Optional[str] = None
    created_at: str = ""

