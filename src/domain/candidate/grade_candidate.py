"""领域模型文件：定义 Living-Guideline 项目中的核心数据结构，只描述数据形状，不负责文件读写或数据库操作。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from dataclasses import dataclass, field
from typing import Optional

from src.domain.common import (
    CandidateStatus,
    Certainty,
    ExtractionMethod,
    GradeDomainJudgement,
    GradeSystem,
    JsonDict,
    PublicationBiasJudgement,
    RecommendationStrength,
    SerializableMixin,
)


@dataclass
class GradeCandidate(SerializableMixin):
    grade_candidate_id: str
    recommendation_candidate_id: str
    record_id: str
    guideline_id: Optional[str] = None
    pico_id: Optional[str] = None
    outcome_name: Optional[str] = None
    model_trace_id: Optional[str] = None
    grade_system: GradeSystem = "unknown"
    certainty: Certainty = "unclear"
    strength: RecommendationStrength = "unclear"
    risk_of_bias: GradeDomainJudgement = "not_reported"
    inconsistency: GradeDomainJudgement = "not_reported"
    indirectness: GradeDomainJudgement = "not_reported"
    imprecision: GradeDomainJudgement = "not_reported"
    publication_bias: PublicationBiasJudgement = "not_reported"
    reasons_for_downgrade: list[str] = field(default_factory=list)
    reasons_for_upgrade: list[str] = field(default_factory=list)
    source_text: str = ""
    judgement_rationale: Optional[str] = None
    source_span: Optional[str] = None
    source_section: Optional[str] = None
    source_url: Optional[str] = None
    extraction_method: ExtractionMethod = "model"
    extraction_confidence: Optional[float] = None
    status: CandidateStatus = "pending"
    review_note: Optional[str] = None
    normalized_payload: JsonDict = field(default_factory=dict)
    raw_payload: JsonDict = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

