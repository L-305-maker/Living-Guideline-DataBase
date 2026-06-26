from dataclasses import dataclass, field
from typing import Optional

from src.domain.common import (
    CandidateStatus,
    Certainty,
    Direction,
    ExtractionMethod,
    JsonDict,
    RecommendationStrength,
    SerializableMixin,
)


@dataclass
class RecommendationCandidate(SerializableMixin):
    candidate_id: str  # 候选推荐意见唯一ID，用于从抽取层追踪到正式推荐版本
    record_id: str  # 来源清洗记录ID，关联 origin.SourceRecord.record_id
    guideline_id: Optional[str] = None  # 所属指南ID，能确定时关联 origin.Guideline.guideline_id
    model_trace_id: Optional[str] = None  # 产生该候选结果的模型运行ID，关联 candidate.ModelTrace.model_trace_id
    source_text: str = ""  # 候选推荐意见在原文中的完整片段
    recommendation_text: str = ""  # 抽取出的推荐意见正文
    recommendation_code: Optional[str] = None  # 原指南中的推荐编号，例如 Recommendation 1、1.2.3
    source_section: Optional[str] = None  # 原文来源章节路径，例如 Treatment > Anticoagulation
    source_url: Optional[str] = None  # 原网页、PDF 或 DOI URL
    population: Optional[str] = None  # 候选推荐适用人群
    intervention: Optional[str] = None  # 候选推荐涉及的干预措施
    comparator: Optional[str] = None  # 候选推荐涉及的对照措施
    outcomes: list[JsonDict] = field(default_factory=list)  # 候选推荐涉及的结局指标列表
    direction: Direction = "unclear"  # 推荐方向：for、against、neutral、no_recommendation、unclear
    strength: RecommendationStrength = "unclear"  # 推荐强度：strong、conditional、weak、good_practice、none、unclear
    certainty: Certainty = "unclear"  # 证据确定性：high、moderate、low、very_low、none、unclear
    rationale: Optional[str] = None  # 原文中支持推荐的理由或解释
    remarks: Optional[str] = None  # 实施说明、适用条件、备注或补充信息
    extraction_method: ExtractionMethod = "model"  # 抽取方式：rule、model、hybrid、manual
    extraction_confidence: Optional[float] = None  # 该候选推荐抽取置信度，建议范围 0.0 到 1.0
    status: CandidateStatus = "pending"  # 候选状态：pending、accepted、rejected、needs_review、merged
    review_note: Optional[str] = None  # 人工审核备注，说明接受、拒绝或合并原因
    normalized_payload: JsonDict = field(default_factory=dict)  # 规范化后的中间结构，供后续生成正式知识表
    raw_payload: JsonDict = field(default_factory=dict)  # 原始抽取结果快照，便于排错和回溯
    created_at: str = ""  # 候选记录创建时间
    updated_at: str = ""  # 候选记录最后更新时间
