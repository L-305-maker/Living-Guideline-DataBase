"""领域模型文件：定义 Living-Guideline 项目中的核心数据结构，只描述数据形状，不负责文件读写或数据库操作。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from dataclasses import dataclass, field
from typing import Optional

from src.domain.common import (
    Certainty,
    Direction,
    JsonDict,
    RecommendationStatus,
    RecommendationStrength,
    SerializableMixin,
)


@dataclass
class Recommendation(SerializableMixin):
    recommendation_id: str
    guideline_id: str
    current_version_id: Optional[str] = None
    status: RecommendationStatus = "active"
    recommendation_code: Optional[str] = None
    title: Optional[str] = None
    population: Optional[str] = None
    intervention: Optional[str] = None
    comparator: Optional[str] = None
    direction: Direction = "unclear"
    strength: RecommendationStrength = "unclear"
    certainty: Certainty = "unclear"
    first_created_at: str = ""
    last_published_at: str = ""
    last_updated_at: str = ""
    withdrawn_at: Optional[str] = None
    withdrawn_reason: Optional[str] = None
    superseded_by_recommendation_id: Optional[str] = None
    normalized_payload: JsonDict = field(default_factory=dict)

