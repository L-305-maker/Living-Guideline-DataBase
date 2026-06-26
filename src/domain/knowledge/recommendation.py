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
