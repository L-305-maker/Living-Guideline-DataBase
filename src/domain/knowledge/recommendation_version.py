from dataclasses import dataclass, field
from typing import Optional

from src.domain.common import (
    Certainty,
    Direction,
    JsonDict,
    RecommendationChangeType,
    RecommendationStrength,
    SerializableMixin,
)


@dataclass
class RecommendationVersion(SerializableMixin):
    recommendation_version_id: str
    recommendation_candidate_id: str
    guideline_id: str
    version_number: str
    recommendation_text: str
    recommendation_id: Optional[str] = None
    previous_version_id: Optional[str] = None
    pico_id: Optional[str] = None
    record_id: Optional[str] = None
    grade_candidate_id: Optional[str] = None
    profile_id: Optional[str] = None
    issuer: Optional[str] = None
    grading_system: Optional[str] = None
    quality_status: str = "needs_review"
    recommendation_code: Optional[str] = None
    direction: Direction = "unclear"
    strength: RecommendationStrength = "unclear"
    certainty: Certainty = "unclear"
    rationale: Optional[str] = None
    remarks: Optional[str] = None
    population: Optional[str] = None
    intervention: Optional[str] = None
    comparator: Optional[str] = None
    outcome_summary: Optional[str] = None
    source_text: Optional[str] = None
    source_span: Optional[str] = None
    source_span_ref: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    source_section: Optional[str] = None
    source_url: Optional[str] = None
    change_type: RecommendationChangeType = "new"
    change_summary: Optional[str] = None
    created_at: str = ""
    published_at: Optional[str] = None
    normalized_payload: JsonDict = field(default_factory=dict)
    raw_payload: JsonDict = field(default_factory=dict)
