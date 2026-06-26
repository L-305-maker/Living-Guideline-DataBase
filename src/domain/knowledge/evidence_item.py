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
