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
