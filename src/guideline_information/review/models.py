"""Review-specific models."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from src.guideline_information.enums import ReviewStatus
from src.guideline_information.models import FrozenModel, SCHEMA_VERSION, utc_now


class ReviewStateChange(FrozenModel):
    sample_id: str
    from_status: ReviewStatus
    to_status: ReviewStatus
    actor_id: str
    reason: str = ""
    schema_version: str = SCHEMA_VERSION
    changed_at: datetime = Field(default_factory=utc_now)
