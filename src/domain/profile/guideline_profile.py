"""领域模型文件：定义 Living-Guideline 项目中的核心数据结构，只描述数据形状，不负责文件读写或数据库操作。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.domain.common import JsonDict, SerializableMixin


@dataclass
class GuidelineProfile(SerializableMixin):
    profile_id: str
    record_id: str
    guideline_id: Optional[str] = None
    collection_source: str = ""
    profile_scope: str = "document"
    issuer: str = "unknown"
    issuer_confidence: float = 0.0
    guideline_title: str = ""
    published_year: Optional[str] = None
    grading_system: str = "unknown"
    grading_system_version: str = "unknown"
    grading_confidence: float = 0.0
    dimensions: JsonDict = field(default_factory=dict)
    accepted_strength_terms: List[str] = field(default_factory=list)
    accepted_certainty_terms: List[str] = field(default_factory=list)
    accepted_direction_terms: List[str] = field(default_factory=list)
    canonical_mapping_note: Optional[str] = None
    evidence_for_profile: List[JsonDict] = field(default_factory=list)
    resolution_state: str = "unresolved"
    detector_version: str = ""
    profiled_at: str = ""
    raw_payload: Dict[str, Any] = field(default_factory=dict)

