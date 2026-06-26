from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.domain.common.types import Block_type
from src.domain.common import SerializableMixin

@dataclass
class SourceBlock(SerializableMixin):
    block_id: str
    record_id: str
    guideline_id: Optional[str] = None
    paper_id: Optional[str] = None
    source: str = ""
    title: str = ""
    section_path: list[str] = field(default_factory=list)
    heading: str = ""
    block_type: Block_type = "unknown"
    text: str = ""
    order: int = 0
    char_start: int = 0
    char_end: int = 0
    token_estimate: int = 0
    candidate_hints: List[str] = field(default_factory=list)
    quality: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# 兼容旧代码中的 Sourceblock 拼写；新代码应统一使用 SourceBlock。
Sourceblock = SourceBlock
