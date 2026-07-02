"""通用基础工具文件：提供 JSONL、文本、质量上下文或项目公共辅助能力，供多个流水线阶段复用。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict

from src.domain.common import stable_id


JsonDict = Dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def source_section(block: JsonDict) -> str:
    section_path = block.get("section_path")
    if isinstance(section_path, list):
        return " > ".join(str(item) for item in section_path if item)
    return str(section_path or "")


def source_url(block: JsonDict) -> str:
    metadata = block.get("metadata")
    if isinstance(metadata, dict):
        return str(metadata.get("source_url") or "")
    return ""


def source_metadata(block: JsonDict) -> JsonDict:
    metadata = block.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    section_path = block.get("section_path")
    return {
        "source": str(block.get("source") or ""),
        "title": str(block.get("title") or ""),
        "paper_id": str(block.get("paper_id") or ""),
        "section_path": section_path if isinstance(section_path, list) else [],
        "heading": str(block.get("heading") or ""),
        "raw_pdf_path": str(metadata.get("raw_pdf_path") or ""),
        "source_url": str(metadata.get("source_url") or ""),
    }

