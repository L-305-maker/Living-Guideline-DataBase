from __future__ import annotations

"""
面向新流水线的 record 去重模块。

主接口直接接收 JSON record dict: 旧 `deduplicator(List[Document])` 保留兼容。

目前最大问题使仅针对正文部分进行去重，但如果有某些指南在不同网站的某些关键信息不同也会被去除, 
后续进一步考虑去重措施的合理性与可执行性
"""

import hashlib
from typing import Any, Dict, Iterable, List, Tuple

from src.process_data.cleaner import TEXT_FIELDS, get_text_field
from src.utils.document import Document


def normalize_for_hash(text: str) -> str:
    """把文本标准化成适合哈希比较的形式。"""

    return " ".join(str(text or "").split())


def hash_text(text: str) -> str:
    """计算文本内容的稳定 sha256 哈希。"""

    normalized = normalize_for_hash(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def record_text(record: Dict[str, Any], text_fields: Tuple[str, ...] = TEXT_FIELDS) -> str:
    """从 record 中取出去重使用的正文文本。"""

    return str(record.get(get_text_field(record, text_fields)) or "")


def deduplicate_records(records: Iterable[Dict[str, Any]], text_fields: Tuple[str, ...] = TEXT_FIELDS) -> List[Dict[str, Any]]:
    """按正文内容对 record 列表去重，保留第一次出现的记录。"""

    seen: set[str] = set()
    unique: List[Dict[str, Any]] = []
    for record in records:
        content_hash = hash_text(record_text(record, text_fields=text_fields))
        if content_hash in seen:
            continue
        seen.add(content_hash)
        unique.append(record)
    return unique


def deduplicator(data: List[Document]) -> List[Document]:
    """兼容旧接口：按 Document.page_content 去重。"""

    seen: set[str] = set()
    unique: List[Document] = []
    for document in data:
        content_hash = hash_text(document.page_content)
        if content_hash in seen:
            continue
        seen.add(content_hash)
        unique.append(document)
    return unique
