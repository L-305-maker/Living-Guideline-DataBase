from __future__ import annotations

"""清洗阶段的轻量去重工具。

当前去重只根据正文文本计算哈希，保留第一次出现的记录。这个策略适合清理完全重复
的抓取结果，但不适合判断“不同网站上的同一指南是否语义等价”。如果后续要做指南级
合并，应新增来源、版本、发布日期、URL 和人工审核信息，而不是扩展这里的文本哈希。
"""

import hashlib
from typing import Any, Dict, Iterable, List, Tuple


TEXT_FIELDS: Tuple[str, ...] = ("content", "content_markdown", "abstract", "text", "page_content")


def get_text_field(record: Dict[str, Any], text_fields: Tuple[str, ...] = TEXT_FIELDS) -> str:
    """返回 record 中第一个有内容的文本字段名。"""

    for field in text_fields:
        if record.get(field):
            return field
    return text_fields[0]


def normalize_for_hash(text: str) -> str:
    """把文本压缩空白后用于哈希，避免换行或多空格导致重复记录无法合并。"""

    return " ".join(str(text or "").split())


def hash_text(text: str) -> str:
    """计算规范化文本的稳定 sha256 哈希。"""

    normalized = normalize_for_hash(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def record_text(record: Dict[str, Any], text_fields: Tuple[str, ...] = TEXT_FIELDS) -> str:
    """从 record 中取出用于去重的正文文本。"""

    return str(record.get(get_text_field(record, text_fields)) or "")


def deduplicate_records(records: Iterable[Dict[str, Any]], text_fields: Tuple[str, ...] = TEXT_FIELDS) -> List[Dict[str, Any]]:
    """按正文哈希对记录去重，保留第一次出现的记录。"""

    seen: set[str] = set()
    unique: List[Dict[str, Any]] = []
    for record in records:
        content_hash = hash_text(record_text(record, text_fields=text_fields))
        if content_hash in seen:
            continue
        seen.add(content_hash)
        unique.append(record)
    return unique


def deduplicator(data: List[Any]) -> List[Any]:
    """兼容旧版 Document-like 对象的去重入口。"""

    seen: set[str] = set()
    unique: List[Any] = []
    for item in data:
        content_hash = hash_text(getattr(item, "page_content", ""))
        if content_hash in seen:
            continue
        seen.add(content_hash)
        unique.append(item)
    return unique
