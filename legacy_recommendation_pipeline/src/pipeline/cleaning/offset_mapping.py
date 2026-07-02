"""清洗阶段文件：把来源记录整理成可追溯的 cleaned record，并在进入解析前处理 PDF 噪声和质量信号。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List


JsonDict = Dict[str, Any]
OFFSET_MAPPING_VERSION = "offset_mapping_v1"


@dataclass(frozen=True)
class OffsetMapResult:
    """clean 文本片段映射回 raw 文本后的坐标结果。

    cleaning 阶段可能修复断词、删除页眉页脚、压缩空白，因此 clean offset 不一定
    能直接等同 raw offset。本对象显式记录映射置信度和映射方法，供抽取结果溯源使用。
    """

    clean_start: int
    clean_end: int
    raw_start: int | None
    raw_end: int | None
    mapping_confidence: str
    mapping_method: str

    def to_dict(self) -> JsonDict:
        return {
            "clean_start": self.clean_start,
            "clean_end": self.clean_end,
            "raw_start": self.raw_start,
            "raw_end": self.raw_end,
            "mapping_confidence": self.mapping_confidence,
            "mapping_method": self.mapping_method,
        }


def _normalized_with_index(text: str) -> tuple[str, List[int]]:
    """生成用于匹配的规范化文本，同时保留规范化字符到原始字符的索引映射。"""

    chars: List[str] = []
    indexes: List[int] = []
    pending_space_index: int | None = None
    index = 0
    text = str(text or "")
    while index < len(text):
        char = text[index]
        if char == "-" and index + 1 < len(text) and text[index + 1].isspace():
            index += 1
            while index < len(text) and text[index].isspace():
                index += 1
            continue
        if char.isspace():
            if chars and chars[-1] != " ":
                pending_space_index = index
                chars.append(" ")
                indexes.append(index)
            index += 1
            continue
        chars.append(char.lower())
        indexes.append(index if pending_space_index is None else index)
        pending_space_index = None
        index += 1
    if chars and chars[-1] == " ":
        chars.pop()
        indexes.pop()
    return "".join(chars), indexes


def map_clean_span_to_raw(raw_text: str, clean_text: str, clean_start: int, clean_end: int) -> OffsetMapResult:
    """把 clean_content 中的 span 尽力映射回 raw_content 坐标。

    映射策略从严格到宽松依次为：原文子串匹配、规范化子串匹配、SequenceMatcher
    模糊匹配。若仍无法定位，则返回 unmapped，而不是伪造 raw offset。
    """

    clean_text = str(clean_text or "")
    raw_text = str(raw_text or "")
    clean_start = max(0, int(clean_start or 0))
    clean_end = min(len(clean_text), int(clean_end or 0))
    if clean_end <= clean_start:
        return OffsetMapResult(clean_start, clean_end, None, None, "unmapped", "empty_span")

    clean_span = clean_text[clean_start:clean_end]
    exact = raw_text.find(clean_span)
    if exact >= 0:
        return OffsetMapResult(clean_start, clean_end, exact, exact + len(clean_span), "exact", "substring")

    normalized_raw, raw_indexes = _normalized_with_index(raw_text)
    normalized_span, _span_indexes = _normalized_with_index(clean_span)
    if normalized_span:
        normalized_exact = normalized_raw.find(normalized_span)
        if normalized_exact >= 0:
            raw_start = raw_indexes[normalized_exact]
            raw_end = raw_indexes[normalized_exact + len(normalized_span) - 1] + 1
            return OffsetMapResult(clean_start, clean_end, raw_start, raw_end, "normalized", "normalized_substring")

        match = SequenceMatcher(None, normalized_span, normalized_raw, autojunk=False).find_longest_match(
            0,
            len(normalized_span),
            0,
            len(normalized_raw),
        )
        if len(normalized_span) and match.size / len(normalized_span) >= 0.75:
            raw_start = raw_indexes[match.b]
            raw_end = raw_indexes[match.b + match.size - 1] + 1
            return OffsetMapResult(clean_start, clean_end, raw_start, raw_end, "fuzzy", "sequence_matcher")

    return OffsetMapResult(clean_start, clean_end, None, None, "unmapped", "not_found")


def span_offset_fields(raw_text: str, clean_text: str, clean_start: int, clean_end: int) -> JsonDict:
    """返回抽取实体统一使用的 clean/raw 坐标字段和 offset_mapping 元数据。"""

    result = map_clean_span_to_raw(raw_text, clean_text, clean_start, clean_end)
    return {
        "clean_start_char": result.clean_start,
        "clean_end_char": result.clean_end,
        "raw_start_char": result.raw_start,
        "raw_end_char": result.raw_end,
        "offset_mapping": {
            "version": OFFSET_MAPPING_VERSION,
            "confidence": result.mapping_confidence,
            "method": result.mapping_method,
        },
    }

