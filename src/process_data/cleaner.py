from __future__ import annotations

"""面向新流水线的原始 record 清洗模块。

主接口直接接收 JSON record dict, 不再要求调用方先包装成旧的 Document。
旧 `DocumentCleaner.clean()` 仍保留为兼容入口。

目前的 clean_inline_text() 会对只存在数字的单行进行删除, 
后续验证此措施会不会对一些重要医疗信息的产生影响
"""

import re
from typing import Any, Dict, Iterable, List, Tuple

from src.utils.document import Document


TEXT_FIELDS: Tuple[str, ...] = ("content_markdown", "content", "abstract")
MIN_TEXT_CHARS = 20

MOJIBAKE_REPLACEMENTS = {
    "\u920d?": ">=",
    "\u920d\ufffd": ">=",
    "\u920d\u6a9a": "'s",
    "\u920d\u6a9b": "'t",
    "\u920d\u6a99": "'r",
    "\u920d\u6a9d": "'v",
    "\u920d\u6a91": "'l",
    "\u920d\uff1f": "'",
    "\u76f2": "a",
    "\ufffd": "",
}


def get_text_field(record: Dict[str, Any], text_fields: Tuple[str, ...] = TEXT_FIELDS) -> str:
    """返回 record 中优先使用的正文键名。"""

    for field in text_fields:
        if str(record.get(field) or "").strip():
            return field
    return text_fields[0]


def clean_inline_text(text: str) -> str:
    """清洗单行文本中的乱码、引用标记和多余空白。"""

    text = str(text or "")
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("/n", "\n")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(bad, good)

    text = re.sub(r"(?im)^\s*#+\s*", "", text)
    text = re.sub(r"(?im)^\s*[-*]\s+", "", text)
    text = re.sub(r"(?im)^\s*(Path|Citation|Footnotes?|References?)\s*$", "", text)
    text = re.sub(r"(?im)^\s*(\[\d+\]|\(\w\)|[a-z]|\d+)\s*$", "", text)
    text = re.sub(r"(?i)\bCitation\s+", "", text)
    text = re.sub(r"\s+\[\d+\]\s+", " ", text)
    text = re.sub(r"\s+\([a-z]\)\s+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]])", r"\1", text)
    return text.strip()


def clean_structured_text(text: str) -> str:
    """按行清洗正文并保留换行，避免破坏 section/parser 边界。"""

    text = str(text or "")
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("/n", "\n")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", text)
    lines = [clean_inline_text(line) for line in text.split("\n")]
    return "\n".join(line for line in lines if line.strip()).strip()


def clean_record(record: Dict[str, Any], text_fields: Tuple[str, ...] = TEXT_FIELDS) -> Dict[str, Any]:
    """清洗单条原始 record，并保留除正文外的全部元数据。"""

    field = get_text_field(record, text_fields)
    cleaned = dict(record)
    cleaned[field] = clean_structured_text(str(record.get(field) or ""))
    return cleaned


def clean_records(
    records: Iterable[Dict[str, Any]],
    text_fields: Tuple[str, ...] = TEXT_FIELDS,
    min_chars: int = MIN_TEXT_CHARS,
) -> List[Dict[str, Any]]:
    """批量清洗 record；正文过短的记录会被过滤。"""

    cleaned_records: List[Dict[str, Any]] = []
    for record in records:
        cleaned = clean_record(record, text_fields=text_fields)
        text = str(cleaned.get(get_text_field(cleaned, text_fields)) or "").strip()
        if len(text) < min_chars:
            continue
        cleaned_records.append(cleaned)
    return cleaned_records


class DocumentCleaner:
    """兼容旧代码的 Document 清洗器。"""

    MOJIBAKE_REPLACEMENTS = MOJIBAKE_REPLACEMENTS

    def clean_text(self, text: str) -> str:
        """兼容旧接口：清洗文本并压成单行。"""

        return re.sub(r"\n+", " ", clean_structured_text(text)).strip()

    def clean(self, documents: List[Document]) -> List[Document]:
        """兼容旧接口：清洗 Document 列表。"""

        cleaned_docs: List[Document] = []
        for doc in documents:
            cleaned_text = self.clean_text(doc.page_content)
            if len(cleaned_text) < MIN_TEXT_CHARS:
                continue
            doc.page_content = cleaned_text
            cleaned_docs.append(doc)
        return cleaned_docs
