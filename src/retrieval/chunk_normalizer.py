# 规范化 chunk JSONL 记录的字段契约，供 storage 入库与 hybrid retrieval 共用。
#
# 关键设计：
# - 收敛多个 chunk schema 版本（旧版可能缺 chunk_index / retrieval_text / token_count）；
# - 自动按 doc_id 编号（chunk_index 缺失时由 per_doc_index 兜底）；
# - 强制重新计算 text_for_embedding 与 retrieval_text，保持入库与检索口径一致；
# - 临床科室从 documents 继承 + 按 chunk 实际内容重算（多科室集合）。
"""Normalize chunk JSONL records for storage and retrieval indexes."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator

from src.utils.clinical_department import classify_chunk_departments
from src.utils.io import DATA_DIR, read_jsonl


# 与 src.utils.text.SEARCH_TOKEN_RE 等价的 token 计数正则：
# 英文按 [A-Za-z0-9]+（含连字符/撇号分词），中文按单字。
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?|[\u4e00-\u9fff]")


def load_document_metadata(data_dir: str | Path = DATA_DIR) -> dict[str, dict[str, Any]]:
    """读 documents.jsonl 返回 doc_id → record 字典，供 chunk 字段继承。"""
    return {record["doc_id"]: record for record in read_jsonl(Path(data_dir) / "documents.jsonl")}


def iter_normalized_chunks(
    data_dir: str | Path = DATA_DIR,
    chunks_path: str | Path | None = None,
) -> Iterator[dict[str, Any]]:
    """流式产出规范化后的 chunk 记录。

    行为：
    - 默认读 chunks/all_chunks.jsonl；
    - 若 chunk_index 缺失，按 per_doc_index 自增编号兜底；
    - 始终返回新 dict（normalize_chunk_record 的输出），不修改原始记录。
    """
    data_path = Path(data_dir)
    docs = load_document_metadata(data_path)
    path = Path(chunks_path) if chunks_path else data_path / "chunks" / "all_chunks.jsonl"
    per_doc_index: dict[str, int] = {}
    for record in read_jsonl(path):
        doc_id = helper_doc_id(record)
        chunk_index = record.get("chunk_index")
        if chunk_index is None:
            chunk_index = per_doc_index.get(doc_id, 0)
        per_doc_index[doc_id] = int(chunk_index) + 1
        yield normalize_chunk_record(record, docs, int(chunk_index))


def normalize_chunk_record(
    record: dict[str, Any],
    documents: dict[str, dict[str, Any]] | None = None,
    chunk_index: int = 0,
) -> dict[str, Any]:
    """把 chunk 记录统一为下游契约所需的字段集。

    输入兼容：doc_id 来自 chunk 或 source_doc_id；
    section_path 兼容 heading_path 别名；
    content 兼容 text / recommendation 别名；
    字段继承顺序：旧字段 > documents 元数据 > 默认值。
    重新计算项：chunk_type / text_for_embedding / retrieval_text / token_count / clinical_department(s)。
    """
    doc_id = helper_doc_id(record)
    doc = (documents or {}).get(doc_id, {})
    section_path = as_list(record.get("section_path") or record.get("heading_path") or [])
    chunk_type = str(record.get("chunk_type") or "other")
    content = str(record.get("content") or record.get("text") or record.get("recommendation") or "")
    title = str(record.get("title") or doc.get("title") or (section_path[0] if section_path else ""))
    text_for_embedding = str(record.get("text_for_embedding") or helper_embedding_text(section_path, chunk_type, content))
    retrieval = str(record.get("retrieval_text") or text_for_embedding)
    parent_departments = doc.get("clinical_departments") or [doc.get("clinical_department") or "未分类"]
    department_result = classify_chunk_departments(section_path, content, parent_departments, chunk_type)

    normalized = dict(record)
    normalized.update(
        {
            "doc_id": doc_id,
            "title": title,
            "publication_date": record.get("publication_date") or doc.get("publication_date") or "unknown",
            "source_institution": record.get("source_institution") or doc.get("source_institution") or "Unknown",
            "clinical_department": department_result["clinical_department"],
            "clinical_departments": department_result["clinical_departments"],
            "department_scope": department_result["department_scope"],
            "document_kind": record.get("document_kind") or doc.get("document_kind") or "guideline",
            "section_path": section_path,
            "chunk_index": int(record.get("chunk_index") if record.get("chunk_index") is not None else chunk_index),
            "content": content,
            "retrieval_text": retrieval,
            "chunk_type": chunk_type,
            "token_count": int(record.get("token_count") or len(TOKEN_RE.findall(content))),
            "retrieval_key": record.get("retrieval_key") or f"{doc_id}#{chunk_index}",
            "source_file": record.get("source_file") or doc.get("source_file") or "",
            "markdown_clean_path": record.get("markdown_clean_path") or doc.get("markdown_clean_path") or "",
            "is_background": bool(record.get("is_background") or chunk_type == "background"),
            "is_reference_section": bool(record.get("is_reference_section")),
            "text_for_embedding": text_for_embedding,
            "recommendation": record.get("recommendation") or "",
            "evidence": as_list(record.get("evidence") or []),
            "metadata": record.get("metadata") or {},
        }
    )
    return normalized


def helper_doc_id(record: dict[str, Any]) -> str:
    """提取 doc_id（兼容 doc_id / source_doc_id 两种字段名）。

    缺失时抛 KeyError（避免 silently 写入空 doc_id 导致检索断裂）。
    """
    doc_id = record.get("doc_id") or record.get("source_doc_id")
    if not doc_id:
        raise KeyError("chunk record missing doc_id/source_doc_id")
    return str(doc_id)


def as_list(value: Any) -> list[Any]:
    """把字段值规整为 list：None → []，list 原样，其它 → [value]。"""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def helper_embedding_text(section_path: list[Any], chunk_type: str, content: str) -> str:
    """构造 text_for_embedding：把章节路径 + chunk 类型 + 内容拼成结构化文本。

    前缀标签（[GUIDELINE SECTION] / [CHUNK TYPE]）让向量模型在嵌入时能区分章节与正文。
    """
    return "\n".join(
        [
            "[GUIDELINE SECTION]: " + " > ".join(str(item) for item in section_path),
            "[CHUNK TYPE]: " + chunk_type,
            "[CONTENT]:",
            content,
        ]
    )