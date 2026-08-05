# 把 section 进一步切成可检索、可追溯的小 chunk。
#
# 流程：
#   markdown → encode_markdown（按 heading 切 sections）→
#   对每个 section 调 split_section_semantic_content 二次切分（按 token 预算 + 类型）→
#   生成 ChunkRecord（chunk_id / section_path / retrieval_key / token_count 等）。
#
# 关键不变量：
# - retrieval_key = "<doc_id>#<chunk_index>" 是稳定主键；
# - 切不到任何 chunk 的文档会走 helper_fallback_chunk，保证至少有一个可检索单位。
"""Section-aware chunking with traceable retrieval keys."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.models.schemas import ChunkRecord, dump_model
from src.pipeline.cleaning.encoder import encode_markdown
from src.pipeline.cleaning.semantic_chunker import (
    estimate_tokens,
    retrieval_text,
    split_section_semantic_content,
)
from src.retrieval.document_repr.section_classifier import classify_section
from src.utils.clinical_department import classify_chunk_departments
from src.utils.front_matter import parse_front_matter
from src.utils.ids import make_chunk_id
from src.utils.io import DATA_DIR, ensure_dir, iter_markdown_files, write_jsonl


# 写入 JSONL 的字段集；新增字段需同步修改这里 + compact_chunk_record 的取值逻辑，
# 否则新字段会被静默丢弃。
CHUNK_OUTPUT_FIELDS = (
    "chunk_id",
    "doc_id",
    "title",
    "publication_date",
    "source_institution",
    "clinical_departments",
    "document_kind",
    "section_path",
    "chunk_index",
    "content",
    "retrieval_text",
    "chunk_type",
    "token_count",
    "retrieval_key",
    "is_background",
    "is_reference_section",
)


def compact_chunk_record(record: dict[str, Any] | ChunkRecord) -> dict[str, Any]:
    """把 ChunkRecord 投影为最小字段集，并归一化 clinical_departments。

    兼容两种来源：
    - Pydantic ChunkRecord（dump_model 后取值）
    - 已 dump 的 dict

    clinical_departments 归一化：
    - list 直接用；
    - "A|B" 字符串按 '|' 拆；
    - 缺省或为空时回退到 clinical_department，再缺省 ["未分类"]。
    最终用 dict.fromkeys 去重保序。
    """
    values = dump_model(record) if isinstance(record, ChunkRecord) else dict(record)
    labels = values.get("clinical_departments") or []
    if isinstance(labels, str):
        labels = labels.split("|")
    if not labels:
        labels = str(values.get("clinical_department") or "未分类").split("|")
    values["clinical_departments"] = list(
        dict.fromkeys(str(label).strip() for label in labels if str(label).strip())
    ) or ["未分类"]
    return {field: values.get(field) for field in CHUNK_OUTPUT_FIELDS}


def helper_fallback_chunk_content(metadata: dict[str, Any], body: str) -> str:
    """兜底 chunk 的正文：从纯文本中筛掉注释行/空白行，取前 ~1200 字符。

    用于无法按 heading 切块的退化文档（如整篇单行 PDF），保证 chunk 链路不丢文档。
    """
    title = str(metadata.get("title") or "").strip()
    lines: list[str] = []
    char_count = 0
    for raw_line in body.splitlines():
        line = raw_line.strip(" #*\t")
        if not line or line.startswith("<!--"):
            continue
        lines.append(line)
        char_count += len(line)
        if char_count >= 1200:
            break
    content = " ".join(lines).strip() or title
    # 把标题拼回开头，避免纯目录式文档产生空 chunk 的同时保留强信号标题。
    if title and title not in content[: max(80, len(title) + 20)]:
        content = f"{title} {content}".strip()
    return content[:2000]


def helper_fallback_chunk(metadata: dict[str, Any], body: str, sections: list[Any], clean_path: str) -> ChunkRecord | None:
    """兜底 chunk：构造单个 chunk_type='summary' 的 ChunkRecord。

    关键设计：
    - doc_id 优先取 front-matter 'id'，否则取 sections[0].doc_id；
    - 若仍然为空则返回 None（视为无法挽救的文档）；
    - section_path 暂用 [title]，便于检索时与正常 chunk 兼容。
    """
    content = helper_fallback_chunk_content(metadata, body)
    if not content:
        return None
    section = sections[0] if sections else None
    doc_id = str(metadata.get("id") or getattr(section, "doc_id", ""))
    if not doc_id:
        return None
    title = str(metadata.get("title") or getattr(section, "title", "") or doc_id)
    section_path = [title]
    parent_departments = (
        metadata.get("clinical_departments")
        or getattr(section, "clinical_departments", [])
        or [metadata.get("clinical_department") or "未分类"]
    )
    department_result = classify_chunk_departments(section_path, content, parent_departments, "summary")
    return ChunkRecord(
        chunk_id=make_chunk_id(doc_id, 0),
        doc_id=doc_id,
        title=title,
        publication_date=str(metadata.get("publication_date") or getattr(section, "publication_date", "") or "unknown"),
        source_institution=str(metadata.get("source_institution") or getattr(section, "source_institution", "") or "Unknown"),
        clinical_department=department_result["clinical_department"],
        clinical_departments=department_result["clinical_departments"],
        department_scope=department_result["department_scope"],
        document_kind=metadata.get("document_kind") or "guideline",
        section_path=section_path,
        chunk_index=0,
        content=content,
        retrieval_text=retrieval_text(title, section_path, "summary", content),
        chunk_type="summary",
        token_count=estimate_tokens(content),
        retrieval_key=f"{doc_id}#0",
        source_file=metadata.get("source_file", ""),
        markdown_clean_path=clean_path,
        is_background=False,
        is_reference_section=False,
    )


def chunk_markdown(markdown: str, clean_path: str = "") -> list[ChunkRecord]:
    """对单篇 Markdown 做两阶段切分：section 级 + section 内语义级。

    流程：
    1. parse_front_matter 取文档级元数据；
    2. encode_markdown 切 section；
    3. 对每个 section 调 split_section_semantic_content 按 token 预算切分；
    4. 用 classify_section 推断 chunk_type（background / summary / recommendation …）；
    5. 若整篇没产出 chunk，构造 helper_fallback_chunk 作为兜底。
    """
    metadata, body = parse_front_matter(markdown)
    sections = encode_markdown(markdown)
    chunks: list[ChunkRecord] = []
    chunk_index = 0
    for section in sections:
        # 参考文献段强制标记 section_type='reference'，跳过正常分类。
        section_type = "reference" if section.is_reference_section else classify_section(
            heading=section.heading or "",
            section_path=section.section_path,
            content=section.content,
        )
        for part in split_section_semantic_content(
            section.content,
            heading=section.heading or "",
            section_path=section.section_path,
            section_type=section_type,
        ):
            chunk_id = make_chunk_id(section.doc_id, chunk_index)
            retrieval = retrieval_text(section.title, section.section_path, part.chunk_type, part.content)
            parent_departments = metadata.get("clinical_departments") or section.clinical_departments or [section.clinical_department]
            department_result = classify_chunk_departments(section.section_path, part.content, parent_departments, part.chunk_type)
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    doc_id=section.doc_id,
                    title=section.title,
                    publication_date=section.publication_date or "unknown",
                    source_institution=section.source_institution or "Unknown",
                    clinical_department=department_result["clinical_department"],
                    clinical_departments=department_result["clinical_departments"],
                    department_scope=department_result["department_scope"],
                    document_kind=metadata.get("document_kind") or "guideline",
                    section_path=section.section_path,
                    chunk_index=chunk_index,
                    content=part.content,
                    retrieval_text=retrieval,
                    chunk_type=part.chunk_type,
                    token_count=part.token_count,
                    retrieval_key=f"{section.doc_id}#{chunk_index}",
                    source_file=metadata.get("source_file", ""),
                    markdown_clean_path=clean_path,
                    is_background=part.chunk_type == "background",
                    is_reference_section=section.is_reference_section,
                )
            )
            chunk_index += 1
    if not chunks:
        fallback = helper_fallback_chunk(metadata, body, sections, clean_path)
        if fallback is not None:
            chunks.append(fallback)
    return chunks


def chunk_file(clean_path: str | Path, output_dir: str | Path = DATA_DIR / "chunks") -> list[ChunkRecord]:
    """对单个 Markdown 文件做切块并写入 chunks/<doc_id>.jsonl。"""
    path = Path(clean_path)
    markdown = path.read_text(encoding="utf-8", errors="replace")
    metadata, _body = parse_front_matter(markdown)
    doc_id = metadata["id"]
    chunks = chunk_markdown(markdown, str(path))
    out = ensure_dir(output_dir) / f"{doc_id}.jsonl"
    write_jsonl(out, [compact_chunk_record(chunk) for chunk in chunks])
    return chunks


def chunk_all(input_dir: str | Path = DATA_DIR / "markdown_clean", output_dir: str | Path = DATA_DIR / "chunks") -> dict[str, Any]:
    """批量切块 input_dir 下的 Markdown；同时构建 chunks/all_chunks.jsonl 聚合文件。

    聚合文件由所有 chunk 拼接而成，便于 PostgreSQL 入库阶段一次性 COPY。
    清理策略：移除 active_doc_ids 之外的 *.jsonl（与 encoder.encode_all 一致）。
    """
    all_chunks: list[dict[str, Any]] = []
    docs = 0
    active_doc_ids: set[str] = set()
    for path in iter_markdown_files(input_dir):
        chunks = chunk_file(path, output_dir)
        active_doc_ids.add(chunks[0].doc_id if chunks else path.stem)
        docs += 1
        all_chunks.extend(compact_chunk_record(chunk) for chunk in chunks)
    chunks_dir = Path(output_dir)
    write_jsonl(chunks_dir / "all_chunks.jsonl", all_chunks)
    stale_files_removed = 0
    for path in chunks_dir.glob("*.jsonl"):
        # 聚合文件 all_chunks.jsonl 永远保留；其它未在 active_doc_ids 的视为过期。
        if path.name != "all_chunks.jsonl" and path.stem not in active_doc_ids:
            path.unlink()
            stale_files_removed += 1
    return {
        "documents": docs,
        "chunks": len(all_chunks),
        "stale_files_removed": stale_files_removed,
        "chunks_dir": str(output_dir),
    }


def main() -> None:
    """CLI 入口：python -m src.pipeline.cleaning.chunker [flags]。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(DATA_DIR / "markdown_clean"))
    parser.add_argument("--output-dir", default=str(DATA_DIR / "chunks"))
    args = parser.parse_args()
    print(json.dumps(chunk_all(args.input_dir, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()