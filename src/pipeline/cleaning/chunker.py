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
    title = str(metadata.get("title") or "").strip()
    lines: list[str] = []
    char_count = 0
    for raw_line in body.splitlines():
        line = raw_line.strip(" #*	")
        if not line or line.startswith("<!--"):
            continue
        lines.append(line)
        char_count += len(line)
        if char_count >= 1200:
            break
    content = " ".join(lines).strip() or title
    if title and title not in content[: max(80, len(title) + 20)]:
        content = f"{title} {content}".strip()
    return content[:2000]


def helper_fallback_chunk(metadata: dict[str, Any], body: str, sections: list[Any], clean_path: str) -> ChunkRecord | None:
    content = helper_fallback_chunk_content(metadata, body)
    if not content:
        return None
    section = sections[0] if sections else None
    doc_id = str(metadata.get("id") or getattr(section, "doc_id", ""))
    if not doc_id:
        return None
    title = str(metadata.get("title") or getattr(section, "title", "") or doc_id)
    section_path = [title]
    parent_departments = metadata.get("clinical_departments") or getattr(section, "clinical_departments", []) or [metadata.get("clinical_department") or "\u672a\u5206\u7c7b"]
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
    # 先按章节语义边界生成分块；正文无法切分时才建立单个可追踪兜底块。
    metadata, body = parse_front_matter(markdown)
    sections = encode_markdown(markdown)
    chunks: list[ChunkRecord] = []
    chunk_index = 0
    for section in sections:
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
    path = Path(clean_path)
    markdown = path.read_text(encoding="utf-8", errors="replace")
    metadata, _body = parse_front_matter(markdown)
    doc_id = metadata["id"]
    chunks = chunk_markdown(markdown, str(path))
    out = ensure_dir(output_dir) / f"{doc_id}.jsonl"
    write_jsonl(out, [compact_chunk_record(chunk) for chunk in chunks])
    return chunks


def chunk_all(input_dir: str | Path = DATA_DIR / "markdown_clean", output_dir: str | Path = DATA_DIR / "chunks") -> dict[str, Any]:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(DATA_DIR / "markdown_clean"))
    parser.add_argument("--output-dir", default=str(DATA_DIR / "chunks"))
    args = parser.parse_args()
    print(json.dumps(chunk_all(args.input_dir, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
