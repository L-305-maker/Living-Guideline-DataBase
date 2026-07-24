"""Section encoding for clean Markdown documents."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from src.models.schemas import SectionRecord, dump_model
from src.utils.front_matter import parse_front_matter
from src.utils.io import DATA_DIR, ensure_dir, iter_markdown_files, write_jsonl


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.M)
REFERENCE_RE = re.compile(r"^(references|bibliography|\u53c2\u8003\u6587\u732e)$", re.I)
REFERENCE_MARKER_RE = re.compile(r"<!--\s*reference_section:\s*true\s*-->", re.I)
REFERENCE_LIST_RE = re.compile(r"(?:^|\n)\s*(?:#{1,6}\s*)?(?:REFERENCES|BIBLIOGRAPHY|\u53c2\u8003\u6587\u732e)\s*\n\s*(?:\d+\.|\[\d+\])", re.I)


def helper_is_reference_section(content: str, section_path: list[str]) -> bool:
    normalized_path = [item.strip() for item in section_path if item and item.strip()]
    return bool(
        any(REFERENCE_RE.match(item) for item in normalized_path)
        or REFERENCE_MARKER_RE.search(content)
        or REFERENCE_LIST_RE.search(content)
    )


def encode_markdown(markdown: str) -> list[SectionRecord]:
    # front matter 提供文档级元数据，标题栈只负责章节路径，二者不能在段落循环中混淆。
    metadata, body = parse_front_matter(markdown)
    matches = list(HEADING_RE.finditer(body))
    doc_id = metadata.get("id", "")
    title = metadata.get("title", "")
    publication_date = metadata.get("publication_date") or "unknown"
    source_institution = metadata.get("source_institution") or "Unknown"
    clinical_department = metadata.get("clinical_department") or "未分类"
    clinical_departments = metadata.get("clinical_departments") or [clinical_department]
    department_scope = metadata.get("department_scope") or ("compositive" if len(clinical_departments) > 1 else "single")
    document_kind = metadata.get("document_kind") or "guideline"
    if not matches:
        return [
            SectionRecord(
                doc_id=doc_id,
                title=title,
                publication_date=publication_date,
                source_institution=source_institution,
                clinical_department=clinical_department,
                clinical_departments=clinical_departments,
                department_scope=department_scope,
                document_kind=document_kind,
                section_path=["Document"],
                heading="Document",
                heading_level=1,
                char_start=0,
                char_end=len(body),
                content=body.strip(),
                is_reference_section=helper_is_reference_section(body, ["Document"]),
            )
        ]

    sections: list[SectionRecord] = []
    stack: list[tuple[int, str]] = []
    if matches[0].start() > 0 and body[: matches[0].start()].strip():
        preface = body[: matches[0].start()]
        sections.append(
            SectionRecord(
                doc_id=doc_id,
                title=title,
                publication_date=publication_date,
                source_institution=source_institution,
                clinical_department=clinical_department,
                clinical_departments=clinical_departments,
                department_scope=department_scope,
                document_kind=document_kind,
                section_path=["Preface"],
                heading="Preface",
                heading_level=1,
                char_start=0,
                char_end=matches[0].start(),
                content=preface.strip(),
            )
        )

    for index, match in enumerate(matches):
        level = len(match.group(1))
        heading = match.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading))
        path = [item[1] for item in stack]
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        content = body[start:end].strip()
        is_reference_section = helper_is_reference_section(content, path)
        sections.append(
            SectionRecord(
                doc_id=doc_id,
                title=title,
                publication_date=publication_date,
                source_institution=source_institution,
                clinical_department=clinical_department,
                clinical_departments=clinical_departments,
                department_scope=department_scope,
                document_kind=document_kind,
                section_path=path,
                heading=heading,
                heading_level=level,
                char_start=start,
                char_end=end,
                content=content,
                is_reference_section=is_reference_section,
            )
        )
    return sections


def encode_file(clean_path: str | Path, output_dir: str | Path = DATA_DIR / "sections") -> list[SectionRecord]:
    path = Path(clean_path)
    sections = encode_markdown(path.read_text(encoding="utf-8", errors="replace"))
    out = ensure_dir(output_dir) / f"{sections[0].doc_id}.jsonl"
    write_jsonl(out, [dump_model(section) for section in sections])
    return sections


def encode_all(input_dir: str | Path = DATA_DIR / "markdown_clean", output_dir: str | Path = DATA_DIR / "sections") -> dict[str, Any]:
    count_docs = 0
    count_sections = 0
    active_doc_ids: set[str] = set()
    for path in iter_markdown_files(input_dir):
        sections = encode_file(path, output_dir)
        active_doc_ids.add(sections[0].doc_id)
        count_docs += 1
        count_sections += len(sections)
    stale_files_removed = 0
    for path in Path(output_dir).glob("*.jsonl"):
        if path.stem not in active_doc_ids:
            path.unlink()
            stale_files_removed += 1
    return {
        "documents": count_docs,
        "sections": count_sections,
        "stale_files_removed": stale_files_removed,
        "sections_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(DATA_DIR / "markdown_clean"))
    parser.add_argument("--output-dir", default=str(DATA_DIR / "sections"))
    args = parser.parse_args()
    print(json.dumps(encode_all(args.input_dir, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
