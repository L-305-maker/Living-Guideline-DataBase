# 把清洗后的 Markdown 编码为 section 记录（heading-aware 段落）。
#
# 输出文件落在 sections/<doc_id>.jsonl，每个 heading 是一个 SectionRecord，
# 字段包含 section_path / heading / heading_level / char_start / char_end / content。
#
# 设计要点：
# - section 边界由 Markdown 标题（# ~ ######）切分；没有标题的文档退化为单个 "Document" 块；
# - 通过正则检测参考文献列表（"参考文献" / "REFERENCES"）并打 is_reference_section，
#   方便下游 chunk 阶段排除参考段落。
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


# 匹配 Markdown 标题行：1-6 个 # + 标题文本 + 行尾。
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.M)

# 参考文献段识别（标题 / front-matter 标记 / 列表起始三种信号）：
# - 标题匹配 references / bibliography / 参考文献（不区分大小写）
# - HTML 注释 <!-- reference_section: true -->
# - 行内以 "## REFERENCES" 或 "## 参考文献" 起头紧跟编号列表
REFERENCE_RE = re.compile(r"^(references|bibliography|\u53c2\u8003\u6587\u732e)$", re.I)
REFERENCE_MARKER_RE = re.compile(r"<!--\s*reference_section:\s*true\s*-->", re.I)
REFERENCE_LIST_RE = re.compile(r"(?:^|\n)\s*(?:#{1,6}\s*)?(?:REFERENCES|BIBLIOGRAPHY|\u53c2\u8003\u6587\u732e)\s*\n\s*(?:\d+\.|\[\d+\])", re.I)


def helper_is_reference_section(content: str, section_path: list[str]) -> bool:
    """判定一个 section 是否属于参考文献列表。

    三个判定源（任一命中即认为是参考文献段）：
    - section_path 任一项是 "references" / "bibliography" / "参考文献"
    - content 含 HTML 标记 <!-- reference_section: true -->
    - content 包含 "## REFERENCES" 或 "## 参考文献" 紧跟编号列表
    """
    normalized_path = [item.strip() for item in section_path if item and item.strip()]
    return bool(
        any(REFERENCE_RE.match(item) for item in normalized_path)
        or REFERENCE_MARKER_RE.search(content)
        or REFERENCE_LIST_RE.search(content)
    )


def encode_markdown(markdown: str) -> list[SectionRecord]:
    """把单篇 Markdown 编码为 SectionRecord 列表。

    关键实现：
    - 解析 front-matter 得到文档级元数据（doc_id / title / 科室等）；
    - 用 HEADING_RE 匹配所有标题，按 heading 栈维护 section_path；
    - 第一个标题之前的正文作为 "Preface" 单独成块（不丢前置摘要等）；
    - 每个 section 记录 char_start / char_end 便于后续 chunk 对齐。
    """
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
        # 无标题文档退化为单块 "Document"，便于下游 chunk 阶段仍能产出检索单位。
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
    # 前置摘要（Preface）：第一个标题之前的正文；不存在则跳过。
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
        # 维护 heading 栈：同级或更高级的标题先 pop，再 push 当前；
        # 这样 section_path 始终是 "层级路径"，例如 ["2. 治疗", "2.1 药物"]。
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
    """编码单个 Markdown 文件并写入 sections/<doc_id>.jsonl。

    返回编码后的 SectionRecord 列表（同时也被持久化）。
    """
    path = Path(clean_path)
    sections = encode_markdown(path.read_text(encoding="utf-8", errors="replace"))
    out = ensure_dir(output_dir) / f"{sections[0].doc_id}.jsonl"
    write_jsonl(out, [dump_model(section) for section in sections])
    return sections


def encode_all(input_dir: str | Path = DATA_DIR / "markdown_clean", output_dir: str | Path = DATA_DIR / "sections") -> dict[str, Any]:
    """批量编码 input_dir 下的 Markdown，并清理 output_dir 中遗留的过期 JSONL。

    清理策略：只保留本轮扫描到的 doc_id 对应的 JSONL；其余视为 stale 文件删除。
    这样在文档被重新分桶后，旧 JSONL 不会继续被检索链路读到。
    """
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
    """CLI 入口：python -m src.pipeline.cleaning.encoder [flags]。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(DATA_DIR / "markdown_clean"))
    parser.add_argument("--output-dir", default=str(DATA_DIR / "sections"))
    args = parser.parse_args()
    print(json.dumps(encode_all(args.input_dir, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()