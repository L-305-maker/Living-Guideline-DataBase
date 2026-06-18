from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.schema import BlockType, ParsedDocument, Provenance, SectionNode, StructuredBlock, keep_year, normalize_text, stable_hash

'''
目前数据全被拍平为文本, 可能存储重要信息的表格无法利用, 后续重新进行爬虫并对表格进行处理
'''

logger = logging.getLogger(__name__)

# 标题关键词
HEADING_KEYWORDS_RE = re.compile(                              
    r"\b("
    r"recommendations?|summary|background|evidence|rationale|management|diagnosis|"
    r"treatment|prevention|screening|monitoring|clinical policy|patient resources|"
    r"quick reference|guideline statements?|methods?|results|discussion|conclusion"
    r")\b",
    re.IGNORECASE,
)
# 推荐意见关键词
RECOMMENDATION_RE = re.compile(
    r"\b("
    r"we\s+recommend|we\s+suggest|recommend(?:ed|s|ing)?|suggest(?:ed|s|ing)?|"
    r"should|should\s+not|must|must\s+not|"
    r"do\s+not\s+(?:use|administer|offer|provide|initiate|start|continue|treat|screen|monitor|perform)|"
    r"avoid|contraindicat|not\s+recommended|not\s+indicated|may\s+be\s+considered|"
    r"it\s+is\s+reasonable\s+to|class\s+(?:i|ii|iii|iia|iib)|level\s+of\s+evidence|"
    r"quality\s+of\s+evidence|certainty\s+of\s+evidence|strong\s+recommendation|"
    r"conditional\s+recommendation|weak\s+recommendation"
    r")\b",
    re.IGNORECASE,
)
# 表格关键词
TABLE_RE = re.compile(r"\|.+\||\btable\s+\d+\b", re.IGNORECASE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|(?<=:)\s+(?=[A-Z])")
# 内标题拆分
INLINE_HEADING_RE = re.compile(
    r"\b(Quick Reference|Related Clinical Policy|Related Guidelines|Perspectives|Education|"
    r"Apps and Tools|Patient Resources|Slides|Recommendations?|Background|Evidence|"
    r"Rationale|Summary|Diagnosis|Treatment|Management|Prevention|Screening)\b",
    re.I,
)

# 暂存一个 section 的标题、路径和正文行。
@dataclass
class SectionBuffer:
    title: str
    path: str
    level: int
    index: int
    start: int
    lines: List[Tuple[str, int, int]]


# 清洗文本但保留换行结构，避免破坏章节和列表边界。
def clean_text_preserve_structure(text: str) -> str:
    text = str(text or "")
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("/n", "\n")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufffd", "")
    text = re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", text)
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"^\s*#+\s*", "", line)
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            lines.append(line)
        elif lines and lines[-1] != "":
            lines.append("")
    return "\n".join(lines).strip()


# 判断一行文本是否像章节标题。
def looks_like_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 180:
        return False
    if RECOMMENDATION_RE.search(line) and len(line) > 80:
        return False
    if line.endswith((".", ";", ",")) and not re.match(r"^\d+(?:\.\d+)*\.?\s+", line):
        return False
    if re.match(r"^(?:chapter|section|part|appendix)\s+\d+", line, re.I):
        return True
    if re.match(r"^\d+(?:\.\d+){0,4}\.?\s+\S+", line):
        return True
    if re.match(r"^[A-Z][A-Z0-9 /,&():'-]{3,}$", line) and len(line.split()) <= 14:
        return True
    return bool(HEADING_KEYWORDS_RE.search(line) and len(line.split()) <= 14)


# 根据编号或标题形态估计章节层级. 
def heading_level(line: str) -> int:
    match = re.match(r"^(\d+(?:\.\d+)*)\.?\s+", line)
    if match:
        return min(match.group(1).count(".") + 1, 6)
    if re.match(r"^(?:chapter|part)\s+\d+", line, re.I):
        return 1
    if re.match(r"^[A-Z][A-Z0-9 /,&():'-]{3,}$", line):
        return 2
    return 3


# 将被压成一行的常见标题重新拆出来。
def split_inline_headings(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    parts = INLINE_HEADING_RE.split(text)
    if len(parts) == 1:
        return [text]
    out: List[str] = []
    prefix = parts[0].strip()
    if prefix:
        out.append(prefix)
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        out.append(heading)
        if body:
            out.append(body)
    return out


# 迭代结构行并保留字符偏移，供后续 provenance 使用。
def iter_lines_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    non_empty = [line for line in text.split("\n") if line.strip()]
    if len(non_empty) <= 1:
        cursor = 0
        lines = []
        for part in split_inline_headings(text):
            start = text.find(part, cursor)
            if start < 0:
                start = cursor
            end = start + len(part)
            lines.append((part, start, end))
            cursor = end
        return lines

    lines = []
    cursor = 0
    for raw in text.split("\n"):
        line = raw.strip()
        start = text.find(raw, cursor)
        end = start + len(raw)
        cursor = end + 1
        if line:
            lines.append((line, start, end))
    return lines


# 将结构块粗分为推荐、叙事或表格。
def classify_block(text: str) -> BlockType:
    if TABLE_RE.search(text):
        return "table"
    if RECOMMENDATION_RE.search(text):
        return "recommendation"
    return "narrative"


# 将过长单行拆成更小句子，避免整篇单行文档形成超大 block。
def expand_long_lines(lines: List[Tuple[str, int, int]]) -> List[Tuple[str, int, int]]:
    expanded: List[Tuple[str, int, int]] = []
    for line, start, end in lines:
        if len(line) <= 700 or TABLE_RE.search(line):
            expanded.append((line, start, end))
            continue

        cursor = start
        for part in SENTENCE_SPLIT_RE.split(line):
            part = part.strip()
            if not part:
                continue
            part_start = line.find(part, max(0, cursor - start))
            abs_start = start + part_start if part_start >= 0 else cursor
            abs_end = abs_start + len(part)
            expanded.append((part, abs_start, abs_end))
            cursor = abs_end
    return expanded


# 判断当前行是否应该开启新的 block。
def starts_new_block(line: str) -> bool:
    return bool(re.match(r"^(?:[-*]|\d+\.|[A-Z]\.)\s+", line) or RECOMMENDATION_RE.search(line))


# 根据 section 与文本范围构造统一的 StructuredBlock。
def make_structured_block(section: SectionBuffer, text: str, start: int, end: int) -> StructuredBlock:
    rec_number_match = re.search(r"\b(?:recommendation|statement)\s+([0-9A-Za-z.-]+)", text, re.I)
    return StructuredBlock(
        type=classify_block(text),
        text=text,
        char_start=start,
        char_end=end,
        section_path=section.path,
        section_title=section.title,
        section_index=section.index,
        rec_number=rec_number_match.group(1) if rec_number_match else None,
    )


# 将一个 section 内部进一步聚合成结构 block。
def split_block_units(section: SectionBuffer) -> List[StructuredBlock]:
    text = "\n".join(line for line, _, _ in section.lines).strip()
    if not text:
        return []

    chunks: List[Tuple[str, int, int]] = []
    current: List[str] = []
    current_start: Optional[int] = None
    current_end = section.start

    for line, start, end in expand_long_lines(section.lines):
        if current and starts_new_block(line):
            chunks.append(("\n".join(current), current_start or section.start, current_end))
            current = []
            current_start = None
        if current_start is None:
            current_start = start
        current.append(line)
        current_end = end
    if current:
        chunks.append(("\n".join(current), current_start or section.start, current_end))

    blocks: List[StructuredBlock] = []
    for block_text, start, end in chunks:
        block_text = normalize_text(block_text)
        if not block_text:
            continue
        blocks.append(make_structured_block(section, block_text, start, end))
    return blocks


# 解析单条原始 JSON 记录，失败时降级为单 narrative 文档。
def parse_record(record: Dict[str, Any], fallback_source: str = "") -> ParsedDocument:
    source = str(record.get("source") or fallback_source or "").strip()
    title = str(record.get("title") or "").strip()
    url = str(record.get("url") or "").strip()
    text = clean_text_preserve_structure(record.get("content_markdown") or record.get("content") or record.get("abstract") or "")
    doc_id = stable_hash(url, title, source, text[:500])
    provenance = Provenance(
        source=source,
        issuer=str(record.get("issuer") or "").strip(),
        title=title,
        url=url,
        published_date=keep_year(record.get("published_date") or record.get("last_updated")),
        doc_id=doc_id,
        char_start=0,
        char_end=len(text),
    )

    try:
        return parse_text(text, provenance)
    except Exception as exc:
        logger.exception("Structure parsing failed for doc_id=%s: %s", doc_id, exc)
        section = SectionNode(stable_hash(doc_id, "section", 0), "Document", title or "Document", 1, 0, text)
        block = StructuredBlock("narrative", normalize_text(text), 0, len(text), section.path, section.title, 0)
        return ParsedDocument(doc_id=doc_id, provenance=provenance, sections=[section], blocks=[block])


# 执行真正的 section 树解析，并生成 section 与 block 列表。
def parse_text(text: str, provenance: Provenance) -> ParsedDocument:
    root_title = provenance.title or "Document"
    stack: List[Tuple[int, str]] = [(0, root_title)]
    buffers: List[SectionBuffer] = []
    current = SectionBuffer("Document", root_title, 1, 0, 0, [])

    # 把当前 section 缓冲区落盘，避免标题切换时丢失正文。
    def flush() -> None:
        nonlocal current
        if current.lines:
            buffers.append(current)

    for line, start, end in iter_lines_with_offsets(text):
        if looks_like_heading(line):
            flush()
            level = heading_level(line)
            # 使用栈维护标题层级，形成稳定的面包屑路径。
            while stack and stack[-1][0] >= level:
                stack.pop()
            parent_path = " > ".join(title for _, title in stack if title)
            path = f"{parent_path} > {line}".strip(" >") if parent_path else line
            current = SectionBuffer(line.strip(" :"), path, level, len(buffers), start, [])
            stack.append((level, current.title))
        else:
            current.lines.append((line, start, end))
    flush()

    if not buffers:
        buffers = [SectionBuffer("Document", root_title, 1, 0, 0, [(text, 0, len(text))])]

    sections: List[SectionNode] = []
    blocks: List[StructuredBlock] = []
    for index, buffer in enumerate(buffers):
        section_text = "\n".join(line for line, _, _ in buffer.lines)
        summary = normalize_text(section_text)[:500]
        section_id = stable_hash(provenance.doc_id, "section", index, buffer.path)
        # parent collection 使用 section 作为基本单位。
        sections.append(
            SectionNode(
                section_id=section_id,
                title=buffer.title,
                path=buffer.path,
                level=buffer.level,
                index=index,
                text=section_text,
                summary=summary,
            )
        )
        buffer.index = index
        blocks.extend(split_block_units(buffer))

    return ParsedDocument(doc_id=provenance.doc_id, provenance=provenance, sections=sections, blocks=blocks)
