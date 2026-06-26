"""把 cleaned source record 拆成可追溯的 SourceBlock 行。

结构解析阶段负责保留 section path、字符 offset、block 级质量信号和粗粒度候选提示。
它不做最终抽取决定；真正的 recommendation、GRADE、PICO、evidence 判定交给 router
和各抽取器完成。这样可以让“结构恢复”和“领域抽取”保持边界清晰。
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.domain.common import stable_id
from src.pipeline.cleaning.base_cleaner import clean_inline_text
from src.common.process_jsonl import iter_jsonl, write_jsonl
from src.common.record_quality import record_quality_context


# 本模块负责把一条 cleaned record 拆成多个 SourceBlock。

JsonDict = Dict[str, Any]


@dataclass
class BlockBuildInput:
    """构建单个 SourceBlock 所需的参数包，避免函数参数列表过长。"""

    record: JsonDict
    text: str
    block_type: str
    section_path: List[str]
    heading: str
    order: int
    char_start: int
    char_end: int
    parent_line_type: Optional[str] = None
    is_low_value: bool = False
    forced_heading: bool = False


@dataclass
class ParseState:
    """解析单篇 cleaned document 时持续更新的状态。"""

    section_stack: List[tuple[int, str]]
    current_heading: str
    blocks: List[JsonDict]
    order: int = 0

HEADING_KEYWORDS_RE = re.compile(
    r"\b("
    r"recommendations?|summary|background|evidence|rationale|management|diagnosis|"
    r"treatment|therapy|prevention|screening|monitoring|methods?|scope|"
    r"clinical questions?|key priorities|guideline statements?|implementation"
    r")\b",
    re.I,
)
RECOMMENDATION_HINT_RE = re.compile(
    r"\b("
    r"we\s+recommend|we\s+suggest|recommend(?:ed|s|ing)?|suggest(?:ed|s|ing)?|"
    r"should|should\s+not|must|must\s+not|do\s+not|offer|consider|avoid|"
    r"not\s+recommended|contraindicat|may\s+be\s+considered"
    r")\b",
    re.I,
)
GRADE_HINT_RE = re.compile(
    r"\b("
    r"grade|certainty\s+of\s+evidence|quality\s+of\s+evidence|"
    r"high\s+certainty|moderate\s+certainty|low\s+certainty|very\s+low\s+certainty|"
    r"strong\s+recommendation|conditional\s+recommendation|weak\s+recommendation|"
    r"risk\s+of\s+bias|inconsistency|indirectness|imprecision|publication\s+bias|"
    r"level\s+of\s+evidence|class\s+(?:i|ii|iii|iia|iib)|loe"
    r")\b",
    re.I,
)
PICO_HINT_RE = re.compile(
    r"\b(?:clinical\s+question|pico|picot|population|intervention|comparator|comparison|outcomes?)\b|"
    r"\b(?:in|for|among)\s+(?:adults?|children|patients?)\s+with\b",
    re.I,
)
EVIDENCE_HINT_RE = re.compile(
    r"\b("
    r"systematic\s+review|meta-analysis|randomi[sz]ed|trial|cohort|case-control|"
    r"risk\s+ratio|odds\s+ratio|hazard\s+ratio|confidence\s+interval|"
    r"evidence\s+review|study|studies"
    r")\b",
    re.I,
)
TABLE_HINT_RE = re.compile(r"^\s*\|.+\|\s*$|^\s*table\s+\d+[\s:.-]", re.I)
LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+(?:\.\d+)*[.)]\s+|[A-Za-z][.)]\s+)")
NUMBERED_HEADING_RE = re.compile(r"^\s*(\d+(?:\.\d+){0,5})\.?\s+\S+")
MARKDOWN_HEADING_RE = re.compile(r"^\s*(#{1,6})\s+(.+)")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|(?<=:)\s+(?=[A-Z])")
INLINE_HEADING_RE = re.compile(
    r"\b("
    r"Recommendations?|Background|Evidence|Rationale|Methods?|Summary|Treatment|"
    r"Diagnosis|Management|Prevention|References?|Bibliography|Disclosures?|"
    r"Conflict of Interest|Acknowledgements?|Funding|Appendix|Supplementary Material"
    r")\b",
    re.I,
)
LOW_VALUE_HEADING_RE = re.compile(
    r"\b("
    r"references?|bibliography|acknowledgements?|disclosures?|conflicts?\s+of\s+interest|"
    r"author\s+affiliations?|funding|copyright|appendix|supplementary\s+material"
    r")\b",
    re.I,
)
REFERENCE_LIKE_RE = re.compile(
    r"(?i)(?:"
    r"\bdoi:\s*10\.|\b10\.\d{4,9}/|"
    r"\b(?:pubmed|pmid|clin\s+infect\s+dis|jama|lancet|n\s+engl\s+j\s+med|"
    r"antimicrob\s+agents\s+chemother|arch\s+\w+|journal)\b|"
    r"\b(?:19|20)\d{2};\d+[\w()]*:\w?\d+"
    r")"
)
AFFILIATION_LIKE_RE = re.compile(
    r"(?i)\b(?:department|division|university|school\s+of\s+medicine|hospital|medical\s+center),"
)


def normalize_block_text(text: Any) -> str:
    """在标题识别和块分类之前，统一做行内文本清洗。"""

    return clean_inline_text(str(text or ""))


def estimate_tokens(text: str) -> int:
    """用轻量规则估算 token 数，供后续批处理和质量判断参考。"""

    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def is_list_item(line: str) -> bool:
    """判断当前行是否像项目符号、编号条目或罗马数字条目。"""

    return bool(LIST_ITEM_RE.match(str(line or "")))


def is_heading(line: str) -> bool:
    """识别可能的章节标题，同时避开较长的推荐正文。"""

    text = normalize_block_text(line)
    if not text or len(text) > 180:
        return False
    if REFERENCE_LIKE_RE.search(text):
        return False
    if MARKDOWN_HEADING_RE.match(str(line or "")):
        return True
    if NUMBERED_HEADING_RE.match(text):
        return True
    if text.endswith((".", ";", ",")) and len(text.split()) > 6:
        return False
    if RECOMMENDATION_HINT_RE.search(text) and len(text) > 80:
        return False
    if re.match(r"^(?:chapter|section|part|appendix)\s+\w+", text, re.I):
        return True
    if re.match(r"^[A-Z][A-Z0-9 /,&():'-]{3,}$", text) and len(text.split()) <= 14:
        return True
    return bool(HEADING_KEYWORDS_RE.search(text) and len(text.split()) <= 14)


# 标题层级推断目前采用通用规则，能覆盖多数指南。
# 如果后续接入新的指南来源，可以在这里补充来源专属的标题层级规则。
def heading_level(line: str) -> int:
    """根据 Markdown 标记、编号、章节词和大写样式推断标题层级。"""

    raw = str(line or "")
    markdown = MARKDOWN_HEADING_RE.match(raw)
    if markdown:
        return min(len(markdown.group(1)), 6)
    numbered = NUMBERED_HEADING_RE.match(normalize_block_text(raw))
    if numbered:
        return min(numbered.group(1).count(".") + 1, 6)
    if re.match(r"^(?:chapter|part)\s+\w+", raw, re.I):
        return 1
    if re.match(r"^[A-Z][A-Z0-9 /,&():'-]{3,}$", normalize_block_text(raw)):
        return 2
    return 3


# 维护章节路径栈，保证每个 SourceBlock 都能追溯到所在章节。
def update_section_path(section_stack: List[tuple[int, str]], heading: str, level: int) -> List[tuple[int, str]]:
    """遇到指定层级的新标题后，更新当前有效的章节栈。"""

    next_stack = [(existing_level, title) for existing_level, title in section_stack if existing_level < level]
    next_stack.append((level, heading))
    return next_stack


def section_path_text(section_stack: List[tuple[int, str]], title: str) -> List[str]:
    """返回当前可见章节路径；没有章节时回退到文档标题。"""

    path = [item_title for _, item_title in section_stack if item_title]
    return path or [title or "Document"]


# 判断章节是否位于参考文献、致谢、附录等低价值抽取区域。
def is_low_value_heading(heading: str) -> bool:
    """识别不适合作为推荐、证据候选来源的低价值标题。"""

    return bool(LOW_VALUE_HEADING_RE.search(str(heading or "")))

def is_low_value_section(section_stack: List[tuple[int, str]]) -> bool:
    """判断当前章节路径中是否包含低价值后置内容。"""

    return any(is_low_value_heading(title) for _, title in section_stack)


# 判断文本块是否像参考文献或作者单位信息，避免误送入推荐抽取。
def is_reference_like_block(text: str) -> bool:
    """识别不应抽取的参考文献块或作者单位块。"""

    clean = normalize_block_text(text)
    if not clean:
        return False
    reference_hits = len(REFERENCE_LIKE_RE.findall(clean))
    if reference_hits >= 2:
        return True
    if reference_hits and re.match(r"^\s*\d+[\).]?\s+[A-Z][A-Za-z-]+", clean):
        return True
    if AFFILIATION_LIKE_RE.search(clean) and len(clean) > 160:
        return True
    return False


# 给 SourceBlock 打上候选任务标签；这里只做粗召回，后续 router 再做质量过滤。
def candidate_hints(text: str, skip_candidate_extraction: bool = False) -> List[str]:
    """根据文本内容生成粗粒度候选标签，供后续抽取路由使用。"""

    if skip_candidate_extraction:
        return []
    hints: List[str] = []
    if RECOMMENDATION_HINT_RE.search(text):
        hints.append("recommendation")
    if GRADE_HINT_RE.search(text):
        hints.append("grade")
    if PICO_HINT_RE.search(text):
        hints.append("pico")
    if EVIDENCE_HINT_RE.search(text):
        hints.append("evidence")
    return hints


# 汇总块级质量信号，供 router 判断是否跳过或降级处理。
def block_quality(text: str, hints: List[str], is_low_value: bool, skip_reason: str) -> JsonDict:
    """汇总块级质量信号，供 router 判断是否跳过或降级处理。"""

    return {
        "is_low_value": is_low_value,
        "skip_candidate_extraction": bool(skip_reason),
        "skip_reason": skip_reason,
        "hint_count": len(hints),
        "text_length": len(text),
    }


def _record_context(record: JsonDict) -> JsonDict:
    return record_quality_context(record)


def block_type_for(line: str, text: str) -> str:
    """把规范化后的行分类为标题、表格、列表项或普通段落。"""

    if is_heading(line):
        return "heading"
    if TABLE_HINT_RE.search(text):
        return "table"
    if is_list_item(line):
        return "list_item"
    return "paragraph"


def record_guideline_id(record: JsonDict) -> str:
    """只在记录语义允许时返回 guideline_id，避免论文记录误挂指南 ID。"""

    if str(record.get("record_type") or "") == "paper":
        return ""
    direct = record.get("direct_extraction") if isinstance(record.get("direct_extraction"), dict) else {}
    seed = record.get("guideline_seed") if isinstance(record.get("guideline_seed"), dict) else {}
    return str(record.get("guideline_id") or direct.get("guideline_id") or seed.get("guideline_id") or "")


def record_paper_id(record: JsonDict) -> str:
    """只在记录语义允许时返回 paper_id，避免指南记录误挂论文 ID。"""

    if str(record.get("record_type") or "") == "guideline":
        return ""
    direct = record.get("direct_extraction") if isinstance(record.get("direct_extraction"), dict) else {}
    seed = record.get("paper_seed") if isinstance(record.get("paper_seed"), dict) else {}
    return str(record.get("paper_id") or direct.get("paper_id") or seed.get("paper_id") or "")


# 识别一行里混在正文中的标题，尽量恢复 PDF 抽取丢失的 section 边界。
def split_inline_headings(line: str, start: int) -> List[tuple[str, int, int, bool]]:
    """从扁平化 PDF 文本中恢复行内章节标题。

    返回 ``(text, char_start, char_end, forced_heading)`` 元组。
    ``forced_heading`` 为真时，即使通用标题识别较保守，也会输出为标题块。
    """

    clean = normalize_block_text(line)
    matches = [
        match
        for match in INLINE_HEADING_RE.finditer(clean)
        if (match.start() == 0 or clean[match.start() - 1] in ".;: ")
        and match.group(1)[:1].isupper()
    ]
    should_split = len(clean) > 180 or len(matches) >= 2
    if matches and matches[0].start() == 0:
        tail = clean[matches[0].end() :].strip(" :-")
        should_split = should_split or bool(tail and not INLINE_HEADING_RE.fullmatch(tail))
    if not matches or not should_split:
        return [(clean, start, start + len(clean), False)] if clean else []

    pieces: List[tuple[str, int, int, bool]] = []
    cursor = 0
    for index, match in enumerate(matches):
        if match.start() > cursor:
            body = clean[cursor : match.start()].strip()
            if body:
                body_start = start + clean.find(body, cursor)
                pieces.append((body, body_start, body_start + len(body), False))

        heading = match.group(1).strip()
        heading_start = start + match.start()
        pieces.append((heading, heading_start, heading_start + len(heading), True))
        cursor = match.end()

        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(clean)
        body = clean[cursor:next_start].strip(" :-")
        if body:
            body_start = start + clean.find(body, cursor)
            pieces.append((body, body_start, body_start + len(body), False))
        cursor = next_start

    return pieces or [(clean, start, start + len(clean), False)]


# 遍历正文行并保留原始字符偏移，保证下游结果可以追溯回原文。
def iter_lines_with_offsets(text: str) -> Iterable[tuple[str, int, int, bool]]:
    """产出规范化后的行片段，同时保留原始起止偏移。"""

    cursor = 0
    for raw_line in str(text or "").splitlines():
        start = text.find(raw_line, cursor)
        if start < 0:
            start = cursor
        end = start + len(raw_line)
        cursor = end + 1
        line = normalize_block_text(raw_line)
        if line:
            yield from split_inline_headings(line, start)


# 对于过长句子的切分以及兜底措施。
# 优先按句子边界拆长段；拆不动的超长片段会交给 split_oversized_text 保底。
def split_long_paragraph(text: str, max_chars: int) -> List[str]:
    """优先按近似句子边界切分长段落，再交给硬切分兜底。"""

    text = normalize_block_text(text)
    if len(text) <= max_chars:
        return [text] if text else []

    parts = [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]
    chunks: List[str] = []
    current: List[str] = []
    for part in parts or [text]:
        if len(part) > max_chars:
            if current:
                chunks.append(" ".join(current).strip())
                current = []
            chunks.extend(split_oversized_text(part, max_chars))
            continue
        candidate = " ".join([*current, part]).strip()
        if current and len(candidate) > max_chars:
            chunks.append(" ".join(current).strip())
            current = [part]
        else:
            current.append(part)
    if current:
        chunks.append(" ".join(current).strip())
    return chunks

# 保底按词数切分超大文本，避免单个 SourceBlock 过长拖垮后续抽取。
def split_oversized_text(text: str, max_chars: int) -> List[str]:
    """当句子级切分仍过长时，按词硬切分超大文本。"""

    words = text.split()
    chunks: List[str] = []
    current: List[str] = []
    for word in words:
        candidate = " ".join([*current, word]).strip()
        if current and len(candidate) > max_chars:
            chunks.append(" ".join(current).strip())
            current = [word]
        else:
            current.append(word)
    if current:
        chunks.append(" ".join(current).strip())
    return chunks


# 构造带来源、章节路径、候选标签和质量信息的 SourceBlock。
def build_block(spec: BlockBuildInput) -> JsonDict:
    """生成可追溯的 SourceBlock，并附加下游抽取所需的上下文。"""

    record = spec.record
    text = spec.text
    record_id = str(record.get("record_id") or "")
    block_id = stable_id("block", record_id, spec.order, spec.char_start, spec.char_end, text[:80])
    reference_like = is_reference_like_block(text)
    skip_reason = ""
    if spec.is_low_value:
        skip_reason = "low_value_section"
    elif reference_like:
        skip_reason = "reference_like_block"
    hints = [] if spec.forced_heading else candidate_hints(text, skip_candidate_extraction=bool(skip_reason))
    record_context = _record_context(record)
    quality = block_quality(text, hints, spec.is_low_value or reference_like, skip_reason)
    quality.update(record_context)
    return {
        "block_id": block_id,
        "record_id": record_id,
        "guideline_id": record_guideline_id(record),
        "paper_id": record_paper_id(record),
        "source": str(record.get("source") or ""),
        "title": str(record.get("title") or ""),
        "section_path": spec.section_path,
        "heading": spec.heading,
        "block_type": spec.block_type,
        "text": text,
        "order": spec.order,
        "char_start": spec.char_start,
        "char_end": spec.char_end,
        "token_estimate": estimate_tokens(text),
        "candidate_hints": hints,
        "quality": quality,
        "metadata": {
            "parent_line_type": spec.parent_line_type or spec.block_type,
            "source_url": record.get("url", ""),
            "raw_pdf_path": record.get("raw_pdf_path", ""),
            "reference_like": reference_like,
            "forced_heading": spec.forced_heading,
            **record_context,
        },
    }


def _append_heading_block(record: JsonDict, state: ParseState, title: str, line: str, start: int, end: int) -> None:
    """追加标题块，并同步更新当前章节状态。"""

    level = heading_level(line)
    state.current_heading = line.strip(" :")
    state.section_stack = update_section_path(state.section_stack, state.current_heading, level)
    state.blocks.append(
        build_block(
            BlockBuildInput(
                record=record,
                text=state.current_heading,
                block_type="heading",
                section_path=section_path_text(state.section_stack, title),
                heading=state.current_heading,
                order=state.order,
                char_start=start,
                char_end=end,
                is_low_value=is_low_value_section(state.section_stack),
                forced_heading=True,
            )
        )
    )
    state.order += 1


def _append_text_blocks(
    record: JsonDict,
    state: ParseState,
    title: str,
    content: str,
    line: str,
    start: int,
    max_block_chars: int,
) -> None:
    """为一行正文追加一个或多个文本块，必要时拆分超长段落。"""

    line_type = block_type_for(line, line)
    section_path = section_path_text(state.section_stack, title)
    low_value = is_low_value_section(state.section_stack)
    for piece in split_long_paragraph(line, max_block_chars):
        piece_start = content.find(piece, start)
        if piece_start < 0:
            piece_start = start
        piece_end = piece_start + len(piece)
        state.blocks.append(
            build_block(
                BlockBuildInput(
                    record=record,
                    text=piece,
                    block_type=line_type,
                    section_path=section_path,
                    heading=state.current_heading,
                    order=state.order,
                    char_start=piece_start,
                    char_end=piece_end,
                    parent_line_type=line_type,
                    is_low_value=low_value,
                )
            )
        )
        state.order += 1


# 将 cleaned record 转成 SourceBlock 序列，并维护标题栈、偏移量和顺序号。
def parse_source_record(record: JsonDict, max_block_chars: int = 1600) -> List[JsonDict]:
    """把一条清洗后的记录切成有序、可追溯的 SourceBlock。"""

    content = str(record.get("content") or record.get("content_markdown") or record.get("abstract") or "")
    title = str(record.get("title") or "Document")
    state = ParseState(section_stack=[(0, title)], current_heading=title, blocks=[])

    for line, start, end, forced_heading in iter_lines_with_offsets(content):
        if forced_heading or is_heading(line):
            _append_heading_block(record, state, title, line, start, end)
            continue

        _append_text_blocks(record, state, title, content, line, start, max_block_chars)

    return state.blocks


def parse_records(records: Iterable[JsonDict], max_block_chars: int = 1600) -> Iterable[JsonDict]:
    """遍历清洗记录集合，并逐条产出 SourceBlock。"""

    for record in records:
        yield from parse_source_record(record, max_block_chars=max_block_chars)


def parse_file(input_path: str | Path, output_path: str | Path, max_block_chars: int = 1600) -> int:
    """把 cleaned-record JSONL 文件切分为 SourceBlock JSONL 文件。"""

    return write_jsonl(output_path, parse_records(iter_jsonl(input_path), max_block_chars=max_block_chars))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split cleaned SourceRecord JSONL into traceable SourceBlock JSONL.")
    parser.add_argument("--input", required=True, help="Input cleaned SourceRecord JSONL path.")
    parser.add_argument("--output", required=True, help="Output SourceBlock JSONL path.")
    parser.add_argument("--max-block-chars", type=int, default=1600, help="Maximum characters per paragraph block.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = parse_file(args.input, args.output, max_block_chars=args.max_block_chars)
    print(f"Wrote {count} source blocks to {args.output}")


if __name__ == "__main__":
    main()
