"""Semantic chunking for guideline retrieval.

The splitter keeps high-value clinical statements away from background and
methodology prose, then packs same-type units into retrieval chunks of about
300 tokens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from src.retrieval.document_repr.section_classifier import classify_section


TARGET_TOKENS = 300
MIN_TOKENS = 120
MAX_TOKENS = 420
OVERLAP_TOKENS = 40

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?|[\u4e00-\u9fff]")
WORD_OR_CHAR_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?|\S")
LEADING_HEADING_RE = re.compile(r"^\s*#{1,6}\s+.+?(?:\n+|$)")
BULLET_RE = re.compile(r"^\s*(?:[-*+•]|\d+[.)]|\([a-zA-Z0-9]+\))\s+")
TABLE_RE = re.compile(r"^\s*(?:\|.*\||(?:table|fig(?:ure)?|algorithm)\s+\d*[:.\s].*|\u8868\s*\d*[:：.\s].*)", re.I)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;。！？；])\s+|(?<=[。！？；])")

ACTION_PATTERN = re.compile(
    r"\b(should|recommend|recommended|suggest|suggests|offer|avoid|consider|"
    r"is indicated|are indicated|may be used|should not|do not)\b|"
    r"(\u63a8\u8350|\u5efa\u8bae|\u5e94\u8be5|\u5e94|\u5b9c|\u4e0d\u5efa\u8bae)",
    re.I,
)
BAD_RECOMMENDATION_CONTEXT = re.compile(
    r"\b(return to recommendations|future research|recommendation for research|grade for quality|"
    r"wording evidence|not enough evidence to make|committee made a recommendation|"
    r"we recommend[.]{3}|we suggest[.]{3})\b|"
    r"(\u672a\u6765\u7814\u7a76|\u65b9\u6cd5\u5b66|\u8bc1\u636e\u5206\u7ea7)",
    re.I,
)
PICO_PATTERN = re.compile(
    r"(\bPICO\b|clinical question|key question|population|intervention|comparator|outcome|"
    r"\u4e34\u5e8a\u95ee\u9898|\u5173\u952e\u95ee\u9898|\u4eba\u7fa4|\u5e72\u9884|\u5bf9\u7167|\u7ed3\u5c40)",
    re.I,
)
EVIDENCE_PATTERN = re.compile(
    r"(evidence|certainty|quality of evidence|systematic review|meta-analysis|GRADE|"
    r"\u8bc1\u636e|\u8bc1\u636e\u8d28\u91cf|\u8bc1\u636e\u7b49\u7ea7|\u7cfb\u7edf\u8bc4\u4ef7|\u835f\u8403)",
    re.I,
)
SCOPE_PATTERN = re.compile(
    r"(scope|target population|intended audience|applicability|eligib|"
    r"\u8303\u56f4|\u9002\u7528|\u76ee\u6807\u4eba\u7fa4|\u9002\u7528\u5bf9\u8c61)",
    re.I,
)
BACKGROUND_PATTERN = re.compile(r"(background|introduction|epidemiology|\u80cc\u666f|\u524d\u8a00|\u5f15\u8a00)", re.I)


@dataclass(frozen=True)
class SemanticChunkPart:
    content: str
    chunk_type: str
    token_count: int


@dataclass(frozen=True)
class SemanticUnit:
    text: str
    unit_type: str
    token_count: int


def estimate_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text or ""))


def retrieval_text(title: str, section_path: Iterable[str], chunk_type: str, content: str) -> str:
    path_text = " > ".join(str(item) for item in section_path if item)
    labels = [title, path_text, f"chunk_type: {chunk_type}", content]
    return "\n".join(item for item in labels if item)


def split_section_content(
    content: str,
    min_tokens: int = MIN_TOKENS,
    target_tokens: int = TARGET_TOKENS,
    max_tokens: int = MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    section_type: str = "other",
) -> list[str]:
    return [
        part.content
        for part in split_section_semantic_content(
            content,
            section_type=section_type,
            min_tokens=min_tokens,
            target_tokens=target_tokens,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
    ]


def split_section_semantic_content(
    content: str,
    *,
    heading: str = "",
    section_path: Iterable[str] | None = None,
    section_type: str | None = None,
    min_tokens: int = MIN_TOKENS,
    target_tokens: int = TARGET_TOKENS,
    max_tokens: int = MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[SemanticChunkPart]:
    cleaned = helper_strip_leading_heading(content)
    inferred_type = section_type or classify_section(heading=heading, section_path=section_path or [], content=cleaned)
    units = helper_semantic_units(cleaned, inferred_type)
    return helper_pack_units(units, min_tokens, target_tokens, max_tokens, overlap_tokens)


def classify_semantic_text(text: str, section_type: str = "other") -> str:
    normalized = helper_normalize(text)
    if not normalized:
        return section_type
    if section_type == "reference":
        return "reference"
    if helper_is_table_text(normalized):
        return "table"
    if ACTION_PATTERN.search(normalized) and not BAD_RECOMMENDATION_CONTEXT.search(normalized):
        return "recommendation"
    if PICO_PATTERN.search(normalized):
        return "pico"
    if EVIDENCE_PATTERN.search(normalized):
        return "evidence"
    if SCOPE_PATTERN.search(normalized):
        return "scope_population"
    if section_type in {"recommendation", "pico", "evidence", "scope_population", "table", "conclusion", "background"}:
        return section_type
    if BACKGROUND_PATTERN.search(normalized):
        return "background"
    return "other"


def helper_strip_leading_heading(content: str) -> str:
    return LEADING_HEADING_RE.sub("", content or "", count=1).strip()


def helper_normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def helper_is_table_text(text: str) -> bool:
    lines = [line for line in (text or "").splitlines() if line.strip()]
    return bool(lines) and sum(1 for line in lines if TABLE_RE.search(line)) >= max(1, len(lines) // 2)


def helper_semantic_units(content: str, section_type: str) -> list[SemanticUnit]:
    units: list[SemanticUnit] = []
    for block in helper_raw_blocks(content):
        block_type = classify_semantic_text(block, section_type)
        for text in helper_split_block(block, block_type):
            token_count = estimate_tokens(text)
            if token_count:
                units.append(SemanticUnit(text=text, unit_type=classify_semantic_text(text, block_type), token_count=token_count))
    return units


def helper_raw_blocks(content: str) -> list[str]:
    blocks: list[str] = []
    table_lines: list[str] = []
    paragraph_lines: list[str] = []

    def flush_table() -> None:
        if table_lines:
            blocks.append("\n".join(table_lines).strip())
            table_lines.clear()

    def flush_paragraph() -> None:
        if paragraph_lines:
            blocks.extend(helper_split_paragraph_lines(paragraph_lines))
            paragraph_lines.clear()

    for line in (content or "").splitlines():
        stripped = line.strip()
        if not stripped:
            flush_table()
            flush_paragraph()
            continue
        if TABLE_RE.search(stripped):
            flush_paragraph()
            table_lines.append(stripped)
            continue
        flush_table()
        paragraph_lines.append(stripped)
    flush_table()
    flush_paragraph()
    return [block for block in blocks if block]


def helper_split_paragraph_lines(lines: list[str]) -> list[str]:
    if len(lines) > 1 and all(BULLET_RE.search(line) for line in lines):
        return [BULLET_RE.sub("", line).strip() for line in lines if line.strip()]
    return [" ".join(lines).strip()]


def helper_split_block(block: str, block_type: str) -> list[str]:
    if block_type == "table":
        return [block]
    if estimate_tokens(block) <= TARGET_TOKENS:
        return [block]
    sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(block) if part.strip()]
    return sentences or [block]


def helper_pack_units(
    units: list[SemanticUnit],
    min_tokens: int,
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[SemanticChunkPart]:
    chunks: list[SemanticChunkPart] = []
    current: list[SemanticUnit] = []
    current_tokens = 0
    current_type = ""

    def flush() -> None:
        nonlocal current, current_tokens, current_type
        if current:
            chunks.append(helper_chunk_part(current))
        current = []
        current_tokens = 0
        current_type = ""

    for unit in units:
        if unit.token_count > max_tokens:
            # 超长单元无法与相邻语义安全合并，先落盘当前块再单独滑窗切分。
            flush()
            chunks.extend(helper_split_long_unit(unit, max_tokens, overlap_tokens))
            continue
        same_type = not current_type or current_type == unit.unit_type
        would_fit = current_tokens + unit.token_count <= max_tokens
        should_pack = current_tokens < target_tokens or current_tokens < min_tokens
        # 类型变化、超过硬上限或已达到目标长度时结束当前 chunk。
        if current and (not same_type or not would_fit or not should_pack):
            flush()
        current.append(unit)
        current_type = unit.unit_type
        current_tokens += unit.token_count
    flush()
    return chunks


def helper_chunk_part(units: list[SemanticUnit]) -> SemanticChunkPart:
    text = "\n\n".join(unit.text for unit in units if unit.text).strip()
    chunk_type = units[0].unit_type if units else "other"
    return SemanticChunkPart(content=text, chunk_type=chunk_type, token_count=estimate_tokens(text))


def helper_split_long_unit(unit: SemanticUnit, max_tokens: int, overlap_tokens: int) -> list[SemanticChunkPart]:
    overlap = 0 if unit.unit_type in {"recommendation", "table"} else overlap_tokens
    tokens = WORD_OR_CHAR_RE.findall(unit.text)
    if not tokens:
        return []
    parts = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        text = helper_join_tokens(tokens[start:end])
        token_count = estimate_tokens(text)
        if token_count:
            parts.append(SemanticChunkPart(content=text, chunk_type=unit.unit_type, token_count=token_count))
        if end >= len(tokens):
            break
        start = max(start + 1, end - overlap)
    return parts


def helper_join_tokens(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?，。；：！？%)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    return text.strip()
