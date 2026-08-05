# 面向指南检索的语义切块器。
#
# 设计原则：
# - 把"高价值临床陈述"（推荐、PICO、证据等级、范围）与"背景/方法学文本"分开；
# - 把同类型 unit 打包成约 300 token 的检索 chunk（min 120 / max 420 / overlap 40）；
# - 切分粒度由 token 预算控制，避免单 chunk 过长稀释相关性。
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
from src.utils.text import (
    estimate_tokens,
    normalize_space as helper_normalize,
)


# Token 预算常量：
# - TARGET_TOKENS=300：每个 chunk 的目标 token 数，控制 BM25/向量召回的精度
# - MIN_TOKENS=120：避免碎块；不足时继续打包直到达标
# - MAX_TOKENS=420：硬上限，超过则触发长单元滑窗切分
# - OVERLAP_TOKENS=40：相邻 chunk 重叠 token 数（除 recommendation/table 外）
#   注意：recommendation/table 不做 overlap，因为重排会破坏列表/表格完整性
TARGET_TOKENS = 300
MIN_TOKENS = 120
MAX_TOKENS = 420
OVERLAP_TOKENS = 40

# 单词或单字符：英文按单词切，中文按字符切，简化后续 token 估算与滑窗对齐。
WORD_OR_CHAR_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?|\S")
LEADING_HEADING_RE = re.compile(r"^\s*#{1,6}\s+.+?(?:\n+|$)")

# 列表项前缀：英文 -/*/+、数字编号、中文 •。
BULLET_RE = re.compile(r"^\s*(?:[-*+•]|\d+[.)]|\([a-zA-Z0-9]+\))\s+")

# 表格行识别：含 '|' 或 "Table N"/"图 N"/"表 N" 起头。
TABLE_RE = re.compile(r"^\s*(?:\|.*\||(?:table|fig(?:ure)?|algorithm)\s+\d*[:.\s].*|\u8868\s*\d*[:：.\s].*)", re.I)

# 句子边界：中英文常见句末标点 + 空白。
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;。！？；])\s+|(?<=[。！？；])")

# 推荐/行动语句：英文 should/recommend/suggest + 中文 推荐/建议/应该 等。
# 注意：这些词可能在背景段（如"建议未来研究"）出现，需配合 BAD_RECOMMENDATION_CONTEXT 排除。
ACTION_PATTERN = re.compile(
    r"\b(should|recommend|recommended|suggest|suggests|offer|avoid|consider|"
    r"is indicated|are indicated|may be used|should not|do not)\b|"
    r"(\u63a8\u8350|\u5efa\u8bae|\u5e94\u8be5|\u5e94|\u5b9c|\u4e0d\u5efa\u8bae)",
    re.I,
)

# 误判上下文：未来研究、方法学、证据分级讨论等会让 ACTION_PATTERN 误命中，
# 在这些上下文中即使有"recommend"也归类为非推荐段。
BAD_RECOMMENDATION_CONTEXT = re.compile(
    r"\b(return to recommendations|future research|recommendation for research|grade for quality|"
    r"wording evidence|not enough evidence to make|committee made a recommendation|"
    r"we recommend[.]{3}|we suggest[.]{3})\b|"
    r"(\u672a\u6765\u7814\u7a76|\u65b9\u6cd5\u5b66|\u8bc1\u636e\u5206\u7ea7)",
    re.I,
)

# PICO / 关键问题识别：英文 clinical question + 中文 临床问题/关键问题/人群/干预等。
PICO_PATTERN = re.compile(
    r"(\bPICO\b|clinical question|key question|population|intervention|comparator|outcome|"
    r"\u4e34\u5e8a\u95ee\u9898|\u5173\u952e\u95ee\u9898|\u4eba\u7fa4|\u5e72\u9884|\u5bf9\u7167|\u7ed3\u5c40)",
    re.I,
)

# 证据等级：GRADE / 系统评价 / meta 分析等。
EVIDENCE_PATTERN = re.compile(
    r"(evidence|certainty|quality of evidence|systematic review|meta-analysis|GRADE|"
    r"\u8bc1\u636e|\u8bc1\u636e\u8d28\u91cf|\u8bc1\u636e\u7b49\u7ea7|\u7cfb\u7edf\u8bc4\u4ef7|\u835f\u8403)",
    re.I,
)

# 范围/适用人群：英文 scope / applicability + 中文 范围/适用。
SCOPE_PATTERN = re.compile(
    r"(scope|target population|intended audience|applicability|eligib|"
    r"\u8303\u56f4|\u9002\u7528|\u76ee\u6807\u4eba\u7fa4|\u9002\u7528\u5bf9\u8c61)",
    re.I,
)

# 背景段：introduction / background / 中文 背景/前言。
BACKGROUND_PATTERN = re.compile(r"(background|introduction|epidemiology|\u80cc\u666f|\u524d\u8a8b|\u5f15\u8a00)", re.I)


@dataclass(frozen=True)
class SemanticChunkPart:
    """最终产出的小 chunk：content / chunk_type / token_count。"""
    content: str
    chunk_type: str
    token_count: int


@dataclass(frozen=True)
class SemanticUnit:
    """切分阶段的最小语义单位：text / unit_type / token_count。"""
    text: str
    unit_type: str
    token_count: int




def retrieval_text(title: str, section_path: Iterable[str], chunk_type: str, content: str) -> str:
    """构造送入 BM25/向量召回的"检索口径文本"。

    拼接顺序：title → section_path → chunk_type 标签 → content。
    检索时把元数据放在正文之前，便于关键词命中标题与章节。
    """
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
    """只返回字符串列表的便捷接口（与 split_section_semantic_content 同源）。"""
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
    """对单 section 做语义切块：去标题 → 推类型 → 拆 unit → 打包。

    流程：
    1. helper_strip_leading_heading 去掉开头标题行；
    2. 推断 section_type（未传时调 classify_section）；
    3. helper_semantic_units 把 section 拆成最小语义 unit；
    4. helper_pack_units 按 token 预算打包成最终 chunk。
    """
    cleaned = helper_strip_leading_heading(content)
    inferred_type = section_type or classify_section(heading=heading, section_path=section_path or [], content=cleaned)
    units = helper_semantic_units(cleaned, inferred_type)
    return helper_pack_units(units, min_tokens, target_tokens, max_tokens, overlap_tokens)


def classify_semantic_text(text: str, section_type: str = "other") -> str:
    """对单段文本按内容信号分类（recommendation / pico / evidence / table …）。

    优先级（从高到低）：
    1. reference：section_type 已是 reference（来自 encoder.helper_is_reference_section）
    2. table：过半行匹配表格格式
    3. recommendation：含 ACTION_PATTERN 且不在 BAD_RECOMMENDATION_CONTEXT
    4. pico：含 PICO_PATTERN
    5. evidence：含 EVIDENCE_PATTERN
    6. scope_population：含 SCOPE_PATTERN
    7. 继承 section_type（若已是枚举值）
    8. background：含 BACKGROUND_PATTERN
    9. other：兜底
    """
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
    """去掉内容开头的 Markdown 标题行（避免被吞进 chunk 重复出现）。"""
    return LEADING_HEADING_RE.sub("", content or "", count=1).strip()




def helper_is_table_text(text: str) -> bool:
    """判定一段文本是否主要是表格行：过半非空行匹配 TABLE_RE。"""
    lines = [line for line in (text or "").splitlines() if line.strip()]
    return bool(lines) and sum(1 for line in lines if TABLE_RE.search(line)) >= max(1, len(lines) // 2)


def helper_semantic_units(content: str, section_type: str) -> list[SemanticUnit]:
    """拆成最小语义 unit 列表。

    对每个 raw_block 先按整体类型分类，再细切（表格整体保留、其它按句号切）。
    重复调 classify_semantic_text 保证段落 vs 句子级别分类一致。
    """
    units: list[SemanticUnit] = []
    for block in helper_raw_blocks(content):
        block_type = classify_semantic_text(block, section_type)
        for text in helper_split_block(block, block_type):
            token_count = estimate_tokens(text)
            if token_count:
                units.append(SemanticUnit(text=text, unit_type=classify_semantic_text(text, block_type), token_count=token_count))
    return units


def helper_raw_blocks(content: str) -> list[str]:
    """按表格 vs 段落两路收集 raw_block，列表项保持单独成段。

    flush_table / flush_paragraph 闭包分别把累积的行合并成 block，
    表格保持多行结构，段落则按列表/普通段落再细分（见 helper_split_paragraph_lines）。
    """
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
    """段落细分：列表项各自成段，普通段落合并为单段。

    列表判断：所有行都匹配 BULLET_RE 时认为是一个列表，逐项拆出；
    否则合并为单段，避免过度切碎。
    """
    if len(lines) > 1 and all(BULLET_RE.search(line) for line in lines):
        return [BULLET_RE.sub("", line).strip() for line in lines if line.strip()]
    return [" ".join(lines).strip()]


def helper_split_block(block: str, block_type: str) -> list[str]:
    """单 block 内部切分。

    规则：
    - 表格保留整块（不可按句切，破坏表格）；
    - 不超 TARGET_TOKENS 直接保留；
    - 否则按 SENTENCE_SPLIT_RE 切句。
    """
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
    """按 token 预算 + 类型一致性打包 unit 成最终 chunk。

    打包策略：
    - 超长 unit (> max_tokens) 走 helper_split_long_unit 滑窗切分；
    - 同类型 unit 优先合并（保证 chunk 内信号一致）；
    - 达到 target 或硬超 max 时 flush；
    - 未达 min_tokens 时继续吸（pack 直至达标）。
    """
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
        # 还在努力攒够 min_tokens 时不主动 flush；已达到 target 后遇类型变化或超 max 才 flush。
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
    """把当前缓冲的 unit 列表合成一个 SemanticChunkPart。"""
    text = "\n\n".join(unit.text for unit in units if unit.text).strip()
    chunk_type = units[0].unit_type if units else "other"
    return SemanticChunkPart(content=text, chunk_type=chunk_type, token_count=estimate_tokens(text))


def helper_split_long_unit(unit: SemanticUnit, max_tokens: int, overlap_tokens: int) -> list[SemanticChunkPart]:
    """超长 unit 滑窗切分，保留 overlap 以维持跨 chunk 上下文。

    注意：recommendation / table 不做 overlap，因为重排会破坏列表/表格完整性。
    """
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
    """还原 token 列表为可读文本，清理标点周围的冗余空格。"""
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?，。；：！？%)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    return text.strip()