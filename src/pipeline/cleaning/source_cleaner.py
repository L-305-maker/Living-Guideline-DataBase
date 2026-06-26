from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

from src.common.extraction_common import JsonDict
from src.common.process_jsonl import iter_jsonl, write_jsonl
from src.domain.common import stable_id
from src.pipeline.cleaning.offset_mapping import OFFSET_MAPPING_VERSION
from src.pipeline.cleaning.pdf_noise import enhance_pdf_noise_cleaning
from src.pipeline.cleaning.source_normalizers import add_direct_schema_seeds, base_record, normalize_source
from src.pipeline.extraction.recommendation.candidate_extractor import (
    infer_certainty,
    infer_direction,
    infer_strength,
    split_recommendation_statements,
)


# 模块职责：
# 将 crawler/origin 阶段产生的来源记录规范化为统一 cleaned record。本阶段只做轻量、
# 可追溯、非破坏性的清洗；是否进入解析、修复或跳过由 quality_gate.py 决定。
REFERENCE_HEADING_RE = re.compile(r"(?im)^\s*(references|bibliography)\s*$")
INLINE_REFERENCE_HEADING_RE = re.compile(
    r"(?ims)(^|\n|\.\s+)(references|bibliography)(?=\s*(?:$|\n|\d+\.|\[\d+\]|[A-Z][A-Za-z-]+,))"
)
REFERENCE_ENTRY_RE = re.compile(r"(?m)(?:^|\n)\s*(?:\d{1,3}\.|\[\d{1,3}\])\s+\S+")
REFERENCE_MARKER_RE = re.compile(r"(?:^|\s)(?:\d{1,3}[\.)]|\[\d{1,3}\])\s+\S+")
REFERENCE_CITATION_RE = re.compile(r"(?i)\b(?:et\s+al\.?|doi:?|[.]\s+[A-Z][A-Za-z]+[.])\b|\b(?:19|20)\d{2}\b")
POST_REFERENCE_BODY_RE = re.compile(
    r"(?i)\b(?:chapter\s+\d+|overall\s+key\s+points|key\s+points|executive\s+summary|"
    r"recommendations?|background|introduction|methods?)\b"
)
AFFILIATION_HEADING_RE = re.compile(
    r"(?im)^\s*(authors?\s+and\s+affiliations|author\s+information|affiliations|extended\s+author\s+information)\s*$"
)
INLINE_AFFILIATION_HEADING_RE = re.compile(
    r"(?ims)(^|\n|\.\s+)(authors?\s+and\s+affiliations|author\s+information|affiliations|extended\s+author\s+information)(?=\s*(?:$|\n|department|division|university|school|hospital|medical))"
)
TABLE_START_RE = re.compile(r"(?im)^\s*(table\s+(\d+[A-Za-z]?)[^\n]*)$")
TABLE_CAPTION_RE = re.compile(r"(?im)^\s*((?:table|figure)\s+\d+[A-Za-z]?(?:[.:)\-]\s*)?[^\n]{0,180})$")
INLINE_TABLE_CAPTION_RE = re.compile(
    r"(?P<prefix>^|[.;:]\s+)(?P<caption>(?:Table|Figure)\s+\d+[A-Za-z]?(?:[.:)\-]\s*)?)(?P<tail>\s+(?=[A-Z0-9]))",
    re.I,
)
SECTION_HEADING_RE = re.compile(
    r"(?im)^\s*(abstract|introduction|methods?|methodology|recommendations?|discussion|limitations?|conclusions?|background|evidence|rationale|remarks?)\s*$"
)
INLINE_SECTION_HEADING_RE = re.compile(
    r"(?P<prefix>^|[.;:]\s+)"
    r"(?P<heading>Abstract|Introduction|Methods?|Methodology|Recommendations?|Discussion|Limitations?|Conclusions?|"
    r"Background|Evidence|Rationale|Remarks?)"
    r"(?!\s*\d)"
    r"(?P<tail>\s*(?:[:\-]\s*)?(?=[A-Z0-9]))"
)
INLINE_RECOMMENDATION_LABEL_RE = re.compile(
    r"(?P<prefix>[.;:]\s+)"
    r"(?P<label>(?:Recommendation|Statement|Clinical question|Question)\s*\d+[A-Za-z]?(?:\.\d+)*\s*[:.)-]?)"
    r"(?P<tail>\s+(?=[A-Z]))",
    re.I,
)
INLINE_NUMBERED_RECOMMENDATION_RE = re.compile(
    r"(?P<prefix>[.;:]\s+)"
    r"(?P<label>\d+(?:\.\d+)*[.)]\s+)"
    r"(?P<tail>(?=(?:We|The guideline|Clinicians|Patients|Adults|Children|For|In)\b))",
    re.I,
)
TWO_COLUMN_GAP_RE = re.compile(r"[ \t]{4,}")
RECOMMENDATION_BOX_HEADING_RE = re.compile(
    r"(?im)^\s*((?:box\s+\d+[A-Za-z]?|recommendation\s+box(?:\s+\d+[A-Za-z]?)?|"
    r"key\s+recommendations?|summary\s+of\s+recommendations?)"
    r"(?:[.:)\-]\s*)?[^\n]{0,160})$"
)
INLINE_RECOMMENDATION_BOX_RE = re.compile(
    r"(?P<prefix>^|[.;:]\s+)"
    r"(?P<label>(?:Box\s+\d+[A-Za-z]?|Key recommendations?|Recommendations?)\s*[:.\-]?)"
    r"(?P<tail>\s+(?=(?:We|In|For|Adults|Children|Patients|Clinicians|The)\b))",
    re.I,
)
METHOD_SECTION_NAME_RE = re.compile(r"(?i)^(methods?|methodology)$")
METHODOLOGY_BOILERPLATE_RE = re.compile(
    r"(?i)\b(?:task force|committee|conflict of interest|conflicts of interest|literature search|"
    r"systematic review of the literature|evidence grading system|levels? of recommendations|"
    r"rand/ucla|methodology|search strategy|included studies|excluded studies)\b"
)
SOURCE_SPAN_BOUNDARY_RE = re.compile(
    r"(?im)^\s*(?:abstract|introduction|methods?|methodology|recommendations?|background|evidence|rationale|"
    r"remarks?|table\s+\d+[A-Za-z]?|figure\s+\d+[A-Za-z]?|box\s+\d+[A-Za-z]?)\b"
)
MAX_RECOMMENDATION_BOX_CHARS = 8000
MAX_REFERENCE_REMOVED_RATIO = 0.35
PDF_COLUMN_MERGE_REPAIRS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(?i)\b(use\s+)actigof\s+[^.]{0,220}?\s+raphy(\s+in\s+the\s+assessment\b)",
        ),
        r"\1actigraphy\2",
    ),
    (
        re.compile(
            r"(?i)\b(use\s+)actigintegrated\s+[^.]{0,260}?\s+raphy(\s+to\s+monitor\b)",
        ),
        r"\1actigraphy\2",
    ),
    (
        re.compile(
            r"(?i)\b(use\s+)actigraof\s+[^.]{0,260}?\s+phy(\s+to\s+estimate\b)",
        ),
        r"\1actigraphy\2",
    ),
    (
        re.compile(
            r"(?i)\b(use\s+)actigraof\s+[^.]{0,220}?\.\s+phy(\s+to\s+estimate\b)",
        ),
        r"\1actigraphy\2",
    ),
    (re.compile(r"(?i)\bsignifiand\b"), "significant and"),
    (re.compile(r"(?i)\bfindevaluation\b"), "findings for evaluation"),
    (re.compile(r"(?i)\bprotients\b"), "patients"),
    (re.compile(r"(?i)\bdisorlonged\b"), "disorders. Prolonged"),
    (re.compile(r"(?i)shouldconsider"), "should consider "),
    (re.compile(r"(?i)shouldbe"), "should be "),
    (re.compile(r"(?i)shouldnot"), "should not "),
    (re.compile(r"(?i)shouldreceive"), "should receive "),
    (re.compile(r"(?i)shouldundergo"), "should undergo "),
    (re.compile(r"(?i)shoulduse"), "should use "),
)
GRADE_CERTAINTY_RE = re.compile(
    r"\b(?:(very\s+low|low|moderate|high)\s+(?:certainty|quality)|(?:certainty|quality)(?:\s+of\s+evidence)?\s+(?:was|is|were|are)\s+(very\s+low|low|moderate|high))\b",
    re.I,
)
GRADE_REASON_MAP = {
    "risk_of_bias": re.compile(r"\brisk\s+of\s+bias\b", re.I),
    "inconsistency": re.compile(r"\binconsistency\b", re.I),
    "indirectness": re.compile(r"\bindirectness\b", re.I),
    "imprecision": re.compile(r"\bimprecision\b", re.I),
    "publication_bias": re.compile(r"\bpublication\s+bias\b", re.I),
}


def raw_content_from(record: JsonDict) -> str:
    """从来源记录中取出最接近原始文本的字段，作为 raw_content 保存。"""

    return str(record.get("content_markdown") or record.get("content") or record.get("abstract") or "")


def fix_hyphenation(text: str) -> str:
    """修复 PDF 抽取常见的行尾断词，例如 treat-\nment -> treatment。"""

    text = re.sub(r"([A-Za-z]{3,})-\s*\n\s*([A-Za-z]{2,})", r"\1\2", str(text or ""))
    return re.sub(r"([A-Za-z]{3,})-\s+([a-z]{2,})", r"\1\2", text)


def normalize_spaces(text: str) -> str:
    """压缩重复空白，同时尽量保留段落边界。"""

    lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in str(text or "").splitlines()]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def restore_two_column_reading_order(text: str) -> tuple[str, JsonDict]:
    """Reorder simple two-column rows into left-column then right-column reading order."""

    raw_lines = str(text or "").splitlines()
    candidate_rows: list[tuple[int, str, str]] = []
    for index, line in enumerate(raw_lines):
        if len(line) < 32:
            continue
        match = TWO_COLUMN_GAP_RE.search(line)
        if not match:
            continue
        left = line[: match.start()].strip()
        right = line[match.end() :].strip()
        if len(left) >= 8 and len(right) >= 8:
            candidate_rows.append((index, left, right))

    non_empty = [line for line in raw_lines if line.strip()]
    row_ratio = len(candidate_rows) / max(1, len(non_empty))
    if len(candidate_rows) < 3 or row_ratio < 0.35:
        return str(text or ""), {"applied": False, "two_column_rows": len(candidate_rows), "line_count": len(non_empty)}

    row_map = {index: (left, right) for index, left, right in candidate_rows}
    output: list[str] = []
    pending_left: list[str] = []
    pending_right: list[str] = []

    def flush_columns() -> None:
        if pending_left or pending_right:
            output.extend(pending_left)
            output.extend(pending_right)
            pending_left.clear()
            pending_right.clear()

    for index, line in enumerate(raw_lines):
        if index in row_map:
            left, right = row_map[index]
            pending_left.append(left)
            pending_right.append(right)
            continue
        if line.strip():
            flush_columns()
            output.append(line.strip())
        else:
            flush_columns()
            output.append("")

    flush_columns()
    return "\n".join(output), {
        "applied": True,
        "two_column_rows": len(candidate_rows),
        "line_count": len(non_empty),
        "row_ratio": round(row_ratio, 4),
    }


def remove_headers_footers(text: str) -> str:
    """删除重复页眉、页脚和页码行，尽量不触碰正文内容。"""

    lines = normalize_spaces(text).splitlines()
    counts = Counter(line for line in lines if line)
    kept: list[str] = []
    for line in lines:
        if re.fullmatch(r"(?i)(?:page\s*)?\d+\s+(?:of|/)\s+\d+|page\s+\d+|\d+", line):
            continue
        if counts[line] >= 3 and re.search(r"(?i)\b(journal|guideline|copyright|doi|page)\b", line):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def light_clean_pdf_text(text: str) -> str:
    """应用所有来源共享的最小 PDF 文本清理规则。"""

    return normalize_spaces(remove_headers_footers(fix_hyphenation(text)))


def light_clean_pdf_text_with_layout_report(text: str) -> tuple[str, JsonDict]:
    reordered, layout_report = restore_two_column_reading_order(str(text or ""))
    return light_clean_pdf_text(reordered), layout_report


def repair_pdf_column_word_merges(text: str) -> tuple[str, int]:
    repaired = str(text or "")
    total = 0
    for pattern, replacement in PDF_COLUMN_MERGE_REPAIRS:
        repaired, count = pattern.subn(replacement, repaired)
        total += count
    return repaired, total


def group_or_none(match: re.Match[str], name: str) -> str | None:
    value = match.groupdict().get(name)
    return value if value is not None else None


def restore_inline_section_breaks(text: str) -> tuple[str, int]:
    """为被压扁成一行的 PDF 文本恢复明显的内联章节和推荐条目换行。

    这个函数只处理行数极少的记录。多行记录通常已经带有布局信号；一行记录往往需要
    先恢复章节边界和连续推荐条目边界，后续 structure parser 才能重建 section path，
    recommendation extractor 也能避免把多条推荐粘成一个候选。
    """

    clean = str(text or "")
    non_empty_lines = [line for line in clean.splitlines() if line.strip()]
    if len(clean) < 180 or len(non_empty_lines) > 2:
        return clean, 0

    inserted = 0

    def replace_section(match: re.Match[str]) -> str:
        nonlocal inserted
        prefix = group_or_none(match, "prefix")
        heading = group_or_none(match, "heading")
        tail = group_or_none(match, "tail")
        if heading is None or tail is None or prefix is None:
            return match.group(0)
        if prefix == "":
            inserted += 1
            return f"{heading}\n{tail.lstrip()}"
        inserted += 1
        return f"{prefix.rstrip()}\n{heading}\n{tail.lstrip()}"

    def replace_recommendation(match: re.Match[str]) -> str:
        nonlocal inserted
        prefix = group_or_none(match, "prefix")
        label = group_or_none(match, "label")
        tail = group_or_none(match, "tail")
        if prefix is None or label is None or tail is None:
            return match.group(0)
        inserted += 1
        return f"{prefix.rstrip()}\n{label}{tail}"

    def replace_caption(match: re.Match[str]) -> str:
        nonlocal inserted
        prefix = group_or_none(match, "prefix")
        caption = group_or_none(match, "caption")
        tail = group_or_none(match, "tail")
        if prefix is None or caption is None or tail is None:
            return match.group(0)
        inserted += 1
        return f"{prefix.rstrip()}\n{caption}{tail}"

    restored = INLINE_SECTION_HEADING_RE.sub(replace_section, clean)
    restored = INLINE_TABLE_CAPTION_RE.sub(replace_caption, restored)
    restored = INLINE_RECOMMENDATION_BOX_RE.sub(replace_recommendation, restored)
    restored = INLINE_RECOMMENDATION_LABEL_RE.sub(replace_recommendation, restored)
    restored = INLINE_NUMBERED_RECOMMENDATION_RE.sub(replace_recommendation, restored)
    return normalize_spaces(restored), inserted


def reference_split_start(match: re.Match[str]) -> int:
    return match.start(2) if match.lastindex and match.lastindex >= 2 else match.start()


def reference_density(text: str) -> float:
    sample = str(text or "")[:4000]
    lines = [line.strip() for line in sample.splitlines() if line.strip()]
    if len(lines) >= 3:
        ref_like = 0
        for line in lines[:80]:
            if REFERENCE_MARKER_RE.search(line) or (
                re.search(r"\b(?:19|20)\d{2}\b", line) and REFERENCE_CITATION_RE.search(line)
            ):
                ref_like += 1
        return ref_like / max(1, min(len(lines), 80))

    markers = len(REFERENCE_MARKER_RE.findall(sample))
    citations = len(REFERENCE_CITATION_RE.findall(sample))
    return min(1.0, (markers + min(citations, 12) / 3) / 12)


def is_valid_reference_split(text: str, start: int) -> bool:
    tail = text[start:].strip()
    if not tail:
        return False

    position_ratio = start / max(1, len(text))
    removed_ratio = len(tail) / max(1, len(text))
    early_tail = tail[:20000]
    if position_ratio < 0.45 and POST_REFERENCE_BODY_RE.search(early_tail):
        return False
    if len(text) < 5000:
        return True
    if removed_ratio > MAX_REFERENCE_REMOVED_RATIO and position_ratio < 0.55:
        return False

    reference_entries = len(REFERENCE_ENTRY_RE.findall(tail[:12000]))
    density = reference_density(tail)
    if position_ratio >= 0.55:
        return reference_entries >= 1 or density >= 0.12 or len(tail) < 8000
    return (reference_entries >= 8 or density >= 0.35) and not POST_REFERENCE_BODY_RE.search(early_tail)


def split_references(text: str) -> tuple[str, str]:
    """拆出 References/Bibliography 之后的尾部文本，避免参考文献污染正文抽取。"""

    candidates = sorted(
        [*REFERENCE_HEADING_RE.finditer(text), *INLINE_REFERENCE_HEADING_RE.finditer(text)],
        key=reference_split_start,
    )
    for match in candidates:
        start = reference_split_start(match)
        if is_valid_reference_split(text, start):
            return text[:start].rstrip(), text[start:].strip()
    return text, ""


def split_affiliations(text: str) -> tuple[str, str]:
    """拆出作者单位信息，避免机构地址和作者说明被当成临床正文。"""

    match = AFFILIATION_HEADING_RE.search(text) or INLINE_AFFILIATION_HEADING_RE.search(text)
    if not match:
        return text, ""
    start = match.start(2) if match.lastindex and match.lastindex >= 2 else match.start()
    return text[:start].rstrip(), text[start:].strip()


def extract_tables_from_text(text: str) -> list[JsonDict]:
    """当上游没有结构化 tables 时，从正文 Table 标题粗提取表格片段作为兜底。"""

    matches = list(TABLE_CAPTION_RE.finditer(text))
    tables: list[JsonDict] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        following_boundaries = [
            boundary.start()
            for boundary in (
                SECTION_HEADING_RE.search(text, match.end(), end),
                RECOMMENDATION_BOX_HEADING_RE.search(text, match.end(), end),
            )
            if boundary
        ]
        if following_boundaries:
            end = min(following_boundaries)
        raw_text = text[start:end].strip()
        if raw_text:
            tables.append(
                {
                    "table_id": f"T{index + 1}",
                    "caption": match.group(1).strip(),
                    "raw_text": raw_text,
                    "structured": False,
                }
            )
    return tables


def remove_table_blocks_from_main_text(text: str, tables: list[JsonDict]) -> str:
    """用 caption 替换已识别表格正文，降低推荐抽取被表格噪声污染的概率。"""

    cleaned = str(text or "")
    for table in tables:
        raw_text = table.get("raw_text") if isinstance(table, dict) else ""
        if raw_text:
            cleaned = cleaned.replace(str(raw_text), str(table.get("caption") or ""))
    return normalize_spaces(cleaned)


def extract_recommendation_boxes(text: str) -> list[JsonDict]:
    boxes: list[JsonDict] = []
    matches = list(RECOMMENDATION_BOX_HEADING_RE.finditer(text))
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        following_boundaries = [
            boundary.start()
            for boundary in (
                SECTION_HEADING_RE.search(text, match.end(), end),
                TABLE_CAPTION_RE.search(text, match.end(), end),
            )
            if boundary
        ]
        if following_boundaries:
            end = min(following_boundaries)
        raw_text = text[start:end].strip()
        if (
            raw_text
            and len(raw_text) <= MAX_RECOMMENDATION_BOX_CHARS
            and re.search(r"(?i)\b(should|recommend|suggest|is indicated|do not|avoid)\b", raw_text)
        ):
            boxes.append(
                {
                    "box_id": f"RB{index + 1}",
                    "heading": match.group(1).strip(),
                    "text": raw_text,
                    "clean_start_char": start,
                    "clean_end_char": end,
                }
            )
    return boxes


def split_methodology_blocks(text: str) -> tuple[str, list[JsonDict]]:
    sections = split_sections(text)
    if not sections:
        return text, []

    kept: list[str] = []
    isolated: list[JsonDict] = []
    search_from = 0
    for section in sections:
        name = str(section.get("section_name") or "")
        body = str(section.get("text") or "")
        section_text = f"{name}\n{body}".strip()
        is_method_section = METHOD_SECTION_NAME_RE.match(name) is not None
        is_boilerplate = METHODOLOGY_BOILERPLATE_RE.search(body) is not None
        has_direct_recommendation = re.search(r"(?i)\b(recommendation\s+\d+|we recommend|we suggest|should)\b", body)
        if is_method_section and is_boilerplate and not has_direct_recommendation:
            start = text.find(body, search_from)
            end = start + len(body) if start >= 0 else None
            isolated.append(
                {
                    "section_name": name,
                    "text": body,
                    "clean_start_char": start if start >= 0 else None,
                    "clean_end_char": end,
                    "reason": "methodology_boilerplate_section",
                }
            )
        else:
            kept.append(section_text)
        found = text.find(body, search_from)
        if found >= 0:
            search_from = found + len(body)
    if not isolated:
        return text, []
    return normalize_spaces("\n\n".join(kept)), isolated


def source_span_boundaries(text: str) -> list[JsonDict]:
    boundaries: list[JsonDict] = []
    for match in SOURCE_SPAN_BOUNDARY_RE.finditer(text):
        line_end = text.find("\n", match.start())
        if line_end < 0:
            line_end = min(len(text), match.start() + 160)
        boundaries.append(
            {
                "label": text[match.start() : line_end].strip()[:180],
                "clean_start_char": match.start(),
                "clean_end_char": line_end,
            }
        )
    return boundaries


def table_stats_for(tables: list[object]) -> JsonDict:
    """汇总表格数量、行数和粗略单元格数量，供质量门和报告使用。"""

    row_count = 0
    cell_count = 0
    for table in tables:
        if isinstance(table, dict):
            raw_text = str(table.get("raw_text") or "")
            rows = [line for line in raw_text.splitlines() if line.strip()]
            row_count += len(rows)
            cell_count += sum(len(line.split()) for line in rows)
        elif isinstance(table, list):
            row_count += len(table)
            cell_count += sum(len(row) if isinstance(row, list) else 1 for row in table)
    return {"table_count": len(tables), "table_row_count": row_count, "table_cell_count": cell_count}


def split_sections(text: str) -> list[JsonDict]:
    """按常见指南标题切出粗粒度 sections，给后续结构解析提供边界线索。"""

    matches = list(SECTION_HEADING_RE.finditer(text))
    if not matches:
        return [{"section_name": "Document", "text": normalize_spaces(text)}] if text.strip() else []
    sections: list[JsonDict] = []
    if matches[0].start() > 0:
        sections.append({"section_name": "Document", "text": text[: matches[0].start()].strip()})
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append({"section_name": match.group(1).title(), "text": text[match.end() : end].strip()})
    return [section for section in sections if section["text"]]


def extract_recommendations(text: str) -> list[JsonDict]:
    """生成兼容历史 cleaned record 的轻量推荐摘要。

    正式 RecommendationCandidate 以后续 extraction/recommendation 为准；这里保留
    `recommendations` 字段只是为了兼容旧消费者和调试报告。
    """

    statements = split_recommendation_statements(text)
    recommendations: list[JsonDict] = []
    for index, statement in enumerate(statements, start=1):
        recommendations.append(
            {
                "recommendation_id": f"REC-{index:03d}",
                "text": statement,
                "population": None,
                "intervention": None,
                "comparator": None,
                "direction": infer_direction(statement),
                "strength": infer_strength(statement),
                "certainty": infer_certainty(statement),
                "remark": None,
                "evidence_context": "",
                "source_span": statement,
            }
        )
    return recommendations


def extract_grade_assessments(main_text: str, recommendations: list[JsonDict]) -> list[JsonDict]:
    """生成兼容历史 cleaned record 的粗粒度 GRADE 摘要。

    完整 GradeCandidate 以后续 extraction/grade 为准；这里不应承担正式 GRADE 判断职责。
    """

    match = GRADE_CERTAINTY_RE.search(main_text)
    if not match:
        return []
    span = main_text[max(0, match.start() - 180) : min(len(main_text), match.end() + 260)]
    certainty = next(group for group in match.groups() if group).lower().replace(" ", "_")
    assessment = {
        "recommendation_id": recommendations[0]["recommendation_id"] if recommendations else "",
        "outcome_name": "overall",
        "certainty": certainty,
        "risk_of_bias": "not_reported",
        "inconsistency": "not_reported",
        "indirectness": "not_reported",
        "imprecision": "not_reported",
        "publication_bias": "not_reported",
        "benefits": "",
        "harms": "",
        "resource_use": "",
        "health_equity": "",
        "acceptability": "",
        "feasibility": "",
        "rationale": span,
        "source_span": span,
    }
    for field, pattern in GRADE_REASON_MAP.items():
        if pattern.search(span):
            assessment[field] = "serious_or_concern"
    return [assessment]


def metadata_from(record: JsonDict, cleaned: JsonDict) -> JsonDict:
    """把常用来源元数据整理到稳定的 metadata 对象中。"""

    return {
        "title": cleaned.get("title", ""),
        "journal": cleaned.get("journal", ""),
        "published_year": cleaned.get("published_year", ""),
        "doi": cleaned.get("doi", ""),
        "source": cleaned.get("source", ""),
        "url": cleaned.get("url", ""),
        "raw_pdf_path": cleaned.get("raw_pdf_path", ""),
    }


def build_cleaning_log(
    raw_content: str,
    clean_content: str,
    references_text: str,
    affiliations_text: str,
    fallback_tables: list[JsonDict],
    inline_section_breaks_inserted: int = 0,
    layout_report: JsonDict | None = None,
    methodology_blocks: list[JsonDict] | None = None,
    recommendation_boxes: list[JsonDict] | None = None,
    boundary_count: int = 0,
    column_merge_repairs: int = 0,
    pdf_noise_report: JsonDict | None = None,
) -> JsonDict:
    """记录清洗阶段做过的主要非破坏性操作，方便排查是否过度清洗。"""

    raw_chars = len(raw_content or "")
    clean_chars = len(clean_content or "")
    references_chars = len(references_text or "")
    removed_ratio = 0.0 if raw_chars == 0 else 1 - (clean_chars / raw_chars)
    audit_flags = []
    if raw_chars == 0:
        audit_flags.append("empty_raw_content")
    if clean_chars == 0:
        audit_flags.append("empty_clean_content")
    if raw_chars > 0 and removed_ratio > MAX_REFERENCE_REMOVED_RATIO:
        audit_flags.append("removed_ratio_high")
    if references_chars > clean_chars * 2:
        audit_flags.append("references_larger_than_clean_text")
    if POST_REFERENCE_BODY_RE.search(references_text or ""):
        audit_flags.append("references_contains_body_heading")

    return {
        "raw_content_preserved": True,
        "light_cleaning_applied": True,
        "hyphenation_fixed": fix_hyphenation(raw_content) != raw_content,
        "headers_footers_removed": True,
        "references_split": bool(references_text),
        "affiliations_split": bool(affiliations_text),
        "tables_extracted_from_text": bool(fallback_tables),
        "inline_section_breaks_inserted": inline_section_breaks_inserted,
        "two_column_layout_repaired": bool((layout_report or {}).get("applied")),
        "two_column_layout_report": layout_report or {"applied": False},
        "column_merge_repairs": column_merge_repairs,
        "methodology_blocks_isolated": len(methodology_blocks or []),
        "recommendation_boxes_detected": len(recommendation_boxes or []),
        "source_span_boundaries_detected": boundary_count,
        "enhanced_pdf_noise_cleaning": pdf_noise_report or {"applied": False},
        "audit": {
            "raw_chars": raw_chars,
            "clean_chars": clean_chars,
            "removed_chars": max(0, raw_chars - clean_chars),
            "removed_ratio": round(removed_ratio, 4),
            "references_chars": references_chars,
            "audit_flags": audit_flags,
        },
    }


def build_cleaning_warnings(record: JsonDict, fallback_tables: list[JsonDict]) -> list[JsonDict]:
    """当清洗阶段启用了兜底逻辑时输出 warning，提示后续质量报告关注。"""

    if record.get("tables") or not fallback_tables:
        return []
    return [
        {
            "warning_type": "tables_empty_but_table_text_detected",
            "message": "record.tables is empty, but table markers were found in content.",
        }
    ]


def clean_source_record(record: JsonDict, source: Optional[str] = None) -> JsonDict:
    """清洗单条来源记录，并保留 raw_content 到 clean_content 的可追溯关系。"""

    raw_content = raw_content_from(record)
    source_key = normalize_source(source or record.get("source"))
    cleaned = base_record({**record, "source": source_key})

    light_text, layout_report = light_clean_pdf_text_with_layout_report(raw_content)
    noise_cleaned = enhance_pdf_noise_cleaning({**cleaned, "content": light_text})
    light_text = str(noise_cleaned.get("content") or light_text)
    pdf_noise_report = dict(noise_cleaned.get("cleaning_report") or {}).get("enhanced_pdf_noise_cleaning")
    light_text, column_merge_repairs = repair_pdf_column_word_merges(light_text)
    light_text, inline_section_breaks_inserted = restore_inline_section_breaks(light_text)
    main_text, references_text = split_references(light_text)
    main_text, affiliations_text = split_affiliations(main_text)
    fallback_tables = [] if record.get("tables") else extract_tables_from_text(main_text)
    tables = cleaned.get("tables") or fallback_tables
    table_dicts = [table for table in tables if isinstance(table, dict)]
    clean_content = remove_table_blocks_from_main_text(main_text, table_dicts)
    clean_content, methodology_blocks = split_methodology_blocks(clean_content)
    recommendation_boxes = extract_recommendation_boxes(clean_content)
    span_boundaries = source_span_boundaries(clean_content)
    sections = split_sections(clean_content)
    recommendations = extract_recommendations(clean_content)
    grade_assessments = extract_grade_assessments(clean_content, recommendations)

    cleaned.update(
        {
            "record_id": cleaned.get("record_id") or stable_id("record", source_key, cleaned.get("title"), cleaned.get("url")),
            "metadata": metadata_from(record, cleaned),
            "raw_content": raw_content,
            "clean_content": clean_content,
            "content": clean_content,
            "sections": sections,
            "recommendations": recommendations,
            "pico_questions": [],
            "grade_assessments": grade_assessments,
            "tables": tables,
            "recommendation_boxes": recommendation_boxes,
            "methodology_blocks": methodology_blocks,
            "source_span_boundaries": span_boundaries,
            **table_stats_for(tables),
            "references_text": references_text,
            "affiliations_text": affiliations_text,
            "cleaning_log": build_cleaning_log(
                raw_content,
                clean_content,
                references_text,
                affiliations_text,
                fallback_tables,
                inline_section_breaks_inserted=inline_section_breaks_inserted,
                layout_report=layout_report,
                methodology_blocks=methodology_blocks,
                recommendation_boxes=recommendation_boxes,
                boundary_count=len(span_boundaries),
                column_merge_repairs=column_merge_repairs,
                pdf_noise_report=pdf_noise_report if isinstance(pdf_noise_report, dict) else None,
            ),
            "offset_mapping_version": OFFSET_MAPPING_VERSION,
            "cleaning_warnings": build_cleaning_warnings(record, fallback_tables),
        }
    )
    return add_direct_schema_seeds(cleaned)


def clean_source_records(records: Iterable[JsonDict], source: Optional[str] = None) -> Iterable[JsonDict]:
    """批量清洗来源记录，输出统一 cleaned record 列表。"""

    for record in records:
        yield clean_source_record(record, source=source)


def clean_file(input_path: str | Path, output_path: str | Path, source: Optional[str] = None) -> int:
    """读取 raw JSONL，执行清洗，并写出 cleaned JSONL。"""

    return write_jsonl(output_path, clean_source_records(iter_jsonl(input_path), source=source))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将来源记录清洗为统一 cleaned-record schema。")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = clean_file(args.input, args.output, source=args.source)
    print(f"Wrote {count} cleaned records to {args.output}")


if __name__ == "__main__":
    main()
