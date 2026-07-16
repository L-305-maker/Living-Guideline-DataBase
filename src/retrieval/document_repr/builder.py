"""Build document cards and document views from clean Markdown guidelines."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from src.pipeline.cleaning.encoder import encode_markdown
from src.retrieval.document_repr.section_classifier import classify_section
from src.utils.front_matter import parse_front_matter
from src.utils.io import DATA_DIR, iter_markdown_files, write_jsonl
from src.utils.metadata import extract_abstract


VIEW_TYPES = [
    "title_abstract",
    "scope_population",
    "recommendation_summary",
    "pico_questions",
    "heading_tree",
    "table_titles",
    "conclusion",
]

VIEW_PRIORITIES = {
    "title_abstract": 1.0,
    "scope_population": 1.15,
    "recommendation_summary": 1.35,
    "pico_questions": 1.2,
    "heading_tree": 0.8,
    "table_titles": 1.05,
    "conclusion": 0.85,
}

TABLE_LINE_RE = re.compile(r"^\s*(?:\|.*\||(?:table|fig(?:ure)?|algorithm)\s+\d*[:.\s].*|\u8868\s*\d*[:：.\s].*)", re.I)
TABLE_CAPTION_RE = re.compile(
    r"^\s*(?:table|fig(?:ure)?|algorithm)\s*\d*[A-Za-z]?\s*[:.\-–]\s+.+|"
    r"^\s*\u8868\s*\d*[A-Za-z]?\s*[:：.\-–\s].+",
    re.I,
)
PICO_LINE_RE = re.compile(r"(\bPICO\b|clinical question|key question|\u4e34\u5e8a\u95ee\u9898|\u5173\u952e\u95ee\u9898)", re.I)
QUESTION_LINE_RE = re.compile(
    r"(\b(?:clinical|key|research)\s+question\b|\bPICO\b|^\s*Q\d+[:.)-]|\?|？|"
    r"\u4e34\u5e8a\u95ee\u9898|\u5173\u952e\u95ee\u9898)",
    re.I,
)
RECOMMENDATION_LINE_RE = re.compile(
    r"(recommend(?:ation|ed)?|we recommend|we suggest|suggests?\s+against|should(?:\s+not)?|must|"
    r"is recommended|are recommended|is indicated|are indicated|not routinely indicated|"
    r"may be used|may be offered|avoid|offer|standard|option|"
    r"\u63a8\u8350|\u5efa\u8bae|\u5e94\u8be5|\u5b9c)",
    re.I,
)
SCOPE_LINE_RE = re.compile(
    r"(scope|population|target population|intended audience|applicability|eligible|eligibility|indication|"
    r"\u8303\u56f4|\u9002\u7528|\u4eba\u7fa4|\u76ee\u6807\u4eba\u7fa4|\u9002\u7528\u5bf9\u8c61)",
    re.I,
)
POPULATION_LINE_RE = re.compile(
    r"\b(?:adult|adults|children|child|pediatric|paediatric|infant|infants|adolescent|adolescents|"
    r"patient|patients|pregnan(?:t|cy)|elderly|older adults?)\b|"
    r"(成人|儿童|患儿|患者|婴幼儿|青少年|孕妇|妊娠|老年)",
    re.I,
)
ABSTRACT_START_RE = re.compile(
    r"^\s*(abstract|summary|background|objective|purpose|methods?|recommendations?|conclusions?)\s*[:：]",
    re.I,
)
ABSTRACT_STOP_RE = re.compile(r"^\s*(keywords?|citation|1(?:\.0)?\s+introduction|introduction|references)\b", re.I)
LOW_VALUE_LINE_RE = re.compile(
    r"^\s*(?:<!--\s*page:|page\s+\d+\b|keywords?\s*:|citation\s*:|doi\s*:|https?://|www\.|"
    r"submitted for publication|accepted for publication|copyright|conflict of interest|none were declared)\b",
    re.I,
)
SPACED_OCR_RE = re.compile(r"(?:[A-Za-zＡ-Ｚａ-ｚ０-９]\s+){12,}")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;。！？；])\s+")


def helper_normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def helper_clip(text: str, limit: int) -> str:
    text = helper_normalize_space(text)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def helper_is_useful_line(line: str, min_chars: int = 8) -> bool:
    cleaned = helper_normalize_space(line.strip(" -*\t|"))
    if len(cleaned) < min_chars:
        return False
    if LOW_VALUE_LINE_RE.search(cleaned):
        return False
    if SPACED_OCR_RE.search(cleaned):
        return False
    visible = [char for char in cleaned if not char.isspace()]
    if not visible:
        return False
    letters_digits = sum(1 for char in visible if char.isalnum() or "\u4e00" <= char <= "\u9fff")
    return letters_digits / max(1, len(visible)) >= 0.35


def helper_section_heading(section: Any) -> str:
    return str(getattr(section, "heading", "") or ((getattr(section, "section_path", []) or [""])[-1]))


def helper_section_path_text(section: Any) -> str:
    return " > ".join(str(item) for item in (getattr(section, "section_path", []) or []) if item)


def helper_dedupe_lines(lines: Iterable[str], max_items: int) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for line in lines:
        cleaned = helper_normalize_space(line)
        if not cleaned or cleaned.lower() in seen:
            continue
        seen.add(cleaned.lower())
        output.append(cleaned)
        if len(output) >= max_items:
            break
    return output


def helper_candidate_lines(content: str) -> Iterable[str]:
    for raw_line in content.splitlines():
        stripped = raw_line.strip(" -*\t")
        if not stripped:
            continue
        parts = [stripped]
        if len(stripped) > 700:
            parts = [part.strip() for part in SENTENCE_SPLIT_RE.split(stripped) if part.strip()]
        for part in parts:
            yield part


def helper_matching_lines(content: str, pattern: re.Pattern[str], max_items: int, max_chars: int = 700) -> list[str]:
    lines = []
    for line in helper_candidate_lines(content):
        if pattern.search(line) and helper_is_useful_line(line):
            lines.append(helper_clip(line, max_chars))
    return helper_dedupe_lines(lines, max_items)


def helper_matching_lines_from_sections(
    sections: Iterable[Any],
    pattern: re.Pattern[str],
    max_items: int,
    max_chars: int = 700,
) -> list[str]:
    lines = []
    for section in sections:
        heading = helper_section_path_text(section) or helper_section_heading(section)
        if heading and pattern.search(heading) and helper_is_useful_line(heading):
            lines.append(helper_clip(heading, max_chars))
        lines.extend(helper_matching_lines(getattr(section, "content", ""), pattern, max_items, max_chars))
        if len(lines) >= max_items * 2:
            break
    return helper_dedupe_lines(lines, max_items)


def helper_interesting_lines(content: str, pattern: re.Pattern[str], max_items: int, fallback_chars: int = 600) -> list[str]:
    lines = helper_matching_lines(content, pattern, max_items)
    if lines:
        return helper_dedupe_lines(lines, max_items)
    fallback = helper_clip(content, fallback_chars)
    return [fallback] if fallback else []


def helper_table_lines(content: str, max_items: int = 16) -> list[str]:
    captions = []
    fallback_headers = []
    for line in content.splitlines():
        stripped = line.strip()
        if not helper_is_useful_line(stripped):
            continue
        if TABLE_CAPTION_RE.search(stripped):
            captions.append(helper_clip(stripped, 500))
        elif TABLE_LINE_RE.search(stripped) and not stripped.startswith("|"):
            fallback_headers.append(helper_clip(stripped, 500))
    return helper_dedupe_lines(captions or fallback_headers, max_items)


def helper_extract_abstract_text(body: str, max_chars: int = 1500) -> str:
    anchored: list[str] = []
    collecting = False
    for line in body.splitlines():
        stripped = line.strip(" #*\t")
        if not stripped:
            continue
        if ABSTRACT_STOP_RE.search(stripped) and anchored:
            break
        if ABSTRACT_START_RE.search(stripped):
            collecting = True
        if collecting and helper_is_useful_line(stripped, min_chars=20):
            anchored.append(stripped)
        if sum(len(item) for item in anchored) >= max_chars:
            break
    if anchored:
        return " ".join(anchored)[:max_chars]

    fallback_lines = []
    for line in extract_abstract(body, max_chars=max_chars * 2).splitlines():
        if helper_is_useful_line(line, min_chars=20):
            fallback_lines.append(line)
    return " ".join(fallback_lines)[:max_chars]


def helper_infer_scope_population(title_abstract: str, recommendation_lines: Iterable[str], max_items: int = 8) -> list[str]:
    haystack = "\n".join([title_abstract, *recommendation_lines])
    return helper_matching_lines(haystack, POPULATION_LINE_RE, max_items)


def helper_collect_document_signals(sections: Iterable[Any]) -> dict[str, list[str]]:
    signals: dict[str, list[str]] = {
        "recommendation": [],
        "question": [],
        "scope": [],
        "population": [],
        "table": [],
    }
    pool_limits = {
        "recommendation": 36,
        "question": 28,
        "scope": 24,
        "population": 20,
        "table": 36,
    }

    def needs(name: str) -> bool:
        return len(signals[name]) < pool_limits[name]

    core_signal_names = ("recommendation", "question", "scope", "population")
    scanned_lines = 0
    # 扫描上限防止超长指南在构建 card 时无限消耗时间和内存。
    max_candidate_lines = 3000
    for section in sections:
        # 核心信号池达到上限后提前结束，不再扫描对 card 无增益的正文。
        if not any(needs(name) for name in core_signal_names):
            break
        heading = helper_section_path_text(section) or helper_section_heading(section)
        if heading and helper_is_useful_line(heading):
            if needs("recommendation") and RECOMMENDATION_LINE_RE.search(heading):
                signals["recommendation"].append(helper_clip(heading, 800))
            if needs("question") and QUESTION_LINE_RE.search(heading):
                signals["question"].append(helper_clip(heading, 700))
            if needs("scope") and SCOPE_LINE_RE.search(heading):
                signals["scope"].append(helper_clip(heading, 700))
            if needs("population") and POPULATION_LINE_RE.search(heading):
                signals["population"].append(helper_clip(heading, 650))

        content = getattr(section, "content", "")
        for line in helper_candidate_lines(content):
            if not any(needs(name) for name in core_signal_names) or scanned_lines >= max_candidate_lines:
                break
            scanned_lines += 1
            if not helper_is_useful_line(line):
                continue
            if needs("table") and TABLE_CAPTION_RE.search(line):
                signals["table"].append(helper_clip(line, 500))
            if needs("recommendation") and RECOMMENDATION_LINE_RE.search(line):
                signals["recommendation"].append(helper_clip(line, 800))
            if needs("question") and QUESTION_LINE_RE.search(line):
                signals["question"].append(helper_clip(line, 700))
            if needs("scope") and SCOPE_LINE_RE.search(line):
                signals["scope"].append(helper_clip(line, 700))
            if needs("population") and POPULATION_LINE_RE.search(line):
                signals["population"].append(helper_clip(line, 650))
        if scanned_lines >= max_candidate_lines:
            break

    return {
        "recommendation": helper_dedupe_lines(signals["recommendation"], 18),
        "question": helper_dedupe_lines(signals["question"], 14),
        "scope": helper_dedupe_lines(signals["scope"], 12),
        "population": helper_dedupe_lines(signals["population"], 10),
        "table": helper_dedupe_lines(signals["table"], 18),
    }


def helper_heading_tree(sections: list[Any], max_items: int = 80) -> str:
    lines = []
    for section in sections:
        heading = helper_section_heading(section)
        if not heading:
            continue
        level = int(getattr(section, "heading_level", 1) or 1)
        indent = "  " * max(0, level - 1)
        lines.append(f"{indent}- {heading}")
        if len(lines) >= max_items:
            break
    return "\n".join(helper_dedupe_lines(lines, max_items))


def helper_metadata_from_markdown(markdown: str) -> tuple[dict[str, str], str]:
    metadata, body = parse_front_matter(markdown)
    return metadata, body


def helper_base_metadata(metadata: dict[str, str], path: str | Path) -> dict[str, Any]:
    return {
        "doc_id": metadata.get("id") or Path(path).stem,
        "title": metadata.get("title") or Path(path).stem,
        "publication_date": metadata.get("publication_date") or "unknown",
        "source_institution": metadata.get("source_institution") or "Unknown",
        "clinical_department": metadata.get("clinical_department") or "\u672a\u5206\u7c7b",
        "clinical_departments": metadata.get("clinical_departments") or [metadata.get("clinical_department") or "未分类"],
        "department_scope": metadata.get("department_scope") or "single",
        "document_kind": metadata.get("document_kind") or "guideline",
        "cleaning_quality": metadata.get("cleaning_quality") or "",
        "cleaning_flags": metadata.get("cleaning_flags") or "",
        "source_file": metadata.get("source_file") or "",
        "source_pdf_text_quality": metadata.get("source_pdf_text_quality") or "",
        "source_pdf_needs_ocr": metadata.get("source_pdf_needs_ocr") or "",
        "source_pdf_is_scanned": metadata.get("source_pdf_is_scanned") or "",
        "pdf_text_quality": metadata.get("pdf_text_quality") or "",
        "pdf_needs_ocr": metadata.get("pdf_needs_ocr") or "",
        "pdf_is_scanned": metadata.get("pdf_is_scanned") or "",
        "ocr_engine": metadata.get("ocr_engine") or "",
        "ocr_applied": metadata.get("ocr_applied") or "",
        "ocr_status": metadata.get("ocr_status") or "",
        "ocr_error": metadata.get("ocr_error") or "",
        "markdown_clean_path": str(path),
    }


def helper_section_groups(sections: list[Any]) -> dict[str, list[Any]]:
    groups: dict[str, list[Any]] = defaultdict(list)
    for section in sections:
        section_type = "reference" if getattr(section, "is_reference_section", False) else classify_section(
            heading=helper_section_heading(section),
            section_path=getattr(section, "section_path", []),
            content=getattr(section, "content", ""),
        )
        groups[section_type].append(section)
    return groups


def helper_section_snippets(sections: Iterable[Any], max_sections: int, chars_per_section: int) -> str:
    parts = []
    for section in sections:
        heading = helper_section_path_text(section) or helper_section_heading(section)
        content = helper_clip(getattr(section, "content", ""), chars_per_section)
        if not content:
            continue
        parts.append(f"{heading}: {content}" if heading else content)
        if len(parts) >= max_sections:
            break
    return "\n".join(parts)


def build_document_card(markdown: str, path: str | Path = "") -> dict[str, Any]:
    """Create one high-density document card from a clean Markdown guideline."""

    metadata, body = helper_metadata_from_markdown(markdown)
    base = helper_base_metadata(metadata, path)
    sections = encode_markdown(markdown)
    groups = helper_section_groups(sections)

    abstract_text = helper_extract_abstract_text(body, max_chars=1500)
    title_abstract = " ".join([base["title"], abstract_text]).strip()
    signals = helper_collect_document_signals(sections)
    recommendation_lines = signals["recommendation"]
    pico_lines = signals["question"]
    scope_lines = signals["scope"]
    population_lines = signals["population"]
    population_lines.extend(helper_infer_scope_population(title_abstract, recommendation_lines, 10))
    population_lines = helper_dedupe_lines(population_lines, 10)
    table_titles = signals["table"]

    fields = {
        "title_abstract": helper_clip(title_abstract, 1800),
        "scope": helper_clip("\n".join(scope_lines) or helper_section_snippets(groups["scope_population"], 4, 700), 2200),
        "target_population": helper_clip("\n".join(population_lines), 1400),
        "key_recommendations": "\n".join(recommendation_lines),
        "clinical_questions_pico": "\n".join(pico_lines),
        "evidence_review": helper_clip(helper_section_snippets(groups["evidence"], 5, 600), 2600),
        "heading_tree": helper_heading_tree(sections),
        "important_tables": "\n".join(helper_dedupe_lines(table_titles, 18)),
        "conclusion": helper_clip(helper_section_snippets(groups["conclusion"], 3, 600), 1600),
    }
    card_sections = [
        ("Title", base["title"]),
        ("Publisher", base["source_institution"]),
        ("Document Type", "guideline / consensus"),
        ("Abstract", abstract_text),
        ("Scope", fields["scope"]),
        ("Target Population", fields["target_population"]),
        ("Key Recommendations", fields["key_recommendations"]),
        ("Clinical Questions / PICO", fields["clinical_questions_pico"]),
        ("Evidence Review", fields["evidence_review"]),
        ("Heading Tree", fields["heading_tree"]),
        ("Important Tables", fields["important_tables"]),
        ("Conclusion", fields["conclusion"]),
    ]
    card_text = "\n\n".join(f"[{name}]\n{value}" for name, value in card_sections if value)
    return {
        **base,
        "card_text": helper_clip(card_text, 12000),
        "fields": fields,
    }


def helper_view_record(card: dict[str, Any], view_type: str, text: str) -> dict[str, Any] | None:
    text = helper_clip(text, 6000)
    if not text:
        return None
    doc_id = card["doc_id"]
    return {
        "view_id": f"{doc_id}__{view_type}",
        "doc_id": doc_id,
        "view_type": view_type,
        "text": text,
        "priority": VIEW_PRIORITIES.get(view_type, 1.0),
        "title": card.get("title", ""),
        "publication_date": card.get("publication_date", "unknown"),
        "source_institution": card.get("source_institution", "Unknown"),
        "clinical_department": card.get("clinical_department", "\u672a\u5206\u7c7b"),
        "clinical_departments": card.get("clinical_departments") or [card.get("clinical_department", "未分类")],
        "department_scope": card.get("department_scope") or "single",
        "document_kind": card.get("document_kind") or "guideline",
        "cleaning_quality": card.get("cleaning_quality", ""),
        "cleaning_flags": card.get("cleaning_flags", ""),
        "source_pdf_text_quality": card.get("source_pdf_text_quality", ""),
        "source_pdf_needs_ocr": card.get("source_pdf_needs_ocr", ""),
        "source_pdf_is_scanned": card.get("source_pdf_is_scanned", ""),
        "pdf_text_quality": card.get("pdf_text_quality", ""),
        "pdf_needs_ocr": card.get("pdf_needs_ocr", ""),
        "pdf_is_scanned": card.get("pdf_is_scanned", ""),
        "ocr_engine": card.get("ocr_engine", ""),
        "ocr_applied": card.get("ocr_applied", ""),
        "ocr_status": card.get("ocr_status", ""),
        "ocr_error": card.get("ocr_error", ""),
        "markdown_clean_path": card.get("markdown_clean_path", ""),
    }


def build_document_views(card: dict[str, Any]) -> list[dict[str, Any]]:
    """Create independently searchable document views from a card."""

    fields = card.get("fields") or {}
    view_texts = {
        "title_abstract": fields.get("title_abstract", ""),
        "scope_population": "\n".join([fields.get("scope", ""), fields.get("target_population", "")]).strip(),
        "recommendation_summary": fields.get("key_recommendations", ""),
        "pico_questions": fields.get("clinical_questions_pico", ""),
        "heading_tree": fields.get("heading_tree", ""),
        "table_titles": fields.get("important_tables", ""),
        "conclusion": fields.get("conclusion", ""),
    }
    records = [helper_view_record(card, view_type, view_texts.get(view_type, "")) for view_type in VIEW_TYPES]
    return [record for record in records if record is not None]


def build_document_representations(
    clean_dir: str | Path = DATA_DIR / "markdown_clean",
    output_dir: str | Path = DATA_DIR,
) -> dict[str, Any]:
    """Build document card and view JSONL artifacts for all clean Markdown files."""

    cards: list[dict[str, Any]] = []
    views: list[dict[str, Any]] = []
    for path in iter_markdown_files(clean_dir):
        markdown = path.read_text(encoding="utf-8", errors="replace")
        card = build_document_card(markdown, path)
        cards.append(card)
        # 每个 card 可派生多个检索意图 view，但仍共享同一个 doc_id。
        views.extend(build_document_views(card))

    output_path = Path(output_dir)
    cards_path = output_path / "document_cards.jsonl"
    views_path = output_path / "document_views.jsonl"
    write_jsonl(cards_path, cards)
    write_jsonl(views_path, views)
    return {
        "documents": len(cards),
        "views": len(views),
        "document_cards": str(cards_path),
        "document_views": str(views_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", default=str(DATA_DIR / "markdown_clean"))
    parser.add_argument("--output-dir", default=str(DATA_DIR))
    args = parser.parse_args()
    print(json.dumps(build_document_representations(args.clean_dir, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
