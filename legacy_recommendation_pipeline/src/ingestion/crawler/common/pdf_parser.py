"""爬虫公共工具文件：封装 HTTP、JSONL、PDF 和文本处理等 ingestion 阶段共享能力。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.ingestion.crawler.common.text import extract_year, normalize_space


logger = logging.getLogger(__name__)


def parse_pdf(path: str | Path) -> dict[str, Any]:
    try:
        import pdfplumber
        from pdfplumber.utils import extract_text as words_to_text
    except ImportError as exc:
        raise RuntimeError("pdfplumber is required for PDF parsing. Install dependencies from requirements.txt.") from exc

    pdf_path = Path(path)
    text_parts: list[str] = []
    all_tables: list[list[list[str]]] = []
    metadata: dict[str, Any] = {}
    first_page_text = ""

    with pdfplumber.open(str(pdf_path)) as pdf:
        metadata = pdf.metadata or {}
        for page_no, page in enumerate(pdf.pages, 1):
            try:
                tables = page.extract_tables() or []
                all_tables.extend(_normalize_table(table) for table in tables if table)
                page_text = _extract_text_without_tables(page, words_to_text)
                if page_no == 1:
                    first_page_text = page_text
                if page_text:
                    text_parts.append(page_text)
            except Exception as exc:
                logger.warning("Failed to parse page %s in %s: %s", page_no, pdf_path, exc)

    content = normalize_space("\n\n".join(text_parts))
    title = _metadata_title(metadata) or _title_from_first_page(first_page_text) or pdf_path.stem
    published_year = extract_year(
        str(metadata.get("CreationDate", "")),
        str(metadata.get("ModDate", "")),
        first_page_text[:3000],
        pdf_path.name,
    )
    return {
        "content": content,
        "title": title,
        "published_year": published_year,
        "tables": all_tables,
        "pdf_metadata": metadata,
    }


def _extract_text_without_tables(page: Any, words_to_text: Any) -> str:
    try:
        table_bboxes = [table.bbox for table in page.find_tables()]
        if not table_bboxes:
            return page.extract_text() or ""
        words = page.extract_words()
        kept_words = []
        for word in words:
            x_mid = (float(word["x0"]) + float(word["x1"])) / 2
            y_mid = (float(word["top"]) + float(word["bottom"])) / 2
            if not any(_point_in_bbox(x_mid, y_mid, bbox) for bbox in table_bboxes):
                kept_words.append(word)
        return words_to_text(kept_words) if kept_words else ""
    except Exception as exc:
        logger.warning("Falling back to page.extract_text after table-aware extraction failed: %s", exc)
        return page.extract_text() or ""


def _point_in_bbox(x: float, y: float, bbox: tuple[float, float, float, float]) -> bool:
    x0, top, x1, bottom = bbox
    return x0 <= x <= x1 and top <= y <= bottom


def _normalize_table(table: list[list[Any]]) -> list[list[str]]:
    return [[normalize_space("" if cell is None else str(cell)) for cell in row] for row in table]


def _metadata_title(metadata: dict[str, Any]) -> str:
    for key in ("Title", "title", "Subject"):
        value = normalize_space(str(metadata.get(key, "")))
        if value and value.lower() not in {"untitled", "none"}:
            return value
    return ""


def _title_from_first_page(text: str) -> str:
    lines = [normalize_space(line) for line in text.splitlines() if normalize_space(line)]
    return lines[0] if lines else ""

