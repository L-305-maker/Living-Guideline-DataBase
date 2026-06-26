from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any,List

from src.domain.common import stable_id
from src.pipeline.cleaning.base_cleaner import clean_inline_text, clean_structured_text
from src.common.extraction_common import JsonDict

MONTH_NAME = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
MONTH_RE = re.compile(
    rf"\b(?:{MONTH_NAME}\s+\d{{1,2}},\s+(?:19|20)\d{{2}}|\d{{1,2}}\s+{MONTH_NAME}\s+(?:19|20)\d{{2}})\b",
    re.I,
)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)

def normalize_source(value: Any) -> str:
    return clean_inline_text(value).lower()

def normalize_url(value: Any) -> str:
    url = re.sub(r"\s+", "", str(value or "").strip())
    return url.rstrip(".,;)")

def normalize_path(value: Any) -> str:
    path = str(value or "").strip()
    path = path.replace("/", "\\")
    path = path.replace("\r\n", "\\").replace("\n", "\\").replace("\r", "\\")
    path = re.sub(r"\\+", r"\\", path)
    return path

def normalize_year(*values: Any) -> str:
    for value in values:
        match = YEAR_RE.search(str(value or ""))
        if match:
            return match.group(0)
    return ""

def first_date_text(*values: Any) -> str:
    for value in values:
        text = str(value or "")
        match = MONTH_RE.search(text)
        if match:
            return clean_inline_text(match.group(0))
    return ""

def first_doi_url(*values: Any) -> str:
    for value in values:
        text = str(value or "")
        match = DOI_RE.search(text)
        if match:
            doi = match.group(0).rstrip(".,;)")
            return f"https://doi.org/{doi}"
    return ""

def collapse_doubled_letters(text: str) -> str:

    text = str(text or "")
    if len(text) < 8:
        return text
    pairs = [text[i : i + 2] for i in range(0, min(len(text), 80), 2)]
    doubled = sum(1 for pair in pairs if len(pair) == 2 and pair[0] == pair[1] and pair[0].isalpha())
    if pairs and doubled / len(pairs) >= 0.35:
        return "".join(text[i] for i in range(0, len(text), 2))
    return text

def remove_spaced_caps_noise(text: str) -> str:
    text = str(text or "")
    noise_patterns = [
        r"(?:[A-Z]\s+){8,}",
        r"C\s*O\s*P\s*Y\s*R\s*I\s*G\s*H\s*T",
        r"D\s*O\s*N\s*O\s*T\s*C\s*O\s*P\s*Y",
        r"M\s*A\s*T\s*E\s*R\s*I\s*A\s*L",
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, " ", text)
    return re.sub(r"\s+", " ", text).strip()

def clean_title(value: Any, fallback_text: str = "") -> str:
    title = collapse_doubled_letters(clean_inline_text(value))
    title = title.replace("_", " ")
    title = remove_spaced_caps_noise(title)
    if not title or len(title) < 8:
        first_line = next((line for line in str(fallback_text or "").splitlines() if line.strip()), "")
        title = clean_inline_text(first_line)[:240]
    return title.strip(" -_:")[:240]

def title_from_path(value: Any) -> str:
    path = str(value or "")
    if not path:
        return ""
    stem = Path(path).stem.replace("_", " ")
    stem = re.sub(r"\s+[0-9a-f]{8,}$", "", stem, flags=re.I)
    return clean_title(stem)

def clean_table_cell(value: Any) -> str:
    return clean_inline_text("" if value is None else value)

def normalize_table(table: Any) -> List[List[str]]:
    if not isinstance(table, list):
        return []

    rows: List[List[str]] = []
    for raw_row in table:
        if isinstance(raw_row, list):
            row = [clean_table_cell(cell) for cell in raw_row]
        else:
            row = [clean_table_cell(raw_row)]
        while row and not row[-1]:
            row.pop()
        if any(cell for cell in row):
            rows.append(row)

    if not rows:
        return []

    width = max(len(row) for row in rows)
    return [row + [""] * (width - len(row)) for row in rows]

def clean_tables(tables: Any) -> List[List[List[str]]]:
    if not isinstance(tables, list):
        return []

    cleaned: List[List[List[str]]] = []
    seen: set[str] = set()
    for table in tables:
        normalized = normalize_table(table)
        if not normalized:
            continue
        table_key = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
        if table_key in seen:
            continue
        seen.add(table_key)
        cleaned.append(normalized)
    return cleaned

def table_stats(tables: List[List[List[str]]]) -> JsonDict:
    return {
        "table_count": len(tables),
        "table_row_count": sum(len(table) for table in tables),
        "table_cell_count": sum(len(row) for table in tables for row in table),
    }

def base_record(record: JsonDict) -> JsonDict:
    source = normalize_source(record.get("source"))
    raw_content = str(record.get("content_markdown") or record.get("content") or record.get("abstract") or "")
    content = clean_structured_text(raw_content)
    title = clean_title(record.get("title"), fallback_text=content)
    url = normalize_url(record.get("url"))
    raw_pdf_path = normalize_path(record.get("raw_pdf_path"))
    published_year = normalize_year(record.get("published_year"), content, title, raw_pdf_path)
    tables = clean_tables(record.get("tables"))
    url_provenance = clean_inline_text(record.get("url_provenance")) or ("existing" if url else "not_found")

    cleaned = dict(record)
    cleaned.update(
        {
            "content": content,
            "title": title,
            "url": url,
            "published_year": published_year,
            "source": source,
            "tables": tables,
            "raw_pdf_path": raw_pdf_path,
            "url_provenance": url_provenance,
            "publication_date_text": first_date_text(record.get("published_year"), content, title),
            "record_id": stable_id("record", source, url, raw_pdf_path, title, published_year),
        }
    )
    cleaned.update(table_stats(tables))
    return add_direct_schema_seeds(cleaned)

def add_direct_schema_seeds(record: JsonDict) -> JsonDict:
    source = str(record.get("source") or "")
    title = str(record.get("title") or "")
    url = str(record.get("url") or "")
    raw_pdf_path = str(record.get("raw_pdf_path") or "")
    year = str(record.get("published_year") or "")
    guideline_id = stable_id("guideline", source, title, url or raw_pdf_path)
    paper_id = stable_id("paper", source, title, url or raw_pdf_path)

    record["guideline_seed"] = {
        "guideline_id": guideline_id,
        "title": title,
        "source": source,
        "issuer": record.get("issuer", ""),
        "publication_url": url,
        "pdf_url": url if url.lower().endswith(".pdf") else "",
        "published_date": year,
        "status": "active",
        "guideline_type": record.get("guideline_type", "standard"),
    }
    record["paper_seed"] = {
        "paper_id": paper_id,
        "title": title,
        "publication_date": year,
        "source_database": source,
        "url": url,
        "screening_status": "unscreened",
    }
    record["direct_extraction"] = {
        "record_id": record["record_id"],
        "guideline_id": guideline_id,
        "paper_id": paper_id,
        "source": source,
        "title": title,
        "url": url,
        "published_year": year,
        "raw_pdf_path": raw_pdf_path,
        "url_provenance": record.get("url_provenance", ""),
        "table_count": record.get("table_count", 0),
    }
    return record
