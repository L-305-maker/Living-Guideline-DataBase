"""Repair common post-ingest metadata and retrieval quality issues."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from src.pipeline.cleaning.encoder import REFERENCE_MARKER_RE, REFERENCE_RE, REFERENCE_LIST_RE
from src.utils.front_matter import dump_front_matter, parse_front_matter
from src.utils.ids import sha256_text
from src.utils.io import DATA_DIR, read_jsonl, write_jsonl
from src.utils.metadata import clean_title, extract_abstract, extract_publication_date, extract_title, is_suspicious_title


VALID_DATE_RE = re.compile(r"^(20[1-2]\d)-\d{2}-\d{2}$")
READABLE_CHAR_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff，。；：、“”‘’（）《》—\-.,;:!?()/%\s]")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def _year(value: str | None) -> int | None:
    match = re.search(r"(19\d{2}|20\d{2})", value or "")
    return int(match.group(1)) if match else None


def _valid_publication_date(value: str | None, year_start: int = 2012, year_end: int = 2026) -> bool:
    if not value or value == "unknown":
        return False
    if not VALID_DATE_RE.match(value):
        return False
    year = _year(value)
    return year is not None and year_start <= year <= year_end


def _reference_like(content: str, section_path: list[Any] | None = None) -> bool:
    normalized_path = [str(item).strip() for item in (section_path or []) if str(item).strip()]
    return bool(
        any(REFERENCE_RE.match(item) for item in normalized_path)
        or REFERENCE_MARKER_RE.search(content or "")
        or REFERENCE_LIST_RE.search(content or "")
    )


def _fallback_abstract(body: str, max_chars: int = 800) -> str:
    text = re.sub(r"<!--.*?-->", " ", body or "", flags=re.S)
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("|") or line.startswith("!"):
            continue
        line = re.sub(r"^#{1,6}\s*", "", line).strip()
        if not line or REFERENCE_RE.match(line):
            continue
        lines.append(line)
        if sum(len(item) for item in lines) >= max_chars:
            break
    return re.sub(r"\s+", " ", " ".join(lines)).strip()[:max_chars]


def _readable_ratio(text: str) -> float:
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return 0.0
    readable = sum(1 for char in compact if READABLE_CHAR_RE.match(char))
    return readable / len(compact)


def _usable_abstract(text: str) -> str:
    abstract = re.sub(r"\s+", " ", text or "").strip()
    if len(abstract) > 800:
        abstract = abstract[:800]
    if not abstract or CONTROL_CHAR_RE.search(abstract) or _readable_ratio(abstract) < 0.55:
        return ""
    return abstract


def _load_markdown(record: dict[str, Any]) -> tuple[Path | None, dict[str, str], str, str]:
    path_value = record.get("markdown_clean_path") or ""
    path = Path(path_value) if path_value else None
    if path is not None and path.exists():
        markdown = path.read_text(encoding="utf-8", errors="replace")
        metadata, body = parse_front_matter(markdown)
        return path, metadata, body, markdown
    return None, {}, "", ""


def _repair_document(record: dict[str, Any], year_start: int, year_end: int) -> tuple[dict[str, Any], bool]:
    path, metadata, body, markdown = _load_markdown(record)
    changed = False
    repaired = dict(record)

    source_file = repaired.get("source_file") or metadata.get("source_file") or repaired.get("markdown_raw_path") or ""
    current_title = clean_title(repaired.get("title") or metadata.get("title") or "")
    candidate_title = extract_title(body or markdown, source_file)
    if not current_title or is_suspicious_title(current_title):
        if candidate_title and candidate_title != current_title:
            repaired["title"] = candidate_title
            changed = True
    else:
        repaired["title"] = current_title

    current_date = repaired.get("publication_date") or metadata.get("publication_date") or "unknown"
    if not _valid_publication_date(current_date, year_start, year_end):
        candidate_date = extract_publication_date(source_file, body or markdown)
        if candidate_date != current_date:
            repaired["publication_date"] = candidate_date
            changed = True

    if body:
        abstract = _usable_abstract(extract_abstract(body)) or _usable_abstract(_fallback_abstract(body))
        if not abstract:
            abstract = repaired.get("title", "")
        if abstract != repaired.get("abstract", ""):
            repaired["abstract"] = abstract
            changed = True

    if path is not None and metadata:
        metadata.update(
            {
                "title": repaired.get("title", ""),
                "publication_date": repaired.get("publication_date", "unknown"),
                "source_institution": repaired.get("source_institution") or metadata.get("source_institution", "Unknown"),
                "source_file": repaired.get("source_file") or metadata.get("source_file", ""),
                "clinical_department": repaired.get("clinical_department") or metadata.get("clinical_department", "未分类"),
            }
        )
        updated_markdown = dump_front_matter(metadata, body)
        if updated_markdown != markdown:
            path.write_text(updated_markdown, encoding="utf-8", newline="\n")
            repaired["content_sha256"] = sha256_text(updated_markdown)
            changed = True

    return repaired, changed


def _repair_sections(data_dir: Path, documents: dict[str, dict[str, Any]]) -> dict[str, int]:
    changed_files = 0
    changed_rows = 0
    for path in sorted((data_dir / "sections").glob("*.jsonl")):
        rows = list(read_jsonl(path))
        new_rows: list[dict[str, Any]] = []
        file_changed = False
        for row in rows:
            new_row = dict(row)
            doc = documents.get(new_row.get("doc_id", ""))
            if not doc:
                file_changed = True
                continue
            for key in ("title", "publication_date", "source_institution", "clinical_department"):
                value = doc.get(key)
                if value is not None and new_row.get(key) != value:
                    new_row[key] = value
                    file_changed = True
            reference = _reference_like(new_row.get("content", ""), new_row.get("section_path") or [])
            if bool(new_row.get("is_reference_section")) != reference:
                new_row["is_reference_section"] = reference
                file_changed = True
            if new_row != row:
                changed_rows += 1
            new_rows.append(new_row)
        if file_changed:
            write_jsonl(path, new_rows)
            changed_files += 1
    return {"section_files_changed": changed_files, "section_rows_changed": changed_rows}


def _repair_chunks(data_dir: Path, documents: dict[str, dict[str, Any]]) -> dict[str, int]:
    changed_files = 0
    changed_rows = 0
    all_rows: list[dict[str, Any]] = []
    chunk_files = sorted(path for path in (data_dir / "chunks").glob("*.jsonl") if not path.name.startswith("all_chunks"))
    for path in chunk_files:
        rows = list(read_jsonl(path))
        new_rows: list[dict[str, Any]] = []
        file_changed = False
        for row in rows:
            new_row = dict(row)
            doc = documents.get(new_row.get("doc_id", ""))
            if not doc:
                file_changed = True
                continue
            for key in ("title", "publication_date", "source_institution", "clinical_department"):
                value = doc.get(key)
                if value is not None and new_row.get(key) != value:
                    new_row[key] = value
                    file_changed = True
            reference = _reference_like(new_row.get("content", ""), new_row.get("section_path") or [])
            if bool(new_row.get("is_reference_section")) != reference:
                new_row["is_reference_section"] = reference
                file_changed = True
            if new_row != row:
                changed_rows += 1
            new_rows.append(new_row)
        if file_changed:
            write_jsonl(path, new_rows)
            changed_files += 1
        all_rows.extend(new_rows)
    write_jsonl(data_dir / "chunks" / "all_chunks.jsonl", all_rows)
    return {"chunk_files_changed": changed_files, "chunk_rows_changed": changed_rows, "chunks_total": len(all_rows)}


def _repair_vector_metadata(data_dir: Path) -> dict[str, Any]:
    index_dir = data_dir / "index"
    metadata_path = index_dir / "vector_indexes_metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {}
    for key, manifest_name in (
        ("document_card_index", "faiss_document_cards_shards.json"),
        ("document_view_index", "faiss_document_views_shards.json"),
        ("chunk_index", "faiss_chunks_shards.json"),
    ):
        manifest_path = index_dir / manifest_name
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata[key] = {
            "count": manifest.get("count"),
            "dim": manifest.get("dim"),
            "shards": len(manifest.get("shards", [])),
            "manifest": str(manifest_path),
        }
        metadata.setdefault("model", manifest.get("model"))
    expected_cards = sum(1 for _ in read_jsonl(data_dir / "document_cards.jsonl"))
    expected_views = sum(1 for _ in read_jsonl(data_dir / "document_views.jsonl"))
    expected_chunks = sum(1 for _ in read_jsonl(data_dir / "chunks" / "all_chunks.jsonl"))
    card_count = (metadata.get("document_card_index") or {}).get("count")
    view_count = (metadata.get("document_view_index") or {}).get("count")
    chunk_count = (metadata.get("chunk_index") or {}).get("count")
    metadata["expected_document_card_count"] = expected_cards
    metadata["expected_document_view_count"] = expected_views
    metadata["expected_chunk_count"] = expected_chunks
    metadata["needs_rebuild"] = card_count != expected_cards or view_count != expected_views or chunk_count != expected_chunks
    metadata["built"] = bool(metadata.get("document_card_index") or metadata.get("document_view_index") or metadata.get("chunk_index"))
    if metadata:
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"vector_metadata_repaired": bool(metadata)}


def repair_quality(data_dir: str | Path = DATA_DIR, year_start: int = 2012, year_end: int = 2026) -> dict[str, Any]:
    data_path = Path(data_dir)
    documents_path = data_path / "documents.jsonl"
    repaired_documents: list[dict[str, Any]] = []
    changed_documents = 0
    title_repairs = 0
    date_repairs = 0

    for record in read_jsonl(documents_path):
        old_title = record.get("title")
        old_date = record.get("publication_date")
        repaired, changed = _repair_document(record, year_start, year_end)
        if changed:
            changed_documents += 1
        if repaired.get("title") != old_title:
            title_repairs += 1
        if repaired.get("publication_date") != old_date:
            date_repairs += 1
        if _valid_publication_date(repaired.get("publication_date"), year_start, year_end):
            repaired_documents.append(repaired)

    write_jsonl(documents_path, repaired_documents)
    documents_by_id = {record["doc_id"]: record for record in repaired_documents}
    section_stats = _repair_sections(data_path, documents_by_id)
    chunk_stats = _repair_chunks(data_path, documents_by_id)
    vector_stats = _repair_vector_metadata(data_path)

    return {
        "documents": len(repaired_documents),
        "year_start": year_start,
        "year_end": year_end,
        "changed_documents": changed_documents,
        "title_repairs": title_repairs,
        "date_repairs": date_repairs,
        **section_stats,
        **chunk_stats,
        **vector_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--year-start", type=int, default=2012)
    parser.add_argument("--year-end", type=int, default=2026)
    args = parser.parse_args()
    print(json.dumps(repair_quality(args.data_dir, args.year_start, args.year_end), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
