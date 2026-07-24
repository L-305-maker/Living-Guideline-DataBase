from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.cleaning.chunker import chunk_all
from src.pipeline.cleaning.cleaner import clean_all
from src.pipeline.cleaning.encoder import encode_all
from src.pipeline.cleaning.pdf_to_md import (
    helper_ensure_title_heading,
    helper_inspect_pdf_for_ingestion,
    helper_pdf_quality_metadata,
    helper_with_pymupdf,
    helper_with_pymupdf4llm,
)
from src.retrieval.document_repr import build_document_representations
from src.utils.clinical_department import classify_clinical_departments
from src.utils.front_matter import dump_front_matter, parse_front_matter
from src.utils.ids import make_doc_id, sha256_file
from src.utils.metadata import extract_abstract, extract_publication_date, extract_source_institution, extract_title


DATA_DIR = ROOT / "data/evidence_last"
AUDIT = ROOT / "data/reports/evidence_candidate_guideline_audit_20260721/candidate_guideline_audit_final.csv"
SOURCE_INVENTORY = ROOT / "data/evidence/source_pdf_inventory.jsonl"
CONSENSUS_FILES = {
    "data/raw_pdf/cua_pdf/7245_v3.pdf",
    "data/raw_pdf/gin/pdfs/92.full.pdf",
}
IMAGE_RE = re.compile(r"(?P<prefix>\(|src=[\"'])images/(?P<name>[^)\"']+)")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def selected_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in rows:
        relative = row.get("relative_source_file", "").replace("\\", "/")
        if row.get("doc_id", "").strip():
            continue
        if row.get("audit_decision") == "medical_guideline":
            selected.append({**row, "document_kind": "guideline"})
        elif relative in CONSENSUS_FILES:
            selected.append({**row, "document_kind": "consensus"})
    return selected


def active_source_hashes(path: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("status") == "active" and row.get("sha256"):
                hashes[str(row["sha256"])] = str(row.get("source_file") or "")
    return hashes


def route_targets(rows: list[dict[str, str]], active_hashes: dict[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for row in rows:
        source = Path(row["source_file"])
        if not source.is_file():
            excluded.append({**row, "exclusion_reason": "missing_source_file", "canonical_source_file": ""})
            continue
        file_hash = sha256_file(source)
        base = {**row, "sha256": file_hash}
        if file_hash in active_hashes:
            excluded.append(
                {**base, "exclusion_reason": "already_processed_hash", "canonical_source_file": active_hashes[file_hash]}
            )
        elif file_hash in seen:
            excluded.append({**base, "exclusion_reason": "duplicate_target_hash", "canonical_source_file": seen[file_hash]})
        else:
            seen[file_hash] = str(source)
            included.append(base)
    return included, excluded


def preferred_title(row: dict[str, Any], markdown: str, source: Path) -> str:
    extracted = extract_title(markdown, source)
    audited = str(row.get("pdf_visible_title") or row.get("inventory_title") or "").strip()
    if extracted and extracted != "Untitled":
        return extracted
    return audited or source.stem


def metadata_for(row: dict[str, Any], raw: str, report: Any) -> dict[str, Any]:
    source = Path(row["source_file"])
    title = preferred_title(row, raw, source)
    publication_date = extract_publication_date(source, raw)
    institution = extract_source_institution(source, raw, title=title)
    doc_id = make_doc_id(institution, publication_date, row["sha256"])
    body = helper_ensure_title_heading(raw, title)
    abstract = extract_abstract(body)
    department = classify_clinical_departments(title, abstract, body)
    quality = helper_pdf_quality_metadata(report, "pdf")
    return {
        "id": doc_id,
        "title": title,
        "publication_date": publication_date,
        "source_institution": institution,
        "source_file": str(source),
        "clinical_department": department["clinical_department"],
        "clinical_departments": department["clinical_departments"],
        "department_scope": department["department_scope"],
        "document_kind": row["document_kind"],
        "source_pdf_text_quality": quality["pdf_text_quality"],
        "source_pdf_needs_ocr": quality["pdf_needs_ocr"],
        "source_pdf_is_scanned": quality["pdf_is_scanned"],
        **quality,
        "ocr_engine": "",
        "ocr_applied": "false",
        "ocr_status": "needed_pending_mineru" if report.needs_ocr else "not_needed",
        "ocr_error": "",
    }


def prepare_one(row: dict[str, Any], raw_dir: Path) -> dict[str, Any]:
    source = Path(row["source_file"])
    report = helper_inspect_pdf_for_ingestion(source)
    if report is None:
        raise RuntimeError(f"Cannot inspect PDF: {source}")
    if report.needs_ocr:
        raw = helper_with_pymupdf(source)
        metadata = metadata_for(row, raw, report)
        return {**row, **metadata, "route": "mineru", "pdf_page_count": report.pages}
    raw = helper_with_pymupdf4llm(source) or helper_with_pymupdf(source)
    metadata = metadata_for(row, raw, report)
    body = helper_ensure_title_heading(raw, str(metadata["title"]))
    out = raw_dir / f"{metadata['id']}.md"
    out.write_text(dump_front_matter(metadata, body), encoding="utf-8", newline="\n")
    return {**row, **metadata, "route": "direct_text", "pdf_page_count": report.pages, "markdown_raw_path": str(out)}


def prepare(workers: int) -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = DATA_DIR / "markdown_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    targets, excluded = route_targets(selected_rows(read_csv(AUDIT)), active_source_hashes(SOURCE_INVENTORY))
    prepared: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(prepare_one, row, raw_dir): row for row in targets}
        for index, future in enumerate(as_completed(futures), start=1):
            prepared.append(future.result())
            if index % 25 == 0:
                print(f"PREPARED {index}/{len(targets)}", flush=True)
    prepared.sort(key=lambda row: str(row["source_file"]))
    fields = [
        "id", "document_kind", "route", "title", "publication_date", "source_institution", "source_file",
        "sha256", "pdf_page_count", "pdf_text_quality", "pdf_needs_ocr", "ocr_status", "markdown_raw_path",
    ]
    write_csv(DATA_DIR / "processing_inventory.csv", prepared, fields)
    excluded_fields = ["document_kind", "source_file", "sha256", "exclusion_reason", "canonical_source_file"]
    write_csv(DATA_DIR / "processing_excluded.csv", excluded, excluded_fields)
    mineru = [
        {
            "candidate_type": "repair_document",
            "doc_id": row["id"],
            "markdown_filename": f"{row['id']}.md",
            "source_institution": row["source_institution"],
            "title": row["title"],
            "source_file": row["source_file"],
            "pdf_page_count": row["pdf_page_count"],
            "cached_pages": 0,
            "reasons": f"pdf_text_quality:{row['pdf_text_quality']}",
            "action": "pending_baidu_ocr",
            "canonical_doc_id": "",
            "source_resolved": "True",
            "review_status": "confirmed_target",
            "inventory_status": "needs_mineru",
            "decision_source": "evidence_last_processing",
        }
        for row in prepared if row["route"] == "mineru"
    ]
    mineru_fields = list(mineru[0]) if mineru else [
        "candidate_type", "doc_id", "markdown_filename", "source_institution", "title", "source_file",
        "pdf_page_count", "cached_pages", "reasons", "action", "canonical_doc_id", "source_resolved",
        "review_status", "inventory_status", "decision_source",
    ]
    write_csv(DATA_DIR / "mineru_inventory.csv", mineru, mineru_fields)
    summary = {
        "selected_before_exclusions": len(targets) + len(excluded),
        "included": len(prepared),
        "excluded": len(excluded),
        "routes": dict(Counter(row["route"] for row in prepared)),
        "document_kinds": dict(Counter(row["document_kind"] for row in prepared)),
    }
    (DATA_DIR / "prepare_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def locate_mineru_markdown(doc_id: str) -> Path:
    matches = list((DATA_DIR / "mineru_batch/output").rglob(f"{doc_id}.md"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one MinerU Markdown for {doc_id}, found {len(matches)}")
    return matches[0]


def link_assets(source_dir: Path, doc_id: str, body: str) -> str:
    images = source_dir / "images"
    if not images.is_dir():
        return body
    target_dir = DATA_DIR / "markdown_raw/assets" / doc_id
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in images.iterdir():
        if not source.is_file():
            continue
        target = target_dir / source.name
        if target.exists():
            continue
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
    return IMAGE_RE.sub(lambda match: f"{match.group('prefix')}../markdown_raw/assets/{doc_id}/{match.group('name')}", body)


def wrap_mineru_rows(rows: list[dict[str, str]]) -> int:
    count = 0
    for row in rows:
        if row["route"] != "mineru":
            continue
        source_md = locate_mineru_markdown(row["id"])
        _source_metadata, body = parse_front_matter(source_md.read_text(encoding="utf-8", errors="replace"))
        body = link_assets(source_md.parent, row["id"], body)
        metadata = {
            "id": row["id"],
            "title": row["title"],
            "publication_date": row["publication_date"],
            "source_institution": row["source_institution"],
            "source_file": row["source_file"],
            "document_kind": row["document_kind"],
            "source_pdf_text_quality": row["pdf_text_quality"],
            "source_pdf_needs_ocr": "true",
            "source_pdf_is_scanned": "true" if row["pdf_text_quality"] == "poor" else "false",
            "pdf_page_count": row["pdf_page_count"],
            "pdf_text_quality": row["pdf_text_quality"],
            "pdf_needs_ocr": "false",
            "pdf_is_scanned": "false",
            "ocr_engine": "mineru_pipeline",
            "ocr_applied": "true",
            "ocr_status": "applied",
            "ocr_error": "",
        }
        target = DATA_DIR / "markdown_raw" / f"{row['id']}.md"
        target.write_text(dump_front_matter(metadata, helper_ensure_title_heading(body, row["title"])), encoding="utf-8", newline="\n")
        count += 1
    return count


def count_jsonl(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def validate(expected: int | None = None) -> dict[str, Any]:
    documents_path = DATA_DIR / "documents.jsonl"
    documents = [json.loads(line) for line in documents_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts = {
        "documents": len(documents),
        "document_cards": count_jsonl(DATA_DIR / "document_cards.jsonl"),
        "document_views": count_jsonl(DATA_DIR / "document_views.jsonl"),
        "section_files": len(list((DATA_DIR / "sections").glob("*.jsonl"))),
        "chunk_files": len([path for path in (DATA_DIR / "chunks").glob("*.jsonl") if path.name != "all_chunks.jsonl"]),
        "chunks": count_jsonl(DATA_DIR / "chunks/all_chunks.jsonl"),
    }
    if expected is not None and counts["documents"] > expected:
        raise RuntimeError(f"Unexpected document count: {counts['documents']} > {expected}")
    for key in ("document_cards", "section_files", "chunk_files"):
        if counts[key] != counts["documents"]:
            raise RuntimeError(f"Count mismatch: {key}={counts[key]}, documents={counts['documents']}")
    if not counts["chunks"]:
        raise RuntimeError("No chunks generated")
    kinds = Counter(str(row.get("document_kind")) for row in documents)
    if set(kinds) - {"guideline", "consensus"}:
        raise RuntimeError(f"Unexpected document_kind values: {dict(kinds)}")
    counts["document_kinds"] = dict(kinds)
    return counts


def finalize() -> dict[str, Any]:
    rows = read_csv(DATA_DIR / "processing_inventory.csv")
    wrapped = wrap_mineru_rows(rows)
    clean = clean_all(DATA_DIR / "markdown_raw", DATA_DIR / "markdown_clean", DATA_DIR / "documents.jsonl", progress_every=50)
    sections = encode_all(DATA_DIR / "markdown_clean", DATA_DIR / "sections")
    representations = build_document_representations(DATA_DIR / "markdown_clean", DATA_DIR)
    chunks = chunk_all(DATA_DIR / "markdown_clean", DATA_DIR / "chunks")
    shutil.copyfile(DATA_DIR / "document_cards.jsonl", DATA_DIR / "document_card.jsonl")
    shutil.copyfile(DATA_DIR / "document_views.jsonl", DATA_DIR / "document_view.jsonl")
    validation = validate(expected=len(rows))
    summary = {
        "mineru_wrapped": wrapped,
        "clean": clean,
        "sections": sections,
        "document_representations": representations,
        "chunks": chunks,
        "validation": validation,
    }
    (DATA_DIR / "processing_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build evidence_last from audited guideline and consensus PDFs")
    parser.add_argument("phase", choices=["prepare", "finalize", "validate"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.phase == "prepare":
        result = prepare(args.workers)
    elif args.phase == "finalize":
        result = finalize()
    else:
        result = validate()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
