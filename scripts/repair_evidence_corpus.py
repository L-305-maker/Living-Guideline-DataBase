"""Repair stale evidence records and make every source PDF disposition explicit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.schemas import DocumentRecord, dump_model
from src.pipeline.cleaning.cleaner import clean_all
from src.pipeline.cleaning.encoder import encode_all
from src.pipeline.cleaning.chunker import chunk_all
from src.pipeline.cleaning.pdf_to_md import helper_convert_pdf_worker, helper_deduplicate_doc_id
from src.retrieval.document_repr import build_document_representations
from src.retrieval.sqlite_store import build_sqlite_store
from src.utils.front_matter import parse_front_matter
from src.utils.io import read_jsonl, write_jsonl


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def path_key(value: str | Path) -> str:
    return str(resolve_path(value)).casefold()


def pdf_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_pdfs(raw_pdf_dir: Path, consensus_pdf_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for root in (raw_pdf_dir, consensus_pdf_dir):
        if root.exists():
            paths.extend(
                path.resolve()
                for path in root.rglob("*.pdf")
                if "quarantine" not in {part.casefold() for part in path.parts}
            )
    return sorted(paths)


def load_review(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, list[dict[str, str]]]]:
    exact: dict[str, dict[str, str]] = {}
    by_name: dict[str, list[dict[str, str]]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            exact[path_key(row["file"])] = row
            by_name.setdefault(Path(row["file"]).name.casefold(), []).append(row)
    return exact, by_name


def review_for(
    path: Path,
    exact: dict[str, dict[str, str]],
    by_name: dict[str, list[dict[str, str]]],
) -> tuple[dict[str, str] | None, str]:
    if row := exact.get(path_key(path)):
        return row, "exact_review_path"
    rows = by_name.get(path.name.casefold(), [])
    if len(rows) == 1:
        return rows[0], "unique_basename_review"
    return None, "no_prior_review"


def document_sources(data_dir: Path) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    documents = list(read_jsonl(data_dir / "documents.jsonl"))
    sources: dict[str, list[str]] = {}
    for row in documents:
        source = str(row.get("source_file") or "")
        if source and resolve_path(source).is_file():
            sources.setdefault(path_key(source), []).append(str(row["doc_id"]))
    return documents, sources


def audit(
    pdfs: list[Path],
    sources: dict[str, list[str]],
    exact_review: dict[str, dict[str, str]],
    by_name_review: dict[str, list[dict[str, str]]],
) -> tuple[list[dict[str, Any]], dict[Path, str]]:
    hashes = {path: pdf_hash(path) for path in pdfs}
    active_hashes: dict[str, tuple[Path, str]] = {}
    for path in pdfs:
        doc_ids = sources.get(path_key(path), [])
        if doc_ids:
            active_hashes.setdefault(hashes[path], (path, doc_ids[0]))

    rows: list[dict[str, Any]] = []
    confirmed: dict[Path, str] = {}
    for path in pdfs:
        doc_ids = sources.get(path_key(path), [])
        review, decision_source = review_for(path, exact_review, by_name_review)
        review_status = review.get("guideline_status", "") if review else ""
        row: dict[str, Any] = {
            "source_file": str(path),
            "sha256": hashes[path],
            "review_status": review_status or "not_reviewed",
            "decision_source": decision_source,
        }
        if doc_ids:
            row.update(status="active", doc_ids=doc_ids, intentional_skip=False)
        elif hashes[path] in active_hashes:
            canonical_path, canonical_doc_id = active_hashes[hashes[path]]
            row.update(
                status="duplicate_pdf_content",
                canonical_source_file=str(canonical_path),
                canonical_doc_id=canonical_doc_id,
                intentional_skip=True,
            )
        elif review_status == "confirmed_signal":
            row.update(status="confirmed_unprocessed", intentional_skip=False)
            confirmed[path] = decision_source
        elif review_status == "clear_non_guideline":
            row.update(status="excluded_clear_non_guideline", intentional_skip=True)
        elif review_status == "needs_content_review":
            row.update(status="review_required", intentional_skip=True)
        else:
            row.update(status="review_required_no_prior_review", intentional_skip=True)
        rows.append(row)
    return rows, confirmed


def convert_confirmed(
    paths: list[Path],
    data_dir: Path,
    consensus_pdf_dir: Path,
    workers: int,
    ocr_mode: str,
    run_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if not paths:
        return [], []
    staging = data_dir / f".repair_markdown_raw_{run_id}"
    staging.mkdir(parents=True, exist_ok=True)
    payloads = [
        (
            str(path),
            str(staging),
            ocr_mode,
            str(data_dir / "ocr_pdf"),
            "chi_sim+eng",
            "consensus" if consensus_pdf_dir in path.parents else "guideline",
        )
        for path in paths
    ]
    converted: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(helper_convert_pdf_worker, payload) for payload in payloads]
        for done, future in enumerate(as_completed(futures), start=1):
            record, error = future.result()
            if record:
                converted.append(record)
            if error:
                errors.append(error)
            if done == len(futures) or done % 100 == 0:
                print(json.dumps({"converted": done, "total": len(futures)}, ensure_ascii=False), flush=True)

    raw_dir = data_dir / "markdown_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    seen_ids: set[str] = set()
    for raw_path in raw_dir.glob("*.md"):
        metadata, _body = parse_front_matter(raw_path.read_text(encoding="utf-8", errors="replace"))
        if metadata.get("id"):
            seen_ids.add(str(metadata["id"]))

    finalized: list[dict[str, Any]] = []
    for row in sorted(converted, key=lambda item: str(item.get("source_file") or "")):
        record = helper_deduplicate_doc_id(DocumentRecord(**row), seen_ids)
        staged_path = Path(record.markdown_raw_path)
        destination = raw_dir / staged_path.name
        if destination.exists():
            raise RuntimeError(f"refusing to overwrite existing raw Markdown: {destination}")
        shutil.move(str(staged_path), destination)
        record.markdown_raw_path = str(destination)
        finalized.append(dump_model(record))
    shutil.rmtree(staging, ignore_errors=True)
    return finalized, errors


def quarantine_excluded(data_dir: Path, run_id: str) -> tuple[Path, int]:
    quarantine = data_dir / "quarantine" / f"corpus_repair_{run_id}" / "markdown_raw"
    moved = 0
    for row in read_jsonl(data_dir / "documents_excluded.jsonl"):
        raw_path = resolve_path(str(row.get("markdown_raw_path") or ""))
        expected_root = (data_dir / "markdown_raw").resolve()
        if raw_path.is_file() and expected_root in raw_path.parents:
            quarantine.mkdir(parents=True, exist_ok=True)
            destination = quarantine / raw_path.name
            if destination.exists():
                raise RuntimeError(f"quarantine collision: {destination}")
            shutil.move(str(raw_path), destination)
            moved += 1
    return quarantine, moved


def sync_raw_manifest(data_dir: Path, converted: list[dict[str, Any]]) -> dict[str, int]:
    raw_manifest = data_dir / "documents_raw.jsonl"
    existing = list(read_jsonl(raw_manifest)) if raw_manifest.exists() else []
    active = {str(row["doc_id"]): row for row in read_jsonl(data_dir / "documents.jsonl")}
    candidates = {str(row.get("doc_id")): row for row in [*existing, *converted] if row.get("doc_id")}
    missing = sorted(set(active) - set(candidates))
    for doc_id in missing:
        candidates[doc_id] = active[doc_id]
    rows = [candidates[doc_id] for doc_id in sorted(active)]
    write_jsonl(raw_manifest, rows)
    return {"records": len(rows), "fallback_records": len(missing)}


def finalize_inventory(
    rows: list[dict[str, Any]],
    data_dir: Path,
    conversion_errors: list[dict[str, str]],
) -> list[dict[str, Any]]:
    _documents, sources = document_sources(data_dir)
    excluded = {
        path_key(row["source_file"]): row
        for row in read_jsonl(data_dir / "documents_excluded.jsonl")
        if row.get("source_file")
    }
    errors = {path_key(row["source_file"]): row["error"] for row in conversion_errors}
    for row in rows:
        key = path_key(row["source_file"])
        if doc_ids := sources.get(key):
            row.update(status="active", doc_ids=doc_ids, intentional_skip=False)
        elif key in excluded:
            exclusion = excluded[key]
            row.update(
                status=str(exclusion["reason"]),
                canonical_doc_id=str(exclusion.get("canonical_doc_id") or ""),
                intentional_skip=True,
            )
        elif key in errors:
            row.update(status="conversion_failed", error=errors[key], intentional_skip=False)
        elif row["status"] == "confirmed_unprocessed":
            row.update(status="confirmed_still_unprocessed", intentional_skip=False)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/evidence"))
    parser.add_argument("--raw-pdf-dir", type=Path, default=Path("data/raw_pdf"))
    parser.add_argument("--consensus-pdf-dir", type=Path, default=Path("data/consensus"))
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=Path("data/reports/raw_pdf_source_department_20260714/final/all_pdf_source_department.csv"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ocr-mode", choices=["never", "auto", "force"], default="never")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    data_dir = resolve_path(args.data_dir)
    raw_pdf_dir = resolve_path(args.raw_pdf_dir)
    consensus_pdf_dir = resolve_path(args.consensus_pdf_dir)
    review_csv = resolve_path(args.review_csv)
    pdfs = current_pdfs(raw_pdf_dir, consensus_pdf_dir)
    _documents, sources = document_sources(data_dir)
    exact_review, by_name_review = load_review(review_csv)
    inventory, confirmed = audit(pdfs, sources, exact_review, by_name_review)
    before = dict(Counter(str(row["status"]) for row in inventory))
    preview = {"apply": args.apply, "pdfs": len(pdfs), "status_before": before, "to_convert": len(confirmed)}
    print(json.dumps(preview, ensure_ascii=False, indent=2), flush=True)
    if not args.apply:
        return

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    converted, conversion_errors = convert_confirmed(
        sorted(confirmed), data_dir, consensus_pdf_dir, args.workers, args.ocr_mode, run_id
    )
    stages: dict[str, Any] = {}
    stages["clean"] = clean_all(
        data_dir / "markdown_raw", data_dir / "markdown_clean", data_dir / "documents.jsonl", progress_every=500
    )
    quarantine, moved = quarantine_excluded(data_dir, run_id)
    stages["quarantine"] = {"moved_raw_markdown": moved, "path": str(quarantine)}
    stages["raw_manifest"] = sync_raw_manifest(data_dir, converted)
    stages["sections"] = encode_all(data_dir / "markdown_clean", data_dir / "sections")
    stages["document_representations"] = build_document_representations(data_dir / "markdown_clean", data_dir)
    stages["chunks"] = chunk_all(data_dir / "markdown_clean", data_dir / "chunks")
    stages["sqlite"] = build_sqlite_store(data_dir, data_dir / "index" / "rag.sqlite")

    inventory = finalize_inventory(inventory, data_dir, conversion_errors)
    status_after = dict(Counter(str(row["status"]) for row in inventory))
    write_jsonl(data_dir / "source_pdf_inventory.jsonl", inventory)
    summary = {
        "run_id": run_id,
        "pdfs": len(pdfs),
        "status_before": before,
        "status_after": status_after,
        "converted": len(converted),
        "conversion_errors": len(conversion_errors),
        "ocr_mode": args.ocr_mode,
        "stages": stages,
    }
    write_json(data_dir / "source_pdf_inventory_summary.json", summary)
    write_json(data_dir / f"corpus_repair_{run_id}.json", summary)
    write_json(
        data_dir / "evidence_pipeline_manifest.json",
        {
            "pipeline_version": "evidence_cleaning_rag_v1",
            "build_reason": "corpus_repair",
            "run_id": run_id,
            "extracts_recommendations": False,
            "extracts_pico_questions": False,
            "artifacts": {
                "data_dir": str(data_dir),
                "raw_pdf_dir": str(raw_pdf_dir),
                "consensus_pdf_dir": str(consensus_pdf_dir),
                "source_pdf_inventory": str(data_dir / "source_pdf_inventory.jsonl"),
                "document_manifest": str(data_dir / "documents.jsonl"),
                "sqlite_db": str(data_dir / "index" / "rag.sqlite"),
            },
            "source_status": status_after,
            "stages": stages,
            "vector": {"skipped": True, "reason": "no vector index exists in the active SQLite-only build"},
        },
    )
    unresolved = status_after.get("confirmed_still_unprocessed", 0) + status_after.get("conversion_failed", 0)
    if unresolved:
        raise RuntimeError(f"repair left {unresolved} confirmed PDFs unresolved")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
