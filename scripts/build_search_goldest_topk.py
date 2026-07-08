from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_retrieve_goldest_topk import QUESTIONS, normalize_retrieval_query
from src.retrieval.sqlite_store import DEFAULT_DB_PATH, search_documents_sqlite
from src.utils.io import DATA_DIR, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 50 search goldest questions with top-k documents.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "goldest" / "search_goldest.json")
    parser.add_argument("--jsonl-output", type=Path, default=DATA_DIR / "goldest" / "search_goldest.jsonl")
    parser.add_argument("--report-output", type=Path, default=DATA_DIR / "goldest" / "search_goldest_report.json")
    parser.add_argument("--topk", type=int, default=10)
    return parser.parse_args()


def load_documents(data_dir: Path) -> dict[str, dict[str, Any]]:
    return {row["doc_id"]: row for row in read_jsonl(data_dir / "documents.jsonl")}


def text_preview(text: str, max_chars: int = 500) -> str:
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def document_payload(rank: int, document: dict[str, Any], source: dict[str, Any] | None) -> dict[str, Any]:
    source = source or {}
    abstract = document.get("abstract") or source.get("abstract") or ""
    return {
        "rank": rank,
        "doc_id": document["doc_id"],
        "title": document.get("title", ""),
        "score": document.get("score"),
        "publication_date": document.get("publication_date", "unknown"),
        "source_institution": document.get("source_institution", ""),
        "clinical_department": document.get("clinical_department", ""),
        "source_file": source.get("source_file", ""),
        "markdown_clean_path": source.get("markdown_clean_path", ""),
        "cleaning_quality": document.get("cleaning_quality", ""),
        "cleaning_flags": document.get("cleaning_flags", ""),
        "pdf_text_quality": document.get("pdf_text_quality", ""),
        "pdf_needs_ocr": document.get("pdf_needs_ocr", False),
        "pdf_is_scanned": document.get("pdf_is_scanned", False),
        "abstract": abstract,
        "abstract_preview": text_preview(abstract),
    }


def build_rows(data_dir: Path, db_path: Path, topk: int) -> list[dict[str, Any]]:
    documents_by_id = load_documents(data_dir)
    rows: list[dict[str, Any]] = []
    for index, seed in enumerate(QUESTIONS, start=1):
        retrieval_query = normalize_retrieval_query(seed["question"])
        results = search_documents_sqlite(
            retrieval_query,
            db_path=db_path,
            clinical_department=seed["clinical_department"],
            topk=topk,
        )
        rows.append(
            {
                "query_id": f"search_goldest_{index:03d}",
                "task": "search",
                "question": seed["question"],
                "query": seed["question"],
                "retrieval_query": retrieval_query,
                "clinical_department": seed["clinical_department"],
                "topk": topk,
                "retrieval_backend": "sqlite_fts_rrf",
                "top_documents": [
                    document_payload(rank, document, documents_by_id.get(document["doc_id"]))
                    for rank, document in enumerate(results, start=1)
                ],
            }
        )
    return rows


def validate(rows: list[dict[str, Any]], topk: int) -> dict[str, Any]:
    errors: list[str] = []
    if len(rows) != 50:
        errors.append(f"expected 50 questions, got {len(rows)}")
    ids = [row["query_id"] for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate query_id values")
    department_counts = Counter(row["clinical_department"] for row in rows)
    for row in rows:
        documents = row["top_documents"]
        if len(documents) != topk:
            errors.append(f"{row['query_id']} expected {topk} documents, got {len(documents)}")
        doc_ids = [document["doc_id"] for document in documents]
        if len(doc_ids) != len(set(doc_ids)):
            errors.append(f"{row['query_id']} has duplicate document ids")
        mismatched = [
            document["doc_id"]
            for document in documents
            if document["clinical_department"] != row["clinical_department"]
        ]
        if mismatched:
            errors.append(f"{row['query_id']} has documents outside department filter: {mismatched[:3]}")
    return {
        "ok": not errors,
        "errors": errors,
        "question_count": len(rows),
        "topk": topk,
        "department_counts": dict(sorted(department_counts.items())),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    rows = build_rows(args.data_dir, args.db_path, args.topk)
    validation = validate(rows, args.topk)
    report = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "db_path": str(args.db_path),
        "output": str(args.output),
        "jsonl_output": str(args.jsonl_output),
        "validation": validation,
    }
    if not validation["ok"]:
        raise SystemExit(json.dumps(report, ensure_ascii=False, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    write_jsonl(args.jsonl_output, rows)
    args.report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
