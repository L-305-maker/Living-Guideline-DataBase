"""Recompute coarse source institution and doc IDs for markdown_raw.

This script does not re-run PDF extraction. It rebuilds the raw Markdown
front matter from existing markdown_raw bodies, then downstream pipeline stages
can regenerate documents/cards/views/sections/chunks from the new IDs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.schemas import DocumentRecord, dump_model
from src.utils.front_matter import dump_front_matter, parse_front_matter
from src.utils.ids import make_doc_id, sha256_file, sha256_text
from src.utils.io import DATA_DIR, iter_markdown_files
from src.utils.metadata import extract_abstract, extract_publication_date, extract_source_institution


def helper_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() == "true"


def helper_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if not value:
        return []
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
    return [part.strip() for part in str(value).split("|") if part.strip()]


def helper_file_sha(source_file: str, fallback: str) -> str:
    path = Path(source_file)
    if path.is_file():
        return sha256_file(path)
    return sha256_text(fallback)


def helper_unique_doc_id(base_id: str, source_file: str, old_path: Path, seen: set[str]) -> str:
    if base_id not in seen:
        seen.add(base_id)
        return base_id
    suffix = hashlib.sha1(f"{source_file}|{old_path}".encode("utf-8")).hexdigest()[:8]
    doc_id = f"{base_id}_{suffix}"
    while doc_id in seen:
        suffix = hashlib.sha1(f"{doc_id}|{source_file}|{old_path}".encode("utf-8")).hexdigest()[:8]
        doc_id = f"{base_id}_{suffix}"
    seen.add(doc_id)
    return doc_id


def helper_record(metadata: dict[str, Any], body: str, raw_path: Path) -> dict[str, Any]:
    abstract = extract_abstract(body)
    departments = helper_list(metadata.get("clinical_departments"))
    clinical_department = str(metadata.get("clinical_department") or (departments[0] if departments else "???"))
    return dump_model(
        DocumentRecord(
            doc_id=str(metadata["id"]),
            title=str(metadata.get("title") or metadata["id"]),
            publication_date=str(metadata.get("publication_date") or "unknown"),
            source_institution=str(metadata.get("source_institution") or "Unknown"),
            clinical_department=clinical_department,
            clinical_departments=departments or [clinical_department],
            department_scope=str(metadata.get("department_scope") or "single"),
            document_kind=str(metadata.get("document_kind") or "guideline"),
            source_file=str(metadata.get("source_file") or ""),
            markdown_raw_path=str(raw_path),
            markdown_clean_path="",
            abstract=abstract,
            content_sha256=sha256_text(dump_front_matter(metadata, body)),
            cleaning_quality=str(metadata.get("cleaning_quality") or ""),
            cleaning_flags=str(metadata.get("cleaning_flags") or ""),
            source_pdf_text_quality=str(metadata.get("source_pdf_text_quality") or ""),
            source_pdf_needs_ocr=helper_bool(metadata.get("source_pdf_needs_ocr")),
            source_pdf_is_scanned=helper_bool(metadata.get("source_pdf_is_scanned")),
            pdf_text_quality=str(metadata.get("pdf_text_quality") or ""),
            pdf_needs_ocr=helper_bool(metadata.get("pdf_needs_ocr")),
            pdf_is_scanned=helper_bool(metadata.get("pdf_is_scanned")),
            ocr_engine=str(metadata.get("ocr_engine") or ""),
            ocr_applied=helper_bool(metadata.get("ocr_applied")),
            ocr_status=str(metadata.get("ocr_status") or ""),
            ocr_error=str(metadata.get("ocr_error") or ""),
        )
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def rebuild_raw_sources(data_dir: Path, apply: bool) -> dict[str, Any]:
    raw_dir = data_dir / "markdown_raw"
    if not raw_dir.exists():
        raise FileNotFoundError(raw_dir)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = data_dir / "reports" / f"source_rebuild_{stamp}"
    backup_dir = data_dir / "backups" / f"source_rebuild_{stamp}"
    tmp_raw_dir = data_dir / "rebuild_tmp" / f"source_rebuild_{stamp}" / "markdown_raw"
    report_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    old_source_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()

    paths = list(iter_markdown_files(raw_dir))
    for old_path in paths:
        markdown = old_path.read_text(encoding="utf-8", errors="replace")
        metadata, body = parse_front_matter(markdown)
        old_doc_id = str(metadata.get("id") or old_path.stem)
        old_source = str(metadata.get("source_institution") or "Unknown")
        title = str(metadata.get("title") or old_doc_id)
        document_kind = str(metadata.get("document_kind") or "guideline")
        source_file = str(metadata.get("source_file") or "")
        publication_date = str(metadata.get("publication_date") or "")
        if not publication_date or publication_date == "unknown":
            publication_date = extract_publication_date(source_file or old_path, body)

        new_source = extract_source_institution(source_file or old_path, body, title=title, document_kind=document_kind)
        base_doc_id = make_doc_id(new_source, publication_date, helper_file_sha(source_file, str(old_path)))
        new_doc_id = helper_unique_doc_id(base_doc_id, source_file, old_path, seen_ids)
        new_metadata = dict(metadata)
        new_metadata["id"] = new_doc_id
        new_metadata["source_institution"] = new_source
        new_metadata["publication_date"] = publication_date
        new_markdown = dump_front_matter(new_metadata, body)
        new_path = raw_dir / f"{new_doc_id}.md"
        old_source_counts[old_source] += 1
        source_counts[new_source] += 1
        changed = old_doc_id != new_doc_id or old_source != new_source or old_path.name != new_path.name
        rows.append(
            {
                "old_doc_id": old_doc_id,
                "new_doc_id": new_doc_id,
                "old_source_institution": old_source,
                "new_source_institution": new_source,
                "title": title,
                "document_kind": document_kind,
                "source_file": source_file,
                "old_markdown_raw_path": str(old_path),
                "new_markdown_raw_path": str(new_path),
                "changed": changed,
            }
        )
        records.append(helper_record(new_metadata, body, new_path))
        if apply:
            target = tmp_raw_dir / f"{new_doc_id}.md"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(new_markdown, encoding="utf-8", newline="\n")

    with (report_dir / "source_rebuild_mapping.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "old_doc_id",
            "new_doc_id",
            "old_source_institution",
            "new_source_institution",
            "title",
            "document_kind",
            "source_file",
            "old_markdown_raw_path",
            "new_markdown_raw_path",
            "changed",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if apply:
        backup_dir.mkdir(parents=True, exist_ok=True)
        documents_raw = data_dir / "documents_raw.jsonl"
        if documents_raw.exists():
            shutil.copy2(documents_raw, backup_dir / "documents_raw.jsonl")
        raw_dir.rename(backup_dir / "markdown_raw")
        tmp_raw_dir.rename(raw_dir)
        write_jsonl(documents_raw, sorted(records, key=lambda item: item.get("doc_id", "")))

    summary = {
        "applied": apply,
        "documents_seen": len(paths),
        "changed_documents": sum(1 for row in rows if row["changed"]),
        "source_changed_documents": sum(1 for row in rows if row["old_source_institution"] != row["new_source_institution"]),
        "doc_id_changed_documents": sum(1 for row in rows if row["old_doc_id"] != row["new_doc_id"]),
        "old_source_counts": old_source_counts.most_common(),
        "new_source_counts": source_counts.most_common(),
        "report_dir": str(report_dir),
        "backup_dir": str(backup_dir) if apply else "",
        "mapping_csv": str(report_dir / "source_rebuild_mapping.csv"),
    }
    (report_dir / "source_rebuild_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute coarse source_institution and doc IDs in markdown_raw.")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--apply", action="store_true", help="Write rebuilt markdown_raw and documents_raw.jsonl.")
    args = parser.parse_args()
    summary = rebuild_raw_sources(Path(args.data_dir), apply=args.apply)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
