"""Backfill clinical department metadata into Markdown and JSONL artifacts."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from src.utils.clinical_department import classify_clinical_department
from src.utils.front_matter import dump_front_matter, parse_front_matter
from src.utils.io import DATA_DIR, read_jsonl
from src.utils.metadata import extract_abstract


def _write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> int:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp, path)
    return len(records)


def _rewrite_jsonl_stream(path: Path, department_by_doc: dict[str, str]) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    tmp = path.with_name(path.name + ".tmp")
    total = 0
    changed = 0
    with path.open("r", encoding="utf-8") as src, tmp.open("w", encoding="utf-8", newline="\n") as dst:
        for line in src:
            if not line.strip():
                continue
            rec = json.loads(line)
            department = department_by_doc.get(rec.get("doc_id", ""), rec.get("clinical_department") or "未分类")
            if rec.get("clinical_department") != department:
                rec["clinical_department"] = department
                changed += 1
            dst.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
            total += 1
    os.replace(tmp, path)
    return total, changed


def _read_front_matter_metadata(path: Path) -> dict[str, str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        first = handle.readline()
        if first.strip() != "---":
            return {}
        for line in handle:
            if line.strip() == "---":
                break
            lines.append(line)
    metadata, _body = parse_front_matter("---\n" + "".join(lines) + "---\n")
    return metadata


def load_existing_departments(markdown_dirs: list[Path]) -> dict[str, str]:
    department_by_doc: dict[str, str] = {}
    for directory in markdown_dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.md")):
            metadata = _read_front_matter_metadata(path)
            doc_id = metadata.get("id")
            department = metadata.get("clinical_department")
            if doc_id and department:
                department_by_doc[doc_id] = department
    return department_by_doc


def classify_markdown(
    path: Path,
    force: bool = True,
    seed_departments: dict[str, str] | None = None,
) -> tuple[str | None, str | None, bool]:
    fast_metadata = _read_front_matter_metadata(path)
    doc_id = fast_metadata.get("id")
    if not doc_id:
        return None, None, False
    existing = fast_metadata.get("clinical_department")
    seeded = (seed_departments or {}).get(doc_id)
    if existing and not force:
        return doc_id, existing, False
    if seeded and existing == seeded:
        return doc_id, seeded, False

    markdown = path.read_text(encoding="utf-8", errors="replace")
    metadata, body = parse_front_matter(markdown)
    doc_id = metadata.get("id")
    if not doc_id:
        return None, None, False
    existing = metadata.get("clinical_department")
    if existing and not force:
        department = existing
    elif seeded:
        department = seeded
    else:
        department = classify_clinical_department(
            metadata.get("title", ""),
            extract_abstract(body),
            body,
        )
    changed = metadata.get("clinical_department") != department
    if changed:
        metadata["clinical_department"] = department
        _write_text_atomic(path, dump_front_matter(metadata, body))
    return doc_id, department, changed


def update_markdown_dirs(
    markdown_dirs: list[Path],
    force: bool = True,
    seed_departments: dict[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    department_by_doc: dict[str, str] = {}
    stats: dict[str, Any] = {"markdown_files": 0, "markdown_changed": 0, "markdown_skipped": 0}
    for directory in markdown_dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.md")):
            doc_id, department, changed = classify_markdown(path, force=force, seed_departments=seed_departments)
            if not doc_id or not department:
                stats["markdown_skipped"] += 1
                continue
            department_by_doc[doc_id] = department
            stats["markdown_files"] += 1
            if changed:
                stats["markdown_changed"] += 1
    return department_by_doc, stats


def update_manifest(path: Path, department_by_doc: dict[str, str]) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    records: list[dict[str, Any]] = []
    changed = 0
    for rec in read_jsonl(path):
        department = department_by_doc.get(rec.get("doc_id", ""), rec.get("clinical_department") or "未分类")
        if rec.get("clinical_department") != department:
            rec["clinical_department"] = department
            changed += 1
        records.append(rec)
    _write_jsonl_atomic(path, records)
    return len(records), changed


def update_partitioned_jsonl(directory: Path, department_by_doc: dict[str, str]) -> tuple[int, int, int]:
    if not directory.exists():
        return 0, 0, 0
    files = 0
    total = 0
    changed = 0
    for path in sorted(directory.glob("*.jsonl")):
        if path.name == "all_chunks.jsonl":
            continue
        rows, updates = _rewrite_jsonl_stream(path, department_by_doc)
        files += 1
        total += rows
        changed += updates
    return files, total, changed


def backfill(data_dir: str | Path = DATA_DIR, force: bool = True, reuse_existing: bool = False) -> dict[str, Any]:
    data_path = Path(data_dir)
    markdown_dirs = [data_path / "markdown_raw", data_path / "markdown_clean"]
    seed_departments = load_existing_departments(markdown_dirs) if reuse_existing else {}
    department_by_doc, stats = update_markdown_dirs(markdown_dirs, force=force, seed_departments=seed_departments)

    manifest_total, manifest_changed = update_manifest(data_path / "documents.jsonl", department_by_doc)
    raw_manifest_total, raw_manifest_changed = update_manifest(data_path / "documents_raw.jsonl", department_by_doc)
    section_files, section_rows, section_changed = update_partitioned_jsonl(data_path / "sections", department_by_doc)
    chunk_files, chunk_rows, chunk_changed = update_partitioned_jsonl(data_path / "chunks", department_by_doc)
    all_chunks_pending_replace = False
    try:
        all_chunk_rows, all_chunk_changed = _rewrite_jsonl_stream(data_path / "chunks" / "all_chunks.jsonl", department_by_doc)
    except PermissionError:
        all_chunks_pending_replace = True
        tmp_path = data_path / "chunks" / "all_chunks.jsonl.tmp"
        all_chunk_rows = 0
        all_chunk_changed = 0
        if tmp_path.exists():
            with tmp_path.open("r", encoding="utf-8") as handle:
                all_chunk_rows = sum(1 for line in handle if line.strip())

    stats.update(
        {
            "documents": len(department_by_doc),
            "documents_manifest_rows": manifest_total,
            "documents_manifest_changed": manifest_changed,
            "raw_documents_manifest_rows": raw_manifest_total,
            "raw_documents_manifest_changed": raw_manifest_changed,
            "section_files": section_files,
            "section_rows": section_rows,
            "section_changed": section_changed,
            "chunk_files": chunk_files,
            "chunk_rows": chunk_rows,
            "chunk_changed": chunk_changed,
            "all_chunk_rows": all_chunk_rows,
            "all_chunk_changed": all_chunk_changed,
            "all_chunks_pending_replace": all_chunks_pending_replace,
            "department_distribution": dict(Counter(department_by_doc.values()).most_common()),
        }
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--no-force", action="store_true", help="Keep existing clinical_department values.")
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse existing Markdown classifications by doc_id.")
    args = parser.parse_args()
    print(json.dumps(backfill(args.data_dir, force=not args.no_force, reuse_existing=args.reuse_existing), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
