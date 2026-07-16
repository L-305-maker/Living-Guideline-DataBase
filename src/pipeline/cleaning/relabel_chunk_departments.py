"""Refresh chunk department labels without changing chunk boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.utils.clinical_department import UNKNOWN_DEPARTMENT, classify_chunk_departments
from src.utils.io import DATA_DIR, read_jsonl


def load_document_departments(path: str | Path) -> dict[str, list[str]]:
    departments: dict[str, list[str]] = {}
    for record in read_jsonl(path):
        labels = [str(item).strip() for item in record.get("clinical_departments") or [] if str(item).strip()]
        if not labels:
            labels = [str(record.get("clinical_department") or UNKNOWN_DEPARTMENT)]
        departments[str(record["doc_id"])] = labels
    return departments


def relabel_chunk(record: dict[str, Any], parent: list[str], force: bool = False) -> tuple[dict[str, Any], bool, bool]:
    current = [str(item).strip() for item in record.get("clinical_departments") or [] if str(item).strip()]
    fallback = current == [UNKNOWN_DEPARTMENT] and UNKNOWN_DEPARTMENT not in parent
    if force:
        result = classify_chunk_departments(
            record.get("section_path") or [],
            str(record.get("content") or ""),
            parent,
            str(record.get("chunk_type") or ""),
        )
        labels = list(result["clinical_departments"])
    elif fallback:
        labels = [parent[0]]
    elif current and set(current).issubset(parent):
        labels = current
    else:
        result = classify_chunk_departments(
            record.get("section_path") or [],
            str(record.get("content") or ""),
            parent,
            str(record.get("chunk_type") or ""),
        )
        labels = list(result["clinical_departments"])

    changed = current != labels or "clinical_department" in record or "department_scope" in record
    record = dict(record)
    record["clinical_departments"] = labels
    record.pop("clinical_department", None)
    record.pop("department_scope", None)
    return record, changed, fallback

def relabel_all(
    document_manifest: str | Path = DATA_DIR / "documents.jsonl",
    chunks_dir: str | Path = DATA_DIR / "chunks",
    force_documents: set[str] | None = None,
) -> dict[str, int | str]:
    documents = load_document_departments(document_manifest)
    directory = Path(chunks_dir)
    chunk_files = sorted(path for path in directory.glob("*.jsonl") if path.name != "all_chunks.jsonl")
    aggregate_path = directory / "all_chunks.jsonl"
    aggregate_tmp = directory / "all_chunks.jsonl.tmp"
    chunks = changed = fallback = files = 0
    documents_with_chunks: set[str] = set()

    try:
        with aggregate_tmp.open("w", encoding="utf-8", newline="\n") as aggregate:
            for path in chunk_files:
                file_tmp = path.with_suffix(path.suffix + ".tmp")
                try:
                    with path.open("r", encoding="utf-8-sig") as source, file_tmp.open(
                        "w", encoding="utf-8", newline="\n"
                    ) as target:
                        for line_no, line in enumerate(source, start=1):
                            if not line.strip():
                                continue
                            try:
                                record = json.loads(line)
                            except json.JSONDecodeError as exc:
                                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
                            doc_id = str(record.get("doc_id") or "")
                            if doc_id not in documents:
                                raise KeyError(f"Chunk {record.get('chunk_id')} has no document metadata")
                            record, was_changed, used_fallback = relabel_chunk(
                                record, documents[doc_id], doc_id in (force_documents or set())
                            )
                            labels = set(record["clinical_departments"])
                            if not labels.issubset(documents[doc_id]):
                                raise ValueError(f"Chunk {record.get('chunk_id')} has labels outside its document")
                            encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                            target.write(encoded)
                            aggregate.write(encoded)
                            chunks += 1
                            changed += was_changed
                            fallback += used_fallback
                            documents_with_chunks.add(doc_id)
                    file_tmp.replace(path)
                except Exception:
                    file_tmp.unlink(missing_ok=True)
                    raise
                files += 1
                if files % 500 == 0:
                    print(json.dumps({"files": files, "chunks": chunks, "changed": changed}), flush=True)
        aggregate_tmp.replace(aggregate_path)
    except Exception:
        aggregate_tmp.unlink(missing_ok=True)
        raise

    return {
        "documents": len(documents),
        "documents_with_chunks": len(documents_with_chunks),
        "chunk_files": files,
        "chunks": chunks,
        "changed_chunks": changed,
        "fallback_to_primary": fallback,
        "forced_documents": len(force_documents or set()),
        "chunks_dir": str(directory),
    }


def load_force_documents(path: str | Path | None) -> set[str]:
    if not path:
        return set()
    return {
        str(record["doc_id"])
        for record in read_jsonl(path)
        if record.get("status") == "identified"
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", default=str(DATA_DIR / "documents.jsonl"))
    parser.add_argument("--chunks-dir", default=str(DATA_DIR / "chunks"))
    parser.add_argument("--force-documents", help="Reclassification report JSONL; identified doc_ids are forced.")
    args = parser.parse_args()
    force_documents = load_force_documents(args.force_documents)
    print(json.dumps(relabel_all(args.documents, args.chunks_dir, force_documents), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
