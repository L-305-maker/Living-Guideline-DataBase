"""Small JSONL and metadata helpers for chunk build scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from src.guideline_chunking.models import DocumentMeta


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    resolved = ensure_parent(path)
    with resolved.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def append_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    resolved = ensure_parent(path)
    with resolved.open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_document_meta(md_path: Path, metadata_dir: str | Path | None = None) -> DocumentMeta:
    payload: dict[str, Any] = {}
    if metadata_dir:
        meta_path = Path(metadata_dir) / f"{md_path.stem}.json"
        if meta_path.exists():
            payload = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    return DocumentMeta(
        doc_id=str(payload.get("doc_id") or md_path.stem),
        title=payload.get("title"),
        publisher=payload.get("publisher"),
        language=payload.get("language"),
        document_type=payload.get("document_type"),
        publication_date=payload.get("publication_date"),
        last_updated_date=payload.get("last_updated_date"),
        source_url=payload.get("source_url"),
    )


def group_by_doc(records: Iterable[Any]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = {}
    for record in records:
        grouped.setdefault(record.doc_id, []).append(record)
    return grouped


def default_warnings_path(output_path: str | Path) -> Path:
    return Path(output_path).parent / "chunk_build_warnings.jsonl"
