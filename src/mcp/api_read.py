"""Read full clean Markdown by doc_id."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.retrieval.sqlite_store import DEFAULT_DB_PATH, read_document_sqlite
from src.models.schemas import ReadInput
from src.utils.front_matter import parse_front_matter
from src.utils.io import DATA_DIR, iter_markdown_files


def helper_validate(payload: dict[str, Any] | ReadInput) -> ReadInput:
    return payload if isinstance(payload, ReadInput) else ReadInput(**payload)


def read(payload: dict[str, Any] | ReadInput, data_dir: str | Path = DATA_DIR) -> dict[str, Any]:
    request = helper_validate(payload)
    sqlite_path = Path(data_dir) / "index" / DEFAULT_DB_PATH.name
    if sqlite_path.exists():
        return read_document_sqlite(
            doc_id=request.doc_id,
            title=request.title,
            db_path=sqlite_path,
            max_chars=request.max_chars,
        )
    if not request.doc_id and not request.title:
        raise ValueError("read requires either doc_id or title")
    title_query = (request.title or "").strip().casefold()
    for path in iter_markdown_files(Path(data_dir) / "markdown_clean"):
        markdown = path.read_text(encoding="utf-8", errors="replace")
        metadata, _body = parse_front_matter(markdown)
        title = metadata.get("title", "")
        if metadata.get("id") == request.doc_id or (title_query and title_query in title.casefold()):
            if request.max_chars and request.max_chars > 0:
                markdown = markdown[: request.max_chars]
            return {
                "doc_id": metadata.get("id", request.doc_id or ""),
                "title": title,
                "publication_date": metadata.get("publication_date", "unknown"),
                "source_institution": metadata.get("source_institution", "Unknown"),
                "clinical_department": metadata.get("clinical_department", "未分类"),
                "source_file": metadata.get("source_file", ""),
                "markdown_clean_path": str(path),
                "content": markdown,
            }
    raise KeyError(f"Document not found: {request.doc_id or request.title}")
