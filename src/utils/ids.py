"""Stable IDs and hashing."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def slug(value: object | None, max_len: int = 80) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text).strip().lower()
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[\s-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return (text[:max_len].strip("_") or "unknown")


def make_doc_id(source_institution: str, publication_date: str, file_sha256: str) -> str:
    year_match = re.search(r"(19\d{2}|20\d{2})", publication_date or "")
    year = year_match.group(1) if year_match else "unknown"
    return f"{slug(source_institution)}_{year}_{file_sha256[:12]}"


def make_chunk_id(doc_id: str, chunk_index: int) -> str:
    return f"{doc_id}#chunk_{chunk_index:05d}"

