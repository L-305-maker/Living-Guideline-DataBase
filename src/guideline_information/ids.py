"""Stable IDs for source blocks and downstream records."""

from __future__ import annotations

import hashlib
import json
from typing import Any


ID_ALGORITHM_VERSION = "gi_id_v1"


def stable_hash(*parts: Any, length: int = 16) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def make_source_block_key(
    doc_id: str,
    section_path: list[str],
    char_start: int,
    char_end: int,
    *,
    algorithm_version: str = ID_ALGORITHM_VERSION,
) -> str:
    return f"{algorithm_version}:sb:{stable_hash(doc_id, section_path, char_start, char_end, length=24)}"


def make_source_revision_id(content: str, *, algorithm_version: str = ID_ALGORITHM_VERSION) -> str:
    digest = hashlib.sha256((content or "").encode("utf-8")).hexdigest()
    return f"{algorithm_version}:rev:{digest}"


def make_candidate_id(
    source_block_key: str,
    source_revision_id: str,
    candidate_text: str,
    *,
    algorithm_version: str = ID_ALGORITHM_VERSION,
) -> str:
    return f"{algorithm_version}:cand:{stable_hash(source_block_key, source_revision_id, candidate_text, length=24)}"


def make_record_id(prefix: str, *parts: Any, algorithm_version: str = ID_ALGORITHM_VERSION) -> str:
    return f"{algorithm_version}:{prefix}:{stable_hash(*parts, length=24)}"
