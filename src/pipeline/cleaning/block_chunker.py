"""Chunk complete source blocks into smaller retrieval units."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.pipeline.cleaning.chunker import chunk_all as _chunk_all
from src.pipeline.cleaning.chunker import chunk_file, chunk_markdown, estimate_tokens
from src.pipeline.cleaning.chunker import split_section_content as split_block_content


def chunk_blocks(
    markdown_clean_dir: str | Path,
    chunks_dir: str | Path,
) -> dict[str, Any]:
    """Split each complete block into smaller traceable chunks."""

    result = _chunk_all(markdown_clean_dir, chunks_dir)
    return {
        "documents": result.get("documents", 0),
        "chunks": result.get("chunks", 0),
        "chunks_dir": str(chunks_dir),
    }


chunk_all = chunk_blocks
