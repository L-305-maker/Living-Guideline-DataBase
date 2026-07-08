"""Block encoding for clean Markdown evidence documents.

In the evidence pipeline, a block is the complete Markdown section produced
from heading-aware parsing. The on-disk directory remains named ``sections`` so
it can be ingested by the existing SQLite/FTS builder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.pipeline.cleaning.encoder import encode_all as _encode_all
from src.pipeline.cleaning.encoder import encode_file
from src.pipeline.cleaning.encoder import encode_markdown


def encode_blocks(
    markdown_clean_dir: str | Path,
    blocks_dir: str | Path,
) -> dict[str, Any]:
    """Encode clean Markdown documents into complete source blocks."""

    result = _encode_all(markdown_clean_dir, blocks_dir)
    return {
        "documents": result.get("documents", 0),
        "blocks": result.get("sections", 0),
        "blocks_dir": str(blocks_dir),
    }


encode_all = encode_blocks
