"""Markdown cleanup entry points for evidence documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.pipeline.cleaning.cleaner import clean_all as _clean_all
from src.pipeline.cleaning.cleaner import clean_file, clean_markdown_text


def clean_markdown_dir(
    markdown_raw_dir: str | Path,
    markdown_clean_dir: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Clean raw Markdown files and write document metadata."""

    return _clean_all(markdown_raw_dir, markdown_clean_dir, manifest_path)


clean_all = clean_markdown_dir
