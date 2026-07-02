"""PDF to Markdown entry points for the evidence-building pipeline.

This module keeps the src pipeline aligned with the implementation that is
already exercised under project.pipeline.pdf_to_md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from project.pipeline.pdf_to_md import convert_all as _convert_all
from project.pipeline.pdf_to_md import convert_pdf


def convert_pdfs(
    raw_pdf_dir: str | Path,
    markdown_raw_dir: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Convert raw PDF files to front-matter Markdown and write a manifest."""

    return _convert_all(raw_pdf_dir, markdown_raw_dir, manifest_path)


convert_all = convert_pdfs
