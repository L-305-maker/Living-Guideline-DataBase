"""PDF to Markdown entry points for the evidence-building pipeline.

This module keeps the src pipeline aligned with the implementation that is
already exercised under src.pipeline.cleaning.pdf_to_md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.pipeline.cleaning.pdf_to_md import convert_all as _convert_all
from src.pipeline.cleaning.pdf_to_md import convert_pdf


def convert_pdfs(
    raw_pdf_dir: str | Path,
    markdown_raw_dir: str | Path,
    manifest_path: str | Path,
    *,
    ocr_mode: str = "auto",
    ocr_output_dir: str | Path | None = None,
    ocr_languages: str = "chi_sim+eng",
    document_kind: str = "guideline",
    append: bool = False,
) -> dict[str, Any]:
    """Convert raw PDF files to front-matter Markdown and write a manifest."""

    return _convert_all(
        raw_pdf_dir,
        markdown_raw_dir,
        manifest_path,
        ocr_mode=ocr_mode,
        ocr_output_dir=ocr_output_dir,
        ocr_languages=ocr_languages,
        document_kind=document_kind,
        append=append,
    )


convert_all = convert_pdfs
