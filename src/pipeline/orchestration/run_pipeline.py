"""Run the evidence-library build pipeline.

Current pipeline:

raw PDF -> Markdown -> clean Markdown -> complete blocks -> chunks -> PostgreSQL-ready JSONL artifacts.

It deliberately does not extract Recommendation, PICOQuestion, GRADE, or
EvidenceItem entities. PostgreSQL ingestion and vectorization are run by
``src.storage`` commands after the artifacts are built.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.pipeline.cleaning.evidence_pipeline import run_evidence_pipeline
from src.utils.io import DATA_DIR


def run_pipeline(
    *,
    data_dir: str | Path = DATA_DIR,
    raw_pdf_dir: str | Path | None = None,
    consensus_pdf_dir: str | Path | None = None,
    skip_pdf_to_markdown: bool = False,
    ocr_mode: str = "auto",
    ocr_languages: str = "chi_sim+eng",
) -> dict[str, Any]:
    """Run the retrieval-first evidence pipeline."""

    return run_evidence_pipeline(
        data_dir=data_dir,
        raw_pdf_dir=raw_pdf_dir,
        consensus_pdf_dir=consensus_pdf_dir,
        skip_pdf_to_markdown=skip_pdf_to_markdown,
        ocr_mode=ocr_mode,
        ocr_languages=ocr_languages,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PostgreSQL-ready guideline evidence artifacts from raw PDFs.")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Evidence artifact root.")
    parser.add_argument("--raw-pdf-dir", default=None, help="Raw PDF directory. Defaults to DATA_DIR/raw_pdf.")
    parser.add_argument("--consensus-pdf-dir", default=None, help="Expert consensus PDF directory. Defaults to DATA_DIR/../consensus.")
    parser.add_argument("--skip-pdf-to-markdown", action="store_true", help="Reuse existing markdown_raw files.")
    parser.add_argument("--ocr-mode", choices=["never", "auto", "force"], default="auto", help="OCRmyPDF mode for scanned PDFs.")
    parser.add_argument("--ocr-languages", default="chi_sim+eng", help="OCR language list passed to OCRmyPDF.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pipeline(
        data_dir=args.data_dir,
        raw_pdf_dir=args.raw_pdf_dir,
        consensus_pdf_dir=args.consensus_pdf_dir,
        skip_pdf_to_markdown=args.skip_pdf_to_markdown,
        ocr_mode=args.ocr_mode,
        ocr_languages=args.ocr_languages,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
