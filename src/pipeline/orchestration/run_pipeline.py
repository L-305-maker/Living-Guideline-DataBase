"""Run the evidence-library build pipeline.

This is the current main pipeline:

raw PDF -> Markdown -> clean Markdown -> complete blocks -> chunks -> indexes.

It deliberately does not extract Recommendation, PICOQuestion, GRADE, or
EvidenceItem entities. The output is a retrieval-ready evidence library for an
evidence-based medicine agent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.pipeline.cleaning.evidence_pipeline import run_evidence_pipeline


def run_pipeline(
    *,
    data_dir: str | Path = "project/data",
    raw_pdf_dir: str | Path | None = None,
    skip_pdf_to_markdown: bool = False,
    skip_vector: bool = False,
    legacy_json_bm25: bool = False,
    embedding_model: str = "BAAI/bge-m3",
) -> dict[str, Any]:
    """Run the retrieval-first evidence pipeline."""

    return run_evidence_pipeline(
        data_dir=data_dir,
        raw_pdf_dir=raw_pdf_dir,
        skip_pdf_to_markdown=skip_pdf_to_markdown,
        skip_vector=skip_vector,
        legacy_json_bm25=legacy_json_bm25,
        embedding_model=embedding_model,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a guideline evidence library from raw PDFs.")
    parser.add_argument("--data-dir", default="project/data", help="Evidence artifact root.")
    parser.add_argument("--raw-pdf-dir", default=None, help="Raw PDF directory. Defaults to DATA_DIR/raw_pdf.")
    parser.add_argument("--skip-pdf-to-markdown", action="store_true", help="Reuse existing markdown_raw files.")
    parser.add_argument("--skip-vector", action="store_true", help="Skip optional FAISS/vector index building.")
    parser.add_argument("--legacy-json-bm25", action="store_true", help="Also build legacy JSON BM25 indexes.")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3", help="SentenceTransformer model for optional vectors.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pipeline(
        data_dir=args.data_dir,
        raw_pdf_dir=args.raw_pdf_dir,
        skip_pdf_to_markdown=args.skip_pdf_to_markdown,
        skip_vector=args.skip_vector,
        legacy_json_bm25=args.legacy_json_bm25,
        embedding_model=args.embedding_model,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
