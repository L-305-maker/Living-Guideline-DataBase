"""Cleaning pipeline for evidence-oriented guideline ingestion."""

from __future__ import annotations

from typing import Any

__all__ = [
    "EvidencePipelinePaths",
    "chunk_blocks",
    "clean_markdown_dir",
    "convert_pdfs",
    "encode_blocks",
    "run_evidence_pipeline",
]


def __getattr__(name: str) -> Any:
    if name == "chunk_blocks":
        from src.pipeline.cleaning.block_chunker import chunk_blocks

        return chunk_blocks
    if name == "encode_blocks":
        from src.pipeline.cleaning.block_encoder import encode_blocks

        return encode_blocks
    if name in {"EvidencePipelinePaths", "run_evidence_pipeline"}:
        from src.pipeline.cleaning.evidence_pipeline import EvidencePipelinePaths, run_evidence_pipeline

        return {"EvidencePipelinePaths": EvidencePipelinePaths, "run_evidence_pipeline": run_evidence_pipeline}[name]
    if name == "clean_markdown_dir":
        from src.pipeline.cleaning.markdown_cleaner import clean_markdown_dir

        return clean_markdown_dir
    if name == "convert_pdfs":
        from src.pipeline.cleaning.pdf_to_markdown import convert_pdfs

        return convert_pdfs
    raise AttributeError(name)
