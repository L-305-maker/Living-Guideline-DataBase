"""Evidence-library cleaning pipeline.

The pipeline is intentionally retrieval-first:

raw PDF -> raw Markdown -> clean Markdown -> complete blocks -> small chunks.

It does not extract recommendations, PICO questions, GRADE candidates, or
evidence items. Those concepts are downstream reasoning concerns for the agent,
not entities produced during cleaning.
"""

from __future__ import annotations

import json
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from project.index.bm25_store import build_bm25_indexes
from project.index.sqlite_store import build_sqlite_store
from project.index.vector_store import build_vector_indexes

from src.pipeline.cleaning.block_chunker import chunk_blocks
from src.pipeline.cleaning.block_encoder import encode_blocks
from src.pipeline.cleaning.markdown_cleaner import clean_markdown_dir
from src.pipeline.cleaning.pdf_to_markdown import convert_pdfs


@dataclass(frozen=True)
class EvidencePipelinePaths:
    """Stable artifact paths for one evidence-library build."""

    data_dir: Path
    raw_pdf_dir: Path
    markdown_raw_dir: Path
    markdown_clean_dir: Path
    blocks_dir: Path
    chunks_dir: Path
    index_dir: Path
    raw_manifest: Path
    document_manifest: Path
    sqlite_db: Path
    run_manifest: Path

    @classmethod
    def from_data_dir(cls, data_dir: str | Path, raw_pdf_dir: str | Path | None = None) -> "EvidencePipelinePaths":
        root = Path(data_dir)
        return cls(
            data_dir=root,
            raw_pdf_dir=Path(raw_pdf_dir) if raw_pdf_dir else root / "raw_pdf",
            markdown_raw_dir=root / "markdown_raw",
            markdown_clean_dir=root / "markdown_clean",
            blocks_dir=root / "sections",
            chunks_dir=root / "chunks",
            index_dir=root / "index",
            raw_manifest=root / "documents_raw.jsonl",
            document_manifest=root / "documents.jsonl",
            sqlite_db=root / "index" / "rag.sqlite",
            run_manifest=root / "evidence_pipeline_manifest.json",
        )

    def as_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in self.__dict__.items()}


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_evidence_pipeline(
    *,
    data_dir: str | Path = "project/data",
    raw_pdf_dir: str | Path | None = None,
    skip_pdf_to_markdown: bool = False,
    skip_vector: bool = True,
    legacy_json_bm25: bool = False,
    embedding_model: str = "BAAI/bge-m3",
) -> dict[str, Any]:
    """Run the evidence-library build and return a machine-readable manifest."""

    paths = EvidencePipelinePaths.from_data_dir(data_dir, raw_pdf_dir)
    for directory in [
        paths.markdown_raw_dir,
        paths.markdown_clean_dir,
        paths.blocks_dir,
        paths.chunks_dir,
        paths.index_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    stages: dict[str, Any] = {}
    if skip_pdf_to_markdown:
        stages["pdf_to_markdown"] = {
            "skipped": True,
            "reason": "skip_pdf_to_markdown was set",
            "markdown_raw_dir": str(paths.markdown_raw_dir),
        }
    else:
        stages["pdf_to_markdown"] = convert_pdfs(paths.raw_pdf_dir, paths.markdown_raw_dir, paths.raw_manifest)

    stages["clean_markdown"] = clean_markdown_dir(paths.markdown_raw_dir, paths.markdown_clean_dir, paths.document_manifest)
    stages["encode_blocks"] = encode_blocks(paths.markdown_clean_dir, paths.blocks_dir)
    stages["chunk_blocks"] = chunk_blocks(paths.markdown_clean_dir, paths.chunks_dir)
    stages["sqlite_fts"] = build_sqlite_store(paths.data_dir, paths.sqlite_db)
    gc.collect()

    if legacy_json_bm25:
        stages["legacy_json_bm25"] = build_bm25_indexes(paths.markdown_clean_dir, paths.chunks_dir / "all_chunks.jsonl", paths.index_dir)
    if skip_vector:
        stages["vector"] = {"skipped": True, "reason": "skip_vector defaults to true for lightweight builds"}
    else:
        stages["vector"] = build_vector_indexes(
            paths.markdown_clean_dir,
            paths.chunks_dir / "all_chunks.jsonl",
            paths.index_dir,
            embedding_model,
        )

    manifest = {
        "pipeline_version": "evidence_cleaning_rag_v1",
        "goal": "clean, split, and index guideline evidence for an evidence-based medicine agent",
        "extracts_recommendations": False,
        "extracts_pico_questions": False,
        "artifacts": paths.as_dict(),
        "stages": stages,
    }
    write_json(paths.run_manifest, manifest)
    return manifest
