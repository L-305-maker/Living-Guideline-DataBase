# 面向检索的 Markdown 清洗、切分、编码子模块集合。
#
# 主要入口（惰性 re-export，避免启动时加载重型依赖）：
# - convert_pdfs：PDF → Markdown（含 OCR 路由）
# - clean_markdown_dir：清洗 Markdown（去噪、修复、metadata）
# - chunk_blocks / encode_blocks：按块切分 + front-matter 编码
# - run_evidence_pipeline：四阶段编排（convert → clean → encode → chunk）
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
    # PEP 562 模块级 __getattr__：让 `from src.pipeline.cleaning import xxx`
    # 在属性实际被访问时才触发 import，减少冷启动开销与循环依赖风险。
    if name == "chunk_blocks":
        from src.pipeline.cleaning.block_chunker import chunk_blocks

        return chunk_blocks
    if name == "encode_blocks":
        from src.pipeline.cleaning.block_encoder import encode_blocks

        return encode_blocks
    if name in {"EvidencePipelinePaths", "run_evidence_pipeline"}:
        from src.pipeline.cleaning.evidence_pipeline import (
            EvidencePipelinePaths,
            run_evidence_pipeline,
        )

        # 同一函数按需返回不同符号，避免一次性 import 整组。
        return {
            "EvidencePipelinePaths": EvidencePipelinePaths,
            "run_evidence_pipeline": run_evidence_pipeline,
        }[name]
    if name == "clean_markdown_dir":
        from src.pipeline.cleaning.markdown_cleaner import clean_markdown_dir

        return clean_markdown_dir
    if name == "convert_pdfs":
        from src.pipeline.cleaning.pdf_to_markdown import convert_pdfs

        return convert_pdfs
    raise AttributeError(name)