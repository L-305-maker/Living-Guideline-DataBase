# 把完整的源块（heading-aware 解析产生的 Markdown section）切成更小的检索单元。
#
# 入口 chunk_blocks 是 evidence_pipeline 编排阶段的"切块"步骤之一，
# 上游为 encode_blocks（产生 sections/*.jsonl），下游为 PostgreSQL 入库。
"""Chunk complete source blocks into smaller retrieval units."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.pipeline.cleaning.chunker import chunk_all as _chunk_all


def chunk_blocks(
    markdown_clean_dir: str | Path,
    chunks_dir: str | Path,
) -> dict[str, Any]:
    """按文档逐个切分小 chunk 并写入 chunks_dir。

    业务契约：
    - 每个 chunk 携带 doc_id / chunk_id / section_path，可被 src.storage 追溯。
    - 返回 {"documents": N, "chunks": M, "chunks_dir": path}，
      manifest 维度足够外部脚本统计进度与失败重跑。
    """
    result = _chunk_all(markdown_clean_dir, chunks_dir)
    return {
        "documents": result.get("documents", 0),
        "chunks": result.get("chunks", 0),
        "chunks_dir": str(chunks_dir),
    }