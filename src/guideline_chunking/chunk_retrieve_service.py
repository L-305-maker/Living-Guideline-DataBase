# BM25 chunk 检索服务（带 chunk_type 权重）。
#
# 关键设计：
# - CHUNK_TYPE_BOOST：recommendation / evidence / table_row 加成 ~1.20，reference 大幅降权 0.10；
# - retrieve 时拉 5x top_k 候选再按 boost 重排，避免 topk 边界处误切；
# - 排序键 (-final_score, chunk_id)：分数降序，同分时按 chunk_id 字典序稳定排序。
"""BM25 chunk retrieval with chunk-type boost."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.guideline_chunking.bm25_index import BM25Index
from src.guideline_chunking.models import ChunkRetrieveResult


# chunk_type 权重（与 src.retrieval.document_repr.section_classifier.CHUNK_TYPE_WEIGHTS 思路类似但粒度更细）。
# 经验值：候选级（recommendation/evidence/table_row）走高，过程级（background/method）走低。
CHUNK_TYPE_BOOST = {
    "recommendation_candidate": 1.30,
    "evidence_candidate": 1.20,
    "table_row": 1.20,
    "table_parent": 1.05,
    "table_note": 1.00,
    "rationale_candidate": 1.10,
    "scope_candidate": 1.00,
    "population_candidate": 1.00,
    "algorithm_candidate": 1.00,
    "background": 0.60,
    "method": 0.40,
    "reference": 0.10,
    "unknown": 0.70,
}


class ChunkRetrieveService:
    def __init__(self, index_dir: str | Path) -> None:
        # 启动时一次性加载 BM25 索引到内存，后续 retrieve 调用直接复用。
        self.index = BM25Index.load(index_dir)

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[ChunkRetrieveResult]:
        """BM25 召回 + chunk_type 权重重排。

        召回数 = max(top_k * 5, 50)：保证有足够候选被 boost 后仍能筛到 top_k；
        排序键 (-final_score, chunk_id)：分数降序 + chunk_id 升序（稳定排序）。
        """
        candidates = self.index.search(query, top_k=max(top_k * 5, 50), filters=filters)
        results: list[ChunkRetrieveResult] = []
        for record, raw_score, matched_terms in candidates:
            boost = CHUNK_TYPE_BOOST.get(record.get("chunk_type"), 1.0)
            final_score = raw_score * boost
            results.append(
                ChunkRetrieveResult(
                    chunk_id=record["chunk_id"],
                    doc_id=record["doc_id"],
                    chunk_type=record["chunk_type"],
                    score=final_score,
                    title=record.get("title"),
                    publisher=record.get("publisher"),
                    heading_path=record.get("heading_path") or [],
                    text=record.get("text") or "",
                    source_span={
                        "page_start": record.get("page_start"),
                        "page_end": record.get("page_end"),
                        "source_block_ids": record.get("source_block_ids") or [],
                    },
                    debug={
                        "raw_score": raw_score,
                        "chunk_type_boost": boost,
                        "final_score": final_score,
                        "matched_terms": matched_terms,
                    },
                )
            )
        results.sort(key=lambda item: (-item.score, item.chunk_id))
        return results[:top_k]