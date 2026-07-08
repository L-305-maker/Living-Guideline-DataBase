"""BM25 chunk retrieval with chunk-type boost."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.guideline_chunking.bm25_index import BM25Index
from src.guideline_chunking.models import ChunkRetrieveResult


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
        self.index = BM25Index.load(index_dir)

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[ChunkRetrieveResult]:
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
