"""Aggregate chunk retrieval hits back to document-level candidates."""

from __future__ import annotations

from typing import Any

from src.retrieval.document_repr.section_classifier import classify_chunk, chunk_type_weight


def aggregate_chunks_to_documents(
    chunks: list[dict[str, Any]],
    *,
    topk: int = 100,
    rrf_k: int = 60,
    max_chunks_per_doc: int = 5,
) -> list[dict[str, Any]]:
    """Return document candidates ranked by weighted chunk evidence."""
    # 聚合以文档为边界，保留高排名分块作为证据摘要，不能把不同文档的分数直接混合。

    scores: dict[str, float] = {}
    matched: dict[str, list[dict[str, Any]]] = {}
    chunk_counts: dict[str, int] = {}
    for rank, chunk in enumerate(chunks, start=1):
        doc_id = str(chunk.get("doc_id") or "")
        if not doc_id:
            continue
        chunk_type = classify_chunk(chunk)
        weight = chunk_type_weight(chunk_type)
        contribution = weight / (rrf_k + rank)
        scores[doc_id] = scores.get(doc_id, 0.0) + contribution
        chunk_counts[doc_id] = chunk_counts.get(doc_id, 0) + 1
        if len(matched.get(doc_id, [])) < max_chunks_per_doc:
            matched.setdefault(doc_id, []).append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "chunk_rank": rank,
                    "chunk_type": chunk_type,
                    "weight": weight,
                    "section_path": chunk.get("section_path") or [],
                    "content": chunk.get("content", ""),
                    "score": chunk.get("score", 0.0),
                }
            )

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    output = []
    for doc_id, score in ranked[:topk]:
        output.append(
            {
                "doc_id": doc_id,
                "score": score,
                "matched_chunks": matched.get(doc_id, []),
                "matched_chunk_count": chunk_counts.get(doc_id, 0),
            }
        )
    return output
