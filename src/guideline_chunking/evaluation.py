# chunk 检索的简单评估指标。
#
# 主要指标：
# - Chunk Recall@{5,10,20,50}：前 k 个返回中是否命中 relevant_chunk_ids；
# - MRR@10：首个命中的倒数排名（Reciprocal Rank）；
# - Doc Recall@10：top10 文档级是否命中；
# - Chunk Type Precision@10：top10 中 high_value_types 占比。
#
# HIGH_VALUE_TYPES 集合定义哪些 chunk_type 算"高价值"，用于类型精度评估。
"""Simple retrieval metrics for chunk queries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.guideline_chunking.chunk_retrieve_service import ChunkRetrieveService
from src.guideline_chunking.io_utils import read_jsonl


# 高价值 chunk_type 集合：评估时用于计算 Type Precision@10。
HIGH_VALUE_TYPES = {
    "recommendation_candidate",
    "evidence_candidate",
    "table_row",
    "table_parent",
    "rationale_candidate",
    "scope_candidate",
    "population_candidate",
    "algorithm_candidate",
}


def evaluate_chunk_retrieval(index_dir: str | Path, queries_path: str | Path, top_k: int = 50) -> dict[str, float]:
    """在给定 queries 上评估 ChunkRetrieveService，返回多档 Recall / MRR / Doc Recall / Type Precision。"""
    service = ChunkRetrieveService(index_dir)
    queries = list(read_jsonl(queries_path))
    if not queries:
        return {name: 0.0 for name in helper_metric_names(top_k)}

    recall_hits = {5: 0, 10: 0, 20: 0, 50: 0}
    reciprocal_rank_total = 0.0
    doc_recall_hits = 0
    type_precision_total = 0.0

    for query in queries:
        results = service.retrieve(query["query"], top_k=top_k)
        result_ids = [result.chunk_id for result in results]
        result_doc_ids = {result.doc_id for result in results[:10]}
        relevant_ids = set(query.get("relevant_chunk_ids") or [])
        relevant_doc_ids = set(query.get("relevant_doc_ids") or [])

        for k in recall_hits:
            if relevant_ids and relevant_ids.intersection(result_ids[:k]):
                recall_hits[k] += 1
        reciprocal_rank_total += helper_reciprocal_rank(result_ids[:10], relevant_ids)
        if relevant_doc_ids and relevant_doc_ids.intersection(result_doc_ids):
            doc_recall_hits += 1
        top10 = results[:10]
        # top10 为空时精度为 0；避免 ZeroDivisionError
        type_precision_total += (
            sum(1 for result in top10 if result.chunk_type in HIGH_VALUE_TYPES) / len(top10) if top10 else 0.0
        )

    denom = len(queries)
    return {
        "Chunk Recall@5": recall_hits[5] / denom,
        "Chunk Recall@10": recall_hits[10] / denom,
        "Chunk Recall@20": recall_hits[20] / denom,
        "Chunk Recall@50": recall_hits[50] / denom,
        "MRR@10": reciprocal_rank_total / denom,
        "Doc Recall@10": doc_recall_hits / denom,
        "Chunk Type Precision@10": type_precision_total / denom,
    }


def format_metrics(metrics: dict[str, float]) -> str:
    """把 metrics 字典按 key 排序格式化为多行文本。"""
    return "\n".join(f"{name}: {value:.4f}" for name, value in metrics.items())


def helper_reciprocal_rank(result_ids: list[str], relevant_ids: set[str]) -> float:
    """计算首个命中的倒数排名（MRR）；无 relevant_ids 返回 0.0。"""
    if not relevant_ids:
        return 0.0
    for index, chunk_id in enumerate(result_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / index
    return 0.0


def helper_metric_names(top_k: int) -> list[str]:
    """返回标准指标名列表（含 Chunk Recall@top_k）。"""
    return [
        "Chunk Recall@5",
        "Chunk Recall@10",
        "Chunk Recall@20",
        f"Chunk Recall@{top_k}",
        "MRR@10",
        "Doc Recall@10",
        "Chunk Type Precision@10",
    ]