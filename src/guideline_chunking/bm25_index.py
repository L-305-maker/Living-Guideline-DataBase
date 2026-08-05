# 小型 BM25 索引：用于 atomic / table chunks 的轻量全文检索。
#
# 关键设计：
# - BM25+ 平滑 IDF（log(1 + (N - df + 0.5)/(df + 0.5))）抑制高频词；
# - 长度归一化系数 k1=1.2 / b=0.75：BM25 经典经验值，避免长文本因词频高天然占优；
# - 排序键 (-score, chunk_id)：分数降序 + chunk_id 升序（稳定排序）。
# - filters 先于计分：避免对被过滤的记录做无谓的 IDF 计算，同时不改变全库 IDF 统计口径。
"""Small BM25 index for atomic and table chunks."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from src.guideline_chunking.io_utils import read_jsonl, write_jsonl
from src.utils.text import tokenize_search_text as tokenize


def build_chunk_index(
    atomic_chunks_path: str | Path,
    table_chunks_path: str | Path,
    index_dir: str | Path,
) -> dict[str, Any]:
    """把 atomic / table JSONL 合并到 index_dir/chunks.jsonl + manifest.json。"""
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)
    records = []
    if Path(atomic_chunks_path).exists():
        records.extend(helper_index_record(record) for record in read_jsonl(atomic_chunks_path))
    if Path(table_chunks_path).exists():
        records.extend(helper_index_record(record) for record in read_jsonl(table_chunks_path))
    write_jsonl(index_path / "chunks.jsonl", records)
    (index_path / "manifest.json").write_text(
        json.dumps({"num_chunks": len(records)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"index_dir": str(index_path), "num_chunks": len(records)}


class BM25Index:
    """BM25 索引类：构造时一次性 tokenize 全量记录到内存。"""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records
        self.tokenized = [tokenize(record.get("text_for_embedding") or record.get("text") or "") for record in records]
        self.avgdl = sum(len(tokens) for tokens in self.tokenized) / max(1, len(self.tokenized))
        self.df: Counter[str] = Counter()
        for tokens in self.tokenized:
            # 文档频次每篇只计一次，因此用 set 去掉同一文档内的重复词。
            self.df.update(set(tokens))

    @classmethod
    def load(cls, index_dir: str | Path) -> "BM25Index":
        """从 index_dir/chunks.jsonl 加载并构造索引。"""
        return cls(list(read_jsonl(Path(index_dir) / "chunks.jsonl")))

    def search(self, query: str, top_k: int = 20, filters: dict[str, Any] | None = None) -> list[tuple[dict[str, Any], float, list[str]]]:
        """BM25 检索：返回 [(record, score, matched_terms), ...] 按 score 降序。

        关键设计：
        - IDF 平滑：log(1 + (N - df + 0.5)/(df + 0.5))，抑制极端高频词；
        - 长度归一化：k1=1.2 / b=0.75（BM25 经典经验值），避免长文本因词频高天然占优；
        - filters 先于计分：被过滤记录不参与 IDF 计算，不影响全库 IDF 统计口径。
        """
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        results: list[tuple[dict[str, Any], float, list[str]]] = []
        n_docs = max(1, len(self.records))
        for record, tokens in zip(self.records, self.tokenized):
            # 元数据过滤先于计分，可减少无关候选，同时不改变全库 IDF 统计口径。
            if filters and not helper_passes_filters(record, filters):
                continue
            tf = Counter(tokens)
            dl = len(tokens) or 1
            score = 0.0
            matched_terms: list[str] = []
            for token in query_tokens:
                if token not in tf:
                    continue
                matched_terms.append(token)
                # 平滑 IDF 抑制高频词，后续长度归一化再避免长文本因词频高而天然占优。
                idf = math.log(1 + (n_docs - self.df[token] + 0.5) / (self.df[token] + 0.5))
                freq = tf[token]
                score += idf * (freq * 2.2) / (freq + 1.2 * (1 - 0.75 + 0.75 * dl / max(self.avgdl, 1)))
            if score > 0:
                results.append((record, score, sorted(set(matched_terms))))
        results.sort(key=lambda item: (-item[1], item[0]["chunk_id"]))
        return results[:top_k]


def helper_index_record(record: dict[str, Any]) -> dict[str, Any]:
    """从原始 chunk 记录投影出 BM25 索引所需的最小字段集。

    关键设计：title / publisher 从 metadata 顶层化（避免每次访问都查 dict），
    text_for_embedding 缺失时回退 text，保证 BM25 总有可索引文本。
    """
    metadata = record.get("metadata") or {}
    return {
        "chunk_id": record["chunk_id"],
        "doc_id": record["doc_id"],
        "chunk_type": record["chunk_type"],
        "title": metadata.get("title"),
        "publisher": metadata.get("publisher"),
        "heading_path": record.get("heading_path") or [],
        "text": record.get("text") or "",
        "text_for_embedding": record.get("text_for_embedding") or record.get("text") or "",
        "page_start": record.get("page_start"),
        "page_end": record.get("page_end"),
        "source_block_ids": record.get("source_block_ids") or [],
        "metadata": metadata,
    }


def helper_passes_filters(record: dict[str, Any], filters: dict[str, Any]) -> bool:
    """判定 record 是否通过所有 filters；None 值跳过（视为不限制）。"""
    for key, value in filters.items():
        if value is None:
            continue
        actual = record.get(key, (record.get("metadata") or {}).get(key))
        if isinstance(value, (list, tuple, set)):
            if actual not in value:
                return False
        elif actual != value:
            return False
    return True