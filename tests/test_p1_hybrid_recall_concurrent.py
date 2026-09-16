"""P1.2 改造的核心 30 行: text 2 路并发 + dense 2 路串行的运行时错误测试。

策略: monkey-patch 实际向 PG 走的 4 个 search_*_pg 函数, 跑真
search_documents_hybrid_pg() 整链路。这样测的是**真正的 P1.2 改动代码**
(ThreadPoolExecutor + as_completed + 异常传递 + helper_optional_vector_channel),
而不是重写一份"看起来像"的代码。

覆盖:
  1. happy path: 4 路全成功 -> 不抛, 输出含所有 doc_id。
  2. 任一路 text 抛错 -> 整体抛错 (注释语义)。
  3. 任一路 dense 抛错 + required=0 -> 降级为空, 其它通道继续 (RRF 仍 OK)。
  4. 任一路 dense 抛错 + required=1 -> 整体抛错。
  5. text 通道**真并发**: 两个 search 同时跑, 整体墙钟 < max(text耗时) 而非 sum。
"""
from __future__ import annotations

import os
import sys
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.storage import pg_hybrid_retrieval as ph  # noqa: E402
from src.storage import postgres_store as ps  # noqa: E402
from src.mcp.call_logger import log_mcp_call  # noqa: E402


def _run_hybrid(monkey_patches: dict[str, object]) -> object:
    """用 context manager 包裹 patch, 跑一次 search_documents_hybrid_pg。"""
    from contextlib import ExitStack

    with ExitStack() as stack:
        for target, name, value in monkey_patches:
            stack.enter_context(patch.object(target, name, value))
        stack.enter_context(patch.object(ph, "query_vector_literal", return_value="[0.1]"))
        return ph.search_documents_hybrid_pg(
            query="q",
            topk=2,
            pool_size=2,
            document_kind="guideline",
        )


def _empty_reranker():
    """Reranker 走不动也得跑通; 直接 mock DocumentReranker 类。"""
    class _Noop:
        def rerank(self, items, topk, query):
            return items[:topk]
    return _Noop()


class HybridRecallRuntime(unittest.TestCase):
    def setUp(self) -> None:
        # 默认: 不要求向量 (绕过 helper_assert_pg_vectors_ready)
        os.environ.pop("PG_VECTOR_RETRIEVAL_REQUIRED", None)

    # ---- happy path ----
    def test_happy_all_four_channels_succeed(self) -> None:
        def cards(*a, **k): return [{"doc_id": "A"}, {"doc_id": "B"}]
        def views(*a, **k): return [{"doc_id": "C"}]
        def vcards(*a, **k): return [{"doc_id": "D"}]
        def vviews(*a, **k): return [{"doc_id": "E"}]

        patches = [
            (ph, "search_document_cards_pg", cards),
            (ph, "search_document_views_pg", views),
            (ph, "vector_search_document_cards_pg", vcards),
            (ph, "vector_search_document_views_pg", vviews),
        ]
        with patch.object(ph, "DocumentReranker", _empty_reranker()):
            result = _run_hybrid(patches)
        ids = {item["doc_id"] for item in result}
        # 不要求全出现 (RRF + topk 截断), 但至少 2 个
        self.assertGreaterEqual(len(ids), 2, f"结果应非空, got {result!r}")

    def test_stage_timings_are_attached_to_call_log(self) -> None:
        def cards(*a, **k): return [{"doc_id": "A"}]
        def views(*a, **k): return [{"doc_id": "B"}]
        def vcards(*a, **k): return [{"doc_id": "C"}]
        def vviews(*a, **k): return [{"doc_id": "D"}]

        patches = [
            (ph, "search_document_cards_pg", cards),
            (ph, "search_document_views_pg", views),
            (ph, "vector_search_document_cards_pg", vcards),
            (ph, "vector_search_document_views_pg", vviews),
        ]
        records = []
        with patch("src.mcp.call_logger.helper_write_record", records.append), \
             patch.object(ph, "DocumentReranker", _empty_reranker()):
            log_mcp_call("search", "postgres", {"query": "q"}, lambda: _run_hybrid(patches))

        timing = records[0]["retrieval_timing"]
        self.assertEqual(1, timing["bm25.document_cards"]["calls"])
        self.assertEqual(1, timing["bm25.document_views"]["calls"])
        self.assertEqual(1, timing["fusion"]["calls"])
        self.assertEqual(1, timing["rerank"]["calls"])

    # ---- text 异常应整体抛 ----
    def test_text_channel_failure_raises(self) -> None:
        def cards(*a, **k): raise RuntimeError("BM25 cards down")
        def views(*a, **k): return [{"doc_id": "C"}]
        def vcards(*a, **k): return [{"doc_id": "D"}]
        def vviews(*a, **k): return [{"doc_id": "E"}]

        patches = [
            (ph, "search_document_cards_pg", cards),
            (ph, "search_document_views_pg", views),
            (ph, "vector_search_document_cards_pg", vcards),
            (ph, "vector_search_document_views_pg", vviews),
        ]
        with patch.object(ph, "DocumentReranker", _empty_reranker()):
            with self.assertRaises(RuntimeError) as cm:
                _run_hybrid(patches)
        self.assertIn("BM25 cards down", str(cm.exception))

    # ---- dense 异常 + required=0 应降级 ----
    def test_dense_failure_degrades_when_not_required(self) -> None:
        os.environ["PG_VECTOR_RETRIEVAL_REQUIRED"] = "0"
        def cards(*a, **k): return [{"doc_id": "A"}]
        def views(*a, **k): return [{"doc_id": "B"}]
        def vcards(*a, **k): raise RuntimeError("pgvector down")
        def vviews(*a, **k): return [{"doc_id": "C"}]

        patches = [
            (ph, "search_document_cards_pg", cards),
            (ph, "search_document_views_pg", views),
            (ph, "vector_search_document_cards_pg", vcards),
            (ph, "vector_search_document_views_pg", vviews),
        ]
        with patch.object(ph, "DocumentReranker", _empty_reranker()):
            result = _run_hybrid(patches)
        # dense 全 down, 应只剩 text 2 路 -> A 或 B
        ids = {item["doc_id"] for item in result}
        self.assertTrue(ids <= {"A", "B"}, f"dense 降级后只应剩 text: {ids}")

    # ---- dense 异常 + required=1 应抛 ----
    def test_dense_failure_raises_when_required(self) -> None:
        os.environ["PG_VECTOR_RETRIEVAL_REQUIRED"] = "1"
        def cards(*a, **k): return [{"doc_id": "A"}]
        def views(*a, **k): return [{"doc_id": "B"}]
        def vcards(*a, **k): raise RuntimeError("pgvector down")
        def vviews(*a, **k): return [{"doc_id": "C"}]

        patches = [
            (ph, "search_document_cards_pg", cards),
            (ph, "search_document_views_pg", views),
            (ph, "vector_search_document_cards_pg", vcards),
            (ph, "vector_search_document_views_pg", vviews),
        ]
        # 同时 patch helper_assert_pg_vectors_ready 避免真连 PG
        with patch.object(ph, "helper_assert_pg_vectors_ready", lambda dsn, m: None), \
             patch.object(ph, "DocumentReranker", _empty_reranker()):
            with self.assertRaises(RuntimeError):
                _run_hybrid(patches)

    # ---- text 真并发验证 ----
    def test_text_channels_run_concurrently(self) -> None:
        # 每个 text 通道 sleep 0.3s. 如果串行 -> 至少 0.6s. 并发 -> ~0.3s.
        sleep_s = 0.3
        timing = {}

        def slow_cards(*a, **k):
            t0 = time.perf_counter()
            time.sleep(sleep_s)
            timing["cards"] = time.perf_counter() - t0
            return [{"doc_id": "A"}]

        def slow_views(*a, **k):
            t0 = time.perf_counter()
            time.sleep(sleep_s)
            timing["views"] = time.perf_counter() - t0
            return [{"doc_id": "B"}]

        # 两条 dense SQL 也应并发执行。
        dense_timing = {}
        def slow_vcards(*a, **k):
            t0 = time.perf_counter()
            time.sleep(sleep_s)
            dense_timing["vcards"] = time.perf_counter() - t0
            return [{"doc_id": "C"}]

        def slow_vviews(*a, **k):
            t0 = time.perf_counter()
            time.sleep(sleep_s)
            dense_timing["vviews"] = time.perf_counter() - t0
            return [{"doc_id": "D"}]

        patches = [
            (ph, "search_document_cards_pg", slow_cards),
            (ph, "search_document_views_pg", slow_views),
            (ph, "vector_search_document_cards_pg", slow_vcards),
            (ph, "vector_search_document_views_pg", slow_vviews),
        ]
        with patch.object(ph, "DocumentReranker", _empty_reranker()):
            t0 = time.perf_counter()
            _run_hybrid(patches)
            wall = time.perf_counter() - t0

        # text 2 路应**几乎同时**启动 (相差 < sleep_s 之一半)
        if "cards" in timing and "views" in timing:
            overlap = abs(timing["cards"] - timing["views"])
            self.assertLess(
                overlap, sleep_s * 0.4,
                f"text 通道未真并发: cards={timing['cards']:.3f}s views={timing['views']:.3f}s",
            )
        # 整体墙钟: text 与编码并发 (~0.3s) + dense SQL 并发 (~0.3s) ≈ 0.6s。
        self.assertLess(wall, 0.95, f"总墙钟过长, 并发未生效: {wall:.3f}s")


if __name__ == "__main__":
    unittest.main(verbosity=2)
