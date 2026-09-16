"""P1 高并发改造第 3 块 —— helper_enrich_chunk_context_pg 单次 IN 查询路径。

关注 runtime 错误:
  1. items 为空 -> 返回 [], 不会碰 PG。
  2. items 全是 (doc_id, section_path) 重复 -> unnest 数组被去重, SQL 只发 1 次。
  3. 命中: heading/char_start/char_end 正确填充, 原顺序保持。
  4. 未命中: 回退到 section_path 最后一项作为 heading (与原循环一致)。
  5. SQL 抛异常 -> 异常透传，连接池上下文仍退出。
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.storage import pg_hybrid_retrieval as ph  # noqa: E402


def _make_pool_and_conn(fetchall_rows: list[tuple] | None = None, exc: Exception | None = None):
    """构造 mock pool + conn, 模拟 fetchall 返回或抛错。"""
    cur = MagicMock()
    if exc is not None:
        cur.execute.side_effect = exc
        cur.fetchall.side_effect = exc
    else:
        cur.fetchall.return_value = fetchall_rows or []

    conn = MagicMock()
    conn.cursor.return_value.__enter__ = lambda s: cur
    conn.cursor.return_value.__exit__ = lambda s, *a: False
    conn.rollback = MagicMock()

    pool = MagicMock()
    pool.connection.return_value.__enter__ = MagicMock(return_value=conn)
    pool.connection.return_value.__exit__ = MagicMock(return_value=False)
    return pool, conn, cur


class EnrichChunkContextRuntime(unittest.TestCase):

    def test_empty_items_returns_empty_no_pg_call(self) -> None:
        with patch.object(ph, "get_pool") as get_pool:
            out = ph.helper_enrich_chunk_context_pg(dsn=None, items=[])
        self.assertEqual(out, [])
        get_pool.assert_not_called()

    def test_dedup_then_query_then_fill_in_order(self) -> None:
        # 构造 4 个 input, 故意重复 (docA, [1,2]) 3 次 -> 去重后 unnest 数组 2 项
        items = [
            {"doc_id": "A", "section_path": ["1", "2"], "content": "x1"},
            {"doc_id": "A", "section_path": ["1", "2"], "content": "x2"},
            {"doc_id": "B", "section_path": ["3"],        "content": "x3"},
            {"doc_id": "A", "section_path": ["1", "2"], "content": "x4"},
        ]
        # SQL 返回 2 行: (A,[1,2]) 与 (B,[3])
        rows = [
            ("A", ["1", "2"], "Heading A", 100, 200),
            ("B", ["3"],      "Heading B", 300, 400),
        ]
        pool, conn, cur = _make_pool_and_conn(fetchall_rows=rows)

        with patch.object(ph, "get_pool", return_value=pool):
            out = ph.helper_enrich_chunk_context_pg(dsn=None, items=items)

        # 1) 长度与输入一致, 顺序保持
        self.assertEqual(len(out), 4)
        # 2) heading 按 (doc_id, section_path) 命中, 不论输入重复几次
        self.assertEqual(out[0]["heading"], "Heading A")
        self.assertEqual(out[1]["heading"], "Heading A")
        self.assertEqual(out[2]["heading"], "Heading B")
        self.assertEqual(out[3]["heading"], "Heading A")
        # 3) char 字段正确
        self.assertEqual(out[0]["char_start"], 100)
        self.assertEqual(out[2]["char_end"],   400)
        for field in ("prev_chunk_id", "next_chunk_id", "source_quote_context"):
            self.assertNotIn(field, out[0])
        # 4) char_span_kind = "section" 当 char_start/char_end 任一非 None
        self.assertEqual(out[0]["char_span_kind"], "section")
        # 5) SQL 实际执行 1 次, 且 unnest 数组长度 == 2 (去重生效)
        self.assertEqual(cur.execute.call_count, 1)
        args, _ = cur.execute.call_args
        sql = args[0]
        self.assertIn("unnest(%s::text[], %s::jsonb[])", sql)
        unnest_args = args[1]
        self.assertEqual(len(unnest_args), 2)
        self.assertEqual(len(unnest_args[0]), 2, "doc_id 数组应只有 A 和 B 两份")
        self.assertEqual(len(unnest_args[1]), 2, "section_path 数组应只有 2 份")

    def test_unmatched_falls_back_to_section_path_last_item(self) -> None:
        items = [
            {"doc_id": "Z", "section_path": ["x", "y", "z"], "content": ""},
        ]
        # SQL 返回 0 行
        pool, conn, cur = _make_pool_and_conn(fetchall_rows=[])

        with patch.object(ph, "get_pool", return_value=pool):
            out = ph.helper_enrich_chunk_context_pg(dsn=None, items=items)

        # 退化分支: heading 应回退到 section_path 最后一项 "z"
        self.assertEqual(out[0]["heading"], "z")
        self.assertIsNone(out[0]["char_start"])
        self.assertEqual(out[0]["char_span_kind"], None)

    def test_sql_exception_propagates_and_conn_returned(self) -> None:
        items = [{"doc_id": "A", "section_path": ["1"], "content": ""}]
        boom = RuntimeError("connection reset")
        pool, conn, cur = _make_pool_and_conn(exc=boom)

        connection_context = pool.connection.return_value
        with patch.object(ph, "get_pool", return_value=pool):
            with self.assertRaises(RuntimeError) as cm:
                ph.helper_enrich_chunk_context_pg(dsn=None, items=items)

        self.assertIn("connection reset", str(cm.exception))
        connection_context.__exit__.assert_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
