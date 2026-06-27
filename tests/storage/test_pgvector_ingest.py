"""存储层测试文件：验证 PostgreSQL schema、contract、原子入库和完整性检查。

阅读测试时，优先看测试名称、输入样例和断言，它们通常说明对应模块的业务边界。
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from src.common.process_jsonl import write_jsonl
from src.storage.connection import get_connection
from src.storage.vector_ingest import (
    FakeEmbeddingProvider,
    fake_embedding,
    ingest_embedding_queue,
    prepare_embedding_record,
    provider_from_args,
    search_embeddings,
)
from src.storage.vector_schema import (
    DEFAULT_VECTOR_DIMENSIONS,
    EMBEDDING_TABLE,
    create_vector_schema,
    create_vector_schema_with_cursor,
)


class PgvectorUnitTests(unittest.TestCase):
    def test_default_dimensions_match_bge_m3_dense_vector_size(self) -> None:
        self.assertEqual(DEFAULT_VECTOR_DIMENSIONS, 1024)

    def test_fake_embedding_is_deterministic_and_dimensioned(self) -> None:
        first = fake_embedding("We recommend treatment.", dimensions=8)
        second = fake_embedding("We recommend treatment.", dimensions=8)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)

    def test_print_json_falls_back_to_utf8_bytes_on_console_encoding_error(self) -> None:
        from src.storage import vector_ingest

        writes: list[bytes] = []

        class Buffer:
            def write(self, value: bytes) -> None:
                writes.append(value)

        class Stdout:
            buffer = Buffer()

            def write(self, _value: str) -> None:
                raise UnicodeEncodeError("gbk", "£", 0, 1, "test")

            def flush(self) -> None:
                return None

        original = vector_ingest.sys.stdout
        vector_ingest.sys.stdout = Stdout()  # type: ignore[assignment]
        try:
            vector_ingest.print_json({"text_for_embedding": "cost £100"})
        finally:
            vector_ingest.sys.stdout = original

        self.assertTrue(writes)
        self.assertIn("text_for_embedding".encode("utf-8"), writes[0])

    def test_prepare_embedding_record_requires_expected_dimension(self) -> None:
        row = {
            "vector_id": "recommendation_version:rv-1:hash",
            "entity_type": "recommendation_version",
            "entity_id": "rv-1",
            "collection": "lg_recommendations_published",
            "text_hash": "hash",
            "embedding_model": "fake",
            "embedding_version": "v1",
            "text_for_embedding": "Recommendation: We recommend treatment.",
            "metadata": {"guideline_id": "g-1"},
            "source_file": "embedding_queue.jsonl",
        }

        prepared = prepare_embedding_record(row, [0.1, 0.2], dimensions=2)

        self.assertEqual(prepared["vector_id"], row["vector_id"])
        self.assertEqual(prepared["embedding"], "[0.10000000,0.20000000]")
        self.assertIn("guideline_id", prepared["metadata"])

    def test_prepare_embedding_record_rejects_missing_required_field(self) -> None:
        with self.assertRaises(ValueError):
            prepare_embedding_record({}, [0.1], dimensions=1)

    def test_search_projection_includes_text_for_embedding_by_default(self) -> None:
        from src.storage import vector_ingest

        executed: dict[str, object] = {}

        class Cursor:
            def execute(self, sql: str, params: tuple[object, ...]) -> None:
                executed["sql"] = sql
                executed["params"] = params

            def fetchall(self) -> list[tuple[object, ...]]:
                return [
                    (
                        "vector-1",
                        "source_block",
                        "block-1",
                        "lg_source_blocks",
                        "hash",
                        "BAAI/bge-m3",
                        "dense-v1",
                        "Title: Demo\nText: We recommend treatment.",
                        {"guideline_id": "g-1"},
                        "blocks.jsonl",
                        0.12,
                    )
                ]

            def __enter__(self) -> "Cursor":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

        class Conn:
            def cursor(self) -> Cursor:
                return Cursor()

            def __enter__(self) -> "Conn":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

        original = vector_ingest.get_connection
        vector_ingest.get_connection = lambda: Conn()  # type: ignore[assignment]
        try:
            results = vector_ingest.search_embeddings(
                "treatment",
                collection="lg_source_blocks",
                provider=FakeEmbeddingProvider(dimensions=4),
                dimensions=4,
                top_k=1,
            )
        finally:
            vector_ingest.get_connection = original

        self.assertIn("text_for_embedding", results[0])
        self.assertIn("text_for_embedding", str(executed["sql"]))

    def test_fake_embeddings_flag_overrides_default_bge_provider(self) -> None:
        class Args:
            fake_embeddings = True
            provider = "bge-m3"
            dimensions = 4

        provider = provider_from_args(Args())

        self.assertIsInstance(provider, FakeEmbeddingProvider)
        self.assertEqual(len(provider.embed_texts(["demo"])[0]), 4)

    def test_recreate_vector_schema_drops_embedding_table_only(self) -> None:
        executed: list[str] = []

        class Cursor:
            def execute(self, sql: str) -> None:
                executed.append(sql)

        create_vector_schema_with_cursor(Cursor(), dimensions=1024, recreate=True)

        self.assertTrue(any(f"DROP TABLE IF EXISTS {EMBEDDING_TABLE}" in sql for sql in executed))
        self.assertTrue(any("CREATE EXTENSION IF NOT EXISTS vector" in sql for sql in executed))


@unittest.skipUnless(os.getenv("RUN_POSTGRES_SMOKE") == "1", "set RUN_POSTGRES_SMOKE=1 to run live PostgreSQL smoke tests")
class PgvectorSmokeTests(unittest.TestCase):
    def test_fake_embedding_ingest_and_search_roundtrip(self) -> None:
        dimensions = DEFAULT_VECTOR_DIMENSIONS
        create_vector_schema(dimensions=dimensions)
        prefix = "pgvector_smoke"
        vector_ids = [f"{prefix}_a", f"{prefix}_b"]
        with tempfile.TemporaryDirectory() as tmp:
            queue_path = Path(tmp) / "embedding_queue.jsonl"
            write_jsonl(
                queue_path,
                [
                    {
                        "vector_id": vector_ids[0],
                        "entity_type": "recommendation_version",
                        "entity_id": "rv-pgvector-a",
                        "collection": "lg_recommendations_published",
                        "text_hash": "hash-a",
                        "embedding_model": "fake",
                        "embedding_version": "v1",
                        "text_for_embedding": "Recommendation: We recommend antibiotics for adults with infection.",
                        "metadata": {"guideline_id": "g-1"},
                    },
                    {
                        "vector_id": vector_ids[1],
                        "entity_type": "recommendation_version",
                        "entity_id": "rv-pgvector-b",
                        "collection": "lg_recommendations_published",
                        "text_hash": "hash-b",
                        "embedding_model": "fake",
                        "embedding_version": "v1",
                        "text_for_embedding": "Recommendation: We suggest sleep therapy for central sleep apnea.",
                        "metadata": {"guideline_id": "g-2"},
                    },
                ],
            )
            try:
                summary = ingest_embedding_queue(
                    queue_path,
                    provider=FakeEmbeddingProvider(dimensions=dimensions),
                    dimensions=dimensions,
                    create_schema_first=False,
                )
                results = search_embeddings(
                    "antibiotics for adults with infection",
                    collection="lg_recommendations_published",
                    provider=FakeEmbeddingProvider(dimensions=dimensions),
                    dimensions=dimensions,
                    top_k=1,
                )

                self.assertEqual(summary["upserted_rows"], 2)
                self.assertEqual(results[0]["entity_id"], "rv-pgvector-a")
            finally:
                with get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(f"DELETE FROM {EMBEDDING_TABLE} WHERE vector_id = ANY(%s)", (vector_ids,))
                    conn.commit()


if __name__ == "__main__":
    unittest.main()

