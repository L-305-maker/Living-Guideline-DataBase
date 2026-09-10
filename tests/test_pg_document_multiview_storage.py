from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from src.storage import pg_hybrid_retrieval, postgres_store, vectorize
from src.storage.postgres_store import (
    SCHEMA_SQL,
    VECTOR_INDEX_SQL,
    VECTOR_SCHEMA_SQL,
    helper_document_card_rows,
    helper_document_rows,
    helper_document_view_rows,
)


class PgDocumentMultiviewStorageTest(unittest.TestCase):
    def test_pg_schema_uses_document_card_and_view_embeddings(self) -> None:
        self.assertIn("CREATE TABLE IF NOT EXISTS document_cards", SCHEMA_SQL)
        self.assertIn("CREATE TABLE IF NOT EXISTS document_views", SCHEMA_SQL)
        self.assertIn("CREATE TABLE IF NOT EXISTS document_card_embeddings", VECTOR_SCHEMA_SQL)
        self.assertIn("CREATE TABLE IF NOT EXISTS document_view_embeddings", VECTOR_SCHEMA_SQL)
        self.assertNotIn("CREATE TABLE IF NOT EXISTS document_embeddings", VECTOR_SCHEMA_SQL)
        document_table_block = SCHEMA_SQL.split("CREATE TABLE IF NOT EXISTS document_cards", 1)[0]
        self.assertNotIn("content_tsv", document_table_block)
        self.assertNotIn("idx_documents_content_tsv", SCHEMA_SQL)

    def test_vector_indexes_are_created_only_after_vectorization(self) -> None:
        self.assertNotIn("USING ivfflat", VECTOR_SCHEMA_SQL)
        self.assertIn("idx_document_card_embeddings_cosine", VECTOR_INDEX_SQL)
        self.assertIn("idx_document_view_embeddings_cosine", VECTOR_INDEX_SQL)
        self.assertIn("idx_chunk_embeddings_cosine", VECTOR_INDEX_SQL)

    def test_pg_schema_uses_trigram_indexes_for_contains_filters(self) -> None:
        self.assertIn("idx_documents_source_trgm", SCHEMA_SQL)
        self.assertIn("USING gin(source_institution gin_trgm_ops)", SCHEMA_SQL)
        self.assertIn("idx_chunks_department_trgm", SCHEMA_SQL)
        self.assertIn("USING gin(clinical_department gin_trgm_ops)", SCHEMA_SQL)

    def test_get_pool_reuses_only_matching_dsn(self) -> None:
        pools = [mock.Mock(), mock.Mock()]
        postgres_store.POOLS.clear()
        try:
            with mock.patch.object(postgres_store, "ConnectionPool", side_effect=pools) as constructor:
                first = postgres_store.get_pool("postgresql://first/db")
                repeated = postgres_store.get_pool("postgresql://first/db")
                second = postgres_store.get_pool("postgresql://second/db")
        finally:
            postgres_store.POOLS.clear()

        self.assertIs(first, repeated)
        self.assertIsNot(first, second)
        self.assertEqual(2, constructor.call_count)
        first.open.assert_called_once_with(wait=True, timeout=10.0)
        second.open.assert_called_once_with(wait=True, timeout=10.0)

    def test_required_vector_check_reads_each_current_snapshot(self) -> None:
        cursor = mock.MagicMock()
        cursor.fetchone.side_effect = [(1,), (1,)] * 6
        connection = mock.MagicMock()
        connection.cursor.return_value.__enter__.return_value = cursor
        pool = mock.MagicMock()
        pool.connection.return_value.__enter__.return_value = connection

        with mock.patch.object(pg_hybrid_retrieval, "get_pool", return_value=pool):
            pg_hybrid_retrieval.helper_assert_pg_vectors_ready(None, "model")
            pg_hybrid_retrieval.helper_assert_pg_vectors_ready(None, "model")

        self.assertEqual(2, pool.connection.call_count)
        self.assertEqual(12, cursor.execute.call_count)

    def test_vectorizer_rejects_dimension_that_cannot_fit_schema(self) -> None:
        embeddings = mock.Mock()
        embeddings.shape = (1, 512)
        pool = mock.MagicMock()

        with (
            mock.patch.object(vectorize, "get_pool", return_value=pool),
            mock.patch.object(vectorize, "_encode_with_model", return_value=embeddings),
        ):
            with self.assertRaisesRegex(ValueError, "expected 1024, got 512"):
                vectorize.helper_vectorize_rows(
                    [("chunk-1", "text")],
                    None,
                    "model",
                    1,
                    vectorize.CHUNKS,
                    model=object(),
                )

    def test_text_search_keeps_filter_parameter_order_and_named_results(self) -> None:
        cursor = mock.MagicMock()
        cursor.fetchall.return_value = [
            {
                "doc_id": "doc-1",
                "title": "Title",
                "abstract": "Abstract",
                "publication_date": "2025-01-01",
                "source_institution": "Org",
                "clinical_department": "Cardiology",
                "score": 0.5,
            }
        ]
        connection = mock.MagicMock()
        connection.cursor.return_value.__enter__.return_value = cursor
        pool = mock.MagicMock()
        pool.connection.return_value.__enter__.return_value = connection

        with mock.patch.object(postgres_store, "get_pool", return_value=pool):
            result = postgres_store.search_document_cards_pg(
                "query",
                source_institution="Org",
                clinical_department="Cardiology",
                time_range="2020-2025",
                topk=2,
                document_kind="guideline",
            )

        _, params = cursor.execute.call_args.args
        self.assertEqual(
            ["query", "query", "%Org%", "%Cardiology%", "guideline", "2020-01-01", "2025-12-31", 2],
            params,
        )
        self.assertEqual("document_card_text", result[0]["text_channel"])
        self.assertEqual(0.5, result[0]["score"])

    def test_vector_search_uses_the_same_filter_mapping(self) -> None:
        cursor = mock.MagicMock()
        cursor.fetchall.return_value = [
            {
                "doc_id": "doc-1",
                "title": "Title",
                "abstract": "Abstract",
                "publication_date": "2025-01-01",
                "source_institution": "Org",
                "clinical_department": "Cardiology",
                "score": 0.75,
            }
        ]
        connection = mock.MagicMock()
        connection.cursor.return_value.__enter__.return_value = cursor
        pool = mock.MagicMock()
        pool.connection.return_value.__enter__.return_value = connection

        with (
            mock.patch.object(pg_hybrid_retrieval, "get_pool", return_value=pool),
            mock.patch.object(pg_hybrid_retrieval, "query_vector_literal", return_value="[0.1]"),
        ):
            result = pg_hybrid_retrieval.vector_search_document_cards_pg(
                "query",
                source_institution="Org",
                topk=2,
                model_name="model",
                document_kind="guideline",
            )

        _, params = cursor.execute.call_args.args
        self.assertEqual(["[0.1]", "model", "%Org%", "guideline", "[0.1]", 2], params)
        self.assertEqual("document_card", result[0]["vector_channel"])
        self.assertEqual(0.75, result[0]["score"])

    def test_pg_vector_requirement_uses_explicit_environment_flag(self) -> None:
        previous = os.environ.get("PG_VECTOR_RETRIEVAL_REQUIRED")
        try:
            os.environ["PG_VECTOR_RETRIEVAL_REQUIRED"] = "1"
            self.assertTrue(pg_hybrid_retrieval.pg_vector_retrieval_required())
            os.environ["PG_VECTOR_RETRIEVAL_REQUIRED"] = "0"
            self.assertFalse(pg_hybrid_retrieval.pg_vector_retrieval_required())
        finally:
            if previous is None:
                os.environ.pop("PG_VECTOR_RETRIEVAL_REQUIRED", None)
            else:
                os.environ["PG_VECTOR_RETRIEVAL_REQUIRED"] = previous

    def test_document_card_and_view_rows_are_loaded_for_pg_ingest(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            data_dir = Path(tmp)
            card = {
                "doc_id": "doc1",
                "title": "Demo Guideline",
                "publication_date": "2026-01-01",
                "source_institution": "Demo",
                "clinical_department": "Cardiology",
                "markdown_clean_path": "demo.md",
                "source_pdf_needs_ocr": "true",
                "source_pdf_is_scanned": "false",
                "pdf_needs_ocr": "yes",
                "pdf_is_scanned": "0",
                "ocr_applied": "1",
                "card_text": "[Title] Demo [Key Recommendations] We recommend treatment.",
                "fields": {"key_recommendations": "We recommend treatment."},
            }
            view = {
                "view_id": "doc1__recommendation_summary",
                "doc_id": "doc1",
                "view_type": "recommendation_summary",
                "priority": 2.0,
                "title": "Demo Guideline",
                "publication_date": "2026-01-01",
                "source_institution": "Demo",
                "clinical_department": "Cardiology",
                "markdown_clean_path": "demo.md",
                "text": "We recommend treatment.",
            }
            (data_dir / "document_cards.jsonl").write_text(json.dumps(card) + "\n", encoding="utf-8")
            (data_dir / "document_views.jsonl").write_text(json.dumps(view) + "\n", encoding="utf-8")

            card_rows = helper_document_card_rows(data_dir, {"doc1"})
            view_rows = helper_document_view_rows(data_dir, {"doc1"})

        self.assertEqual(1, len(card_rows))
        self.assertEqual("doc1", card_rows[0][0])
        self.assertTrue(card_rows[0][10])
        self.assertFalse(card_rows[0][11])
        self.assertTrue(card_rows[0][13])
        self.assertFalse(card_rows[0][14])
        self.assertTrue(card_rows[0][16])
        self.assertEqual({"key_recommendations": "We recommend treatment."}, json.loads(card_rows[0][20]))
        self.assertEqual(1, len(view_rows))
        self.assertEqual("doc1__recommendation_summary", view_rows[0][0])
        self.assertEqual("recommendation_summary", view_rows[0][2])
        self.assertEqual("We recommend treatment.", view_rows[0][22])

    def test_document_rows_resolve_uploaded_markdown_by_doc_id(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            data_dir = Path(tmp)
            (data_dir / "markdown_clean").mkdir()
            (data_dir / "markdown_raw").mkdir()
            (data_dir / "markdown_clean" / "doc1.md").write_text("remote content", encoding="utf-8")
            document = {
                "doc_id": "doc1",
                "title": "Demo",
                "source_institution": "Demo",
                "source_file": "demo.pdf",
                "markdown_raw_path": r"D:\\old\\markdown_raw\\doc1.md",
                "markdown_clean_path": r"D:\\old\\markdown_clean\\doc1.md",
            }
            (data_dir / "documents.jsonl").write_text(json.dumps(document) + "\n", encoding="utf-8")
            rows = helper_document_rows(data_dir)

        self.assertEqual("remote content", rows[0][12])
        self.assertTrue(rows[0][10].replace("\\", "/").endswith("markdown_clean/doc1.md"))
    def test_vectorizer_does_not_expose_legacy_document_embedding_target(self) -> None:
        self.assertTrue(hasattr(vectorize, "vectorize_document_cards"))
        self.assertTrue(hasattr(vectorize, "vectorize_document_views"))
        self.assertFalse(hasattr(vectorize, "vectorize_documents"))
        self.assertFalse(hasattr(pg_hybrid_retrieval, "vector_search_documents_pg"))
        self.assertNotIn("search_documents_pg", pg_hybrid_retrieval.search_documents_hybrid_pg.__code__.co_names)

    def test_incremental_ingest_appends_only_new_document_ids(self) -> None:
        class FakeCursor:
            def __init__(self) -> None:
                self.executed = []
                self.inserted = []
                self.rowcount = 0
                self.selected_rows = []

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def execute(self, sql, params=None) -> None:
                self.executed.append((sql, params))
                self.selected_rows = [("existing",)] if "SELECT doc_id FROM documents" in sql else []

            def executemany(self, sql, rows) -> None:
                batch = list(rows)
                self.inserted.append((sql, batch))
                self.rowcount = len(batch)

            def fetchall(self):
                return self.selected_rows

        class FakeConnection:
            def __init__(self) -> None:
                self.cursor_value = FakeCursor()
                self.commits = 0

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def cursor(self):
                return self.cursor_value

            def commit(self) -> None:
                self.commits += 1

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            data_dir = Path(tmp)
            (data_dir / "markdown_clean").mkdir()
            (data_dir / "sections").mkdir()
            (data_dir / "chunks").mkdir()
            (data_dir / "markdown_clean" / "new.md").write_text("new content", encoding="utf-8")
            documents = [
                {
                    "doc_id": "existing",
                    "title": "Existing",
                    "source_institution": "Demo",
                    "source_file": "existing.pdf",
                },
                {
                    "doc_id": "new",
                    "title": "New",
                    "source_institution": "Demo",
                    "source_file": "new.pdf",
                },
            ]
            cards = [{"doc_id": item["doc_id"], "title": item["title"]} for item in documents]
            views = [
                {
                    "view_id": f"{item['doc_id']}__summary",
                    "doc_id": item["doc_id"],
                    "view_type": "summary",
                    "title": item["title"],
                }
                for item in documents
            ]
            sections = [
                {
                    "doc_id": item["doc_id"],
                    "title": item["title"],
                    "source_institution": "Demo",
                    "content": "section",
                }
                for item in documents
            ]
            chunks = [
                {
                    "chunk_id": f"{item['doc_id']}__chunk",
                    "doc_id": item["doc_id"],
                    "title": item["title"],
                    "source_institution": "Demo",
                    "content": "chunk",
                    "retrieval_key": f"{item['doc_id']}#0",
                }
                for item in documents
            ]
            for filename, rows in (
                ("documents.jsonl", documents),
                ("document_cards.jsonl", cards),
                ("document_views.jsonl", views),
                ("sections/items.jsonl", sections),
                ("chunks/all_chunks.jsonl", chunks),
            ):
                (data_dir / filename).write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )

            connection = FakeConnection()
            pool = mock.Mock()
            pool.connection.return_value = connection
            with mock.patch.object(postgres_store, "get_pool", return_value=pool):
                result = postgres_store.ingest_data(data_dir=data_dir, batch_size=10, incremental=True)

        statements = [sql for sql, _ in connection.cursor_value.executed]
        self.assertTrue(any("LOCK TABLE documents" in sql for sql in statements))
        self.assertFalse(any("DELETE FROM" in sql for sql in statements))
        self.assertEqual(1, connection.commits)
        self.assertEqual(1, result["documents"])
        self.assertEqual(1, result["document_cards"])
        self.assertEqual(1, result["document_views"])
        self.assertEqual(1, result["sections"])
        self.assertEqual(1, result["chunks"])
        self.assertEqual(1, result["skipped_existing_documents"])
        for sql, rows in connection.cursor_value.inserted:
            self.assertIn("ON CONFLICT DO NOTHING", sql)
            doc_id_index = 1 if "INSERT INTO document_views" in sql or "INSERT INTO chunks" in sql else 0
            self.assertEqual({"new"}, {row[doc_id_index] for row in rows})
    def test_ingest_sections_only_appends_rows_for_documents_in_database(self) -> None:
        class FakeCursor:
            def __init__(self) -> None:
                self.statements = []
                self.batches = []
                self.rowcount = 0

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def execute(self, sql, params=None) -> None:
                self.statements.append((sql, params))

            def executemany(self, sql, rows) -> None:
                batch = list(rows)
                self.batches.append((sql, batch))
                self.rowcount = len(batch)

            def fetchall(self):
                return [("in-db",)]

        class FakeConnection:
            def __init__(self) -> None:
                self.cursor_value = FakeCursor()
                self.commits = 0

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def cursor(self):
                return self.cursor_value

            def commit(self) -> None:
                self.commits += 1

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            data_dir = Path(tmp)
            (data_dir / "sections").mkdir()
            documents = [{"doc_id": "in-db"}, {"doc_id": "not-in-db"}]
            sections = [
                {
                    "doc_id": doc_id,
                    "title": doc_id,
                    "source_institution": "Demo",
                    "section_path": ["Section"],
                    "content": "content",
                }
                for doc_id in ("in-db", "not-in-db")
            ]
            (data_dir / "documents.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in documents), encoding="utf-8"
            )
            (data_dir / "sections" / "items.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in sections), encoding="utf-8"
            )
            connection = FakeConnection()
            pool = mock.Mock()
            pool.connection.return_value = connection
            with mock.patch.object(postgres_store, "get_pool", return_value=pool):
                result = postgres_store.ingest_sections(data_dir=data_dir, batch_size=10)

        self.assertEqual({"matched_documents": 1, "sections_inserted": 1}, result)
        self.assertEqual(1, connection.commits)
        self.assertFalse(any("DELETE FROM" in sql for sql, _ in connection.cursor_value.statements))
        self.assertEqual(1, len(connection.cursor_value.batches))
        sql, rows = connection.cursor_value.batches[0]
        self.assertIn("ON CONFLICT DO NOTHING", sql)
        self.assertEqual(["in-db"], [row[0] for row in rows])

if __name__ == "__main__":
    unittest.main()
