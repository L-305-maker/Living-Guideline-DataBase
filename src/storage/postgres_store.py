"""PostgreSQL schema, ingestion, and BM25-like full-text retrieval."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

from src.retrieval.chunk_normalizer import iter_normalized_chunks
from src.utils.io import DATA_DIR, read_jsonl


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL DEFAULT '',
    publication_date TEXT NOT NULL DEFAULT 'unknown',
    publication_year INTEGER,
    source_institution TEXT NOT NULL,
    clinical_department TEXT NOT NULL DEFAULT '未分类',
    document_kind TEXT NOT NULL DEFAULT 'guideline',
    source_file TEXT NOT NULL,
    markdown_raw_path TEXT NOT NULL,
    markdown_clean_path TEXT NOT NULL,
    content_sha256 TEXT NOT NULL DEFAULT '',
    content_md TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS document_cards (
    doc_id TEXT PRIMARY KEY REFERENCES documents(doc_id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    publication_date TEXT NOT NULL DEFAULT 'unknown',
    publication_year INTEGER,
    source_institution TEXT NOT NULL DEFAULT 'Unknown',
    clinical_department TEXT NOT NULL DEFAULT '未分类',
    markdown_clean_path TEXT NOT NULL DEFAULT '',
    cleaning_quality TEXT NOT NULL DEFAULT '',
    cleaning_flags TEXT NOT NULL DEFAULT '',
    source_pdf_text_quality TEXT NOT NULL DEFAULT '',
    source_pdf_needs_ocr BOOLEAN NOT NULL DEFAULT FALSE,
    source_pdf_is_scanned BOOLEAN NOT NULL DEFAULT FALSE,
    pdf_text_quality TEXT NOT NULL DEFAULT '',
    pdf_needs_ocr BOOLEAN NOT NULL DEFAULT FALSE,
    pdf_is_scanned BOOLEAN NOT NULL DEFAULT FALSE,
    ocr_engine TEXT NOT NULL DEFAULT '',
    ocr_applied BOOLEAN NOT NULL DEFAULT FALSE,
    ocr_status TEXT NOT NULL DEFAULT '',
    ocr_error TEXT NOT NULL DEFAULT '',
    card_text TEXT NOT NULL DEFAULT '',
    fields_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    content_tsv TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('simple', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('simple', coalesce(clinical_department, '')), 'B') ||
        setweight(to_tsvector('simple', coalesce(card_text, '')), 'C')
    ) STORED
);

CREATE TABLE IF NOT EXISTS document_views (
    view_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    view_type TEXT NOT NULL,
    priority REAL NOT NULL DEFAULT 1.0,
    title TEXT NOT NULL,
    publication_date TEXT NOT NULL DEFAULT 'unknown',
    publication_year INTEGER,
    source_institution TEXT NOT NULL DEFAULT 'Unknown',
    clinical_department TEXT NOT NULL DEFAULT '未分类',
    markdown_clean_path TEXT NOT NULL DEFAULT '',
    cleaning_quality TEXT NOT NULL DEFAULT '',
    cleaning_flags TEXT NOT NULL DEFAULT '',
    source_pdf_text_quality TEXT NOT NULL DEFAULT '',
    source_pdf_needs_ocr BOOLEAN NOT NULL DEFAULT FALSE,
    source_pdf_is_scanned BOOLEAN NOT NULL DEFAULT FALSE,
    pdf_text_quality TEXT NOT NULL DEFAULT '',
    pdf_needs_ocr BOOLEAN NOT NULL DEFAULT FALSE,
    pdf_is_scanned BOOLEAN NOT NULL DEFAULT FALSE,
    ocr_engine TEXT NOT NULL DEFAULT '',
    ocr_applied BOOLEAN NOT NULL DEFAULT FALSE,
    ocr_status TEXT NOT NULL DEFAULT '',
    ocr_error TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL DEFAULT '',
    content_tsv TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('simple', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('simple', coalesce(view_type, '')), 'B') ||
        setweight(to_tsvector('simple', coalesce(clinical_department, '')), 'B') ||
        setweight(to_tsvector('simple', coalesce(text, '')), 'C')
    ) STORED
);

CREATE TABLE IF NOT EXISTS sections (
    section_id BIGSERIAL PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    section_index INTEGER NOT NULL,
    title TEXT NOT NULL,
    publication_date TEXT NOT NULL DEFAULT 'unknown',
    publication_year INTEGER,
    source_institution TEXT NOT NULL,
    clinical_department TEXT NOT NULL DEFAULT '未分类',
    section_path JSONB NOT NULL DEFAULT '[]'::jsonb,
    heading TEXT,
    heading_level INTEGER,
    char_start INTEGER,
    char_end INTEGER,
    is_reference_section BOOLEAN NOT NULL DEFAULT FALSE,
    content TEXT NOT NULL DEFAULT '',
    UNIQUE (doc_id, section_index)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    publication_date TEXT NOT NULL DEFAULT 'unknown',
    publication_year INTEGER,
    source_institution TEXT NOT NULL,
    clinical_department TEXT NOT NULL DEFAULT '未分类',
    section_path JSONB NOT NULL DEFAULT '[]'::jsonb,
    section_path_text TEXT NOT NULL DEFAULT '',
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    retrieval_text TEXT NOT NULL DEFAULT '',
    chunk_type TEXT NOT NULL DEFAULT 'other',
    text_for_embedding TEXT NOT NULL DEFAULT '',
    recommendation TEXT NOT NULL DEFAULT '',
    evidence JSONB NOT NULL DEFAULT '[]'::jsonb,
    chunk_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    retrieval_key TEXT NOT NULL UNIQUE,
    source_file TEXT NOT NULL DEFAULT '',
    markdown_clean_path TEXT NOT NULL DEFAULT '',
    is_reference_section BOOLEAN NOT NULL DEFAULT FALSE,
    content_tsv TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('simple', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('simple', coalesce(section_path_text, '')), 'B') ||
        setweight(to_tsvector('simple', coalesce(clinical_department, '')), 'B') ||
        setweight(to_tsvector('simple', coalesce(chunk_type, '')), 'B') ||
        setweight(to_tsvector('simple', coalesce(retrieval_text, content, '')), 'C')
    ) STORED
);

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS document_kind TEXT NOT NULL DEFAULT 'guideline';

CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_institution);
CREATE INDEX IF NOT EXISTS idx_documents_department ON documents(clinical_department);
CREATE INDEX IF NOT EXISTS idx_documents_kind_department ON documents(document_kind, clinical_department);
CREATE INDEX IF NOT EXISTS idx_documents_publication_date ON documents(publication_date);
CREATE INDEX IF NOT EXISTS idx_documents_year ON documents(publication_year);
CREATE INDEX IF NOT EXISTS idx_document_cards_source ON document_cards(source_institution);
CREATE INDEX IF NOT EXISTS idx_document_cards_department ON document_cards(clinical_department);
CREATE INDEX IF NOT EXISTS idx_document_cards_date ON document_cards(publication_date);
CREATE INDEX IF NOT EXISTS idx_document_cards_content_tsv ON document_cards USING gin(content_tsv);
CREATE INDEX IF NOT EXISTS idx_document_views_doc ON document_views(doc_id);
CREATE INDEX IF NOT EXISTS idx_document_views_type ON document_views(view_type);
CREATE INDEX IF NOT EXISTS idx_document_views_source ON document_views(source_institution);
CREATE INDEX IF NOT EXISTS idx_document_views_department ON document_views(clinical_department);
CREATE INDEX IF NOT EXISTS idx_document_views_date ON document_views(publication_date);
CREATE INDEX IF NOT EXISTS idx_document_views_content_tsv ON document_views USING gin(content_tsv);
CREATE INDEX IF NOT EXISTS idx_sections_doc ON sections(doc_id);
CREATE INDEX IF NOT EXISTS idx_sections_department ON sections(clinical_department);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_institution);
CREATE INDEX IF NOT EXISTS idx_chunks_department ON chunks(clinical_department);
CREATE INDEX IF NOT EXISTS idx_chunks_publication_date ON chunks(publication_date);
CREATE INDEX IF NOT EXISTS idx_chunks_year ON chunks(publication_year);
CREATE INDEX IF NOT EXISTS idx_chunks_content_tsv ON chunks USING gin(content_tsv);
"""


VECTOR_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_card_embeddings (
    doc_id TEXT NOT NULL REFERENCES document_cards(doc_id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    embedding vector(1024) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (doc_id, model)
);

CREATE TABLE IF NOT EXISTS document_view_embeddings (
    view_id TEXT NOT NULL REFERENCES document_views(view_id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    embedding vector(1024) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (view_id, model)
);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    embedding vector(1024) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (chunk_id, model)
);

"""

VECTOR_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_document_card_embeddings_cosine
    ON document_card_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_document_view_embeddings_cosine
    ON document_view_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_cosine
    ON chunk_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
"""


def get_dsn(cli_dsn: str | None = None) -> str:
    if cli_dsn:
        return cli_dsn
    for name in ["POSTGRES_DSN", "DATABASE_URL", "PostgreSQL"]:
        value = os.environ.get(name)
        if value:
            return value
    host = os.environ.get("PGHOST")
    db = os.environ.get("PGDATABASE")
    user = os.environ.get("PGUSER")
    password = os.environ.get("PGPASSWORD")
    port = os.environ.get("PGPORT", "5432")
    if host and db and user:
        auth = f"{user}:{password}" if password else user
        return f"postgresql://{auth}@{host}:{port}/{db}"
    raise RuntimeError("PostgreSQL DSN not configured. Set POSTGRES_DSN or DATABASE_URL.")


def connect(dsn: str | None = None):
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("psycopg is required: python -m pip install 'psycopg[binary]>=3.2'") from exc
    return psycopg.connect(get_dsn(dsn))


def helper_year(publication_date: str | None) -> int | None:
    match = re.search(r"(19\d{2}|20\d{2})", publication_date or "")
    return int(match.group(1)) if match else None


def serialize_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def helper_truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes", "y"}


def init_schema(dsn: str | None = None, with_vector: bool = False) -> dict[str, Any]:
    with connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            cur.execute(SCHEMA_SQL)
            if with_vector:
                cur.execute(VECTOR_SCHEMA_SQL)
        conn.commit()
    return {"ok": True, "schema": "created", "vector_schema": with_vector}


def create_vector_indexes(dsn: str | None = None) -> dict[str, Any]:
    """Build pgvector indexes after all embeddings have been written."""
    with connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(VECTOR_INDEX_SQL)
            for table in ("document_card_embeddings", "document_view_embeddings", "chunk_embeddings"):
                cur.execute(f"ANALYZE {table}")
        conn.commit()
    return {"ok": True, "vector_indexes": "created"}


def reset_schema(dsn: str | None = None, with_vector: bool = False) -> dict[str, Any]:
    with connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS chunk_embeddings")
            cur.execute("DROP TABLE IF EXISTS document_view_embeddings")
            cur.execute("DROP TABLE IF EXISTS document_card_embeddings")
            cur.execute("DROP TABLE IF EXISTS document_embeddings")
            cur.execute("DROP TABLE IF EXISTS chunks")
            cur.execute("DROP TABLE IF EXISTS sections")
            cur.execute("DROP TABLE IF EXISTS document_views")
            cur.execute("DROP TABLE IF EXISTS document_cards")
            cur.execute("DROP TABLE IF EXISTS documents")
        conn.commit()
    return init_schema(dsn, with_vector)


def helper_department_text(record: dict[str, Any]) -> str:
    labels = record.get("clinical_departments") or [record.get("clinical_department") or "未分类"]
    return "|".join(dict.fromkeys(str(label) for label in labels if label)) or "未分类"


def helper_data_artifact_path(data_dir: Path, stored_path: str | None, folder: str, doc_id: str) -> Path:
    """Resolve uploaded artifacts when JSON still contains a path from another machine."""
    path = Path(stored_path or "")
    if stored_path and path.is_file():
        return path
    return data_dir / folder / f"{doc_id}.md"


def iter_document_rows(data_dir: Path) -> Iterator[tuple[Any, ...]]:
    for rec in read_jsonl(data_dir / "documents.jsonl"):
        doc_id = rec["doc_id"]
        clean_path = helper_data_artifact_path(data_dir, rec.get("markdown_clean_path"), "markdown_clean", doc_id)
        raw_path = helper_data_artifact_path(data_dir, rec.get("markdown_raw_path"), "markdown_raw", doc_id)
        if not clean_path.is_file():
            raise FileNotFoundError(f"Missing clean markdown for {doc_id}: {clean_path}")
        yield (
            doc_id,
            rec["title"],
            rec.get("abstract") or "",
            rec.get("publication_date") or "unknown",
            helper_year(rec.get("publication_date")),
            rec["source_institution"],
            helper_department_text(rec),
            rec.get("document_kind") or "guideline",
            rec["source_file"],
            str(raw_path),
            str(clean_path),
            rec.get("content_sha256") or "",
            clean_path.read_text(encoding="utf-8", errors="replace"),
        )


def helper_document_rows(data_dir: Path) -> list[tuple[Any, ...]]:
    return list(iter_document_rows(data_dir))

def iter_document_card_rows(data_dir: Path, allowed_doc_ids: set[str] | None = None) -> Iterator[tuple[Any, ...]]:
    path = data_dir / "document_cards.jsonl"
    if not path.exists():
        return
    for rec in read_jsonl(path):
        doc_id = rec.get("doc_id")
        if allowed_doc_ids is not None and doc_id not in allowed_doc_ids:
            continue
        yield (
            doc_id,
            rec.get("title") or "",
            rec.get("publication_date") or "unknown",
            helper_year(rec.get("publication_date")),
            rec.get("source_institution") or "Unknown",
            helper_department_text(rec),
            str(helper_data_artifact_path(data_dir, rec.get("markdown_clean_path"), "markdown_clean", doc_id)),
            rec.get("cleaning_quality") or "",
            rec.get("cleaning_flags") or "",
            rec.get("source_pdf_text_quality") or "",
            helper_truthy(rec.get("source_pdf_needs_ocr")),
            helper_truthy(rec.get("source_pdf_is_scanned")),
            rec.get("pdf_text_quality") or "",
            helper_truthy(rec.get("pdf_needs_ocr")),
            helper_truthy(rec.get("pdf_is_scanned")),
            rec.get("ocr_engine") or "",
            helper_truthy(rec.get("ocr_applied")),
            rec.get("ocr_status") or "",
            rec.get("ocr_error") or "",
            rec.get("card_text") or "",
            serialize_json(rec.get("fields") or {}),
        )


def iter_document_view_rows(data_dir: Path, allowed_doc_ids: set[str] | None = None) -> Iterator[tuple[Any, ...]]:
    path = data_dir / "document_views.jsonl"
    if not path.exists():
        return
    for rec in read_jsonl(path):
        doc_id = rec.get("doc_id")
        if allowed_doc_ids is not None and doc_id not in allowed_doc_ids:
            continue
        yield (
            rec.get("view_id") or f"{doc_id}__{rec.get('view_type')}",
            doc_id,
            rec.get("view_type") or "",
            float(rec.get("priority") or 1.0),
            rec.get("title") or "",
            rec.get("publication_date") or "unknown",
            helper_year(rec.get("publication_date")),
            rec.get("source_institution") or "Unknown",
            helper_department_text(rec),
            str(helper_data_artifact_path(data_dir, rec.get("markdown_clean_path"), "markdown_clean", doc_id)),
            rec.get("cleaning_quality") or "",
            rec.get("cleaning_flags") or "",
            rec.get("source_pdf_text_quality") or "",
            helper_truthy(rec.get("source_pdf_needs_ocr")),
            helper_truthy(rec.get("source_pdf_is_scanned")),
            rec.get("pdf_text_quality") or "",
            helper_truthy(rec.get("pdf_needs_ocr")),
            helper_truthy(rec.get("pdf_is_scanned")),
            rec.get("ocr_engine") or "",
            helper_truthy(rec.get("ocr_applied")),
            rec.get("ocr_status") or "",
            rec.get("ocr_error") or "",
            rec.get("text") or "",
        )


def iter_section_rows(data_dir: Path) -> Iterator[tuple[Any, ...]]:
    for path in sorted((data_dir / "sections").glob("*.jsonl")):
        for index, rec in enumerate(read_jsonl(path)):
            yield (
                rec["doc_id"],
                index,
                rec["title"],
                rec.get("publication_date") or "unknown",
                helper_year(rec.get("publication_date")),
                rec["source_institution"],
                helper_department_text(rec),
                serialize_json(rec.get("section_path") or []),
                rec.get("heading"),
                rec.get("heading_level"),
                rec.get("char_start"),
                rec.get("char_end"),
                bool(rec.get("is_reference_section")),
                rec.get("content") or "",
            )


def iter_chunk_rows(data_dir: Path) -> Iterator[tuple[Any, ...]]:
    for rec in iter_normalized_chunks(data_dir):
        yield (
            rec["chunk_id"],
            rec["doc_id"],
            rec["title"],
            rec.get("publication_date") or "unknown",
            helper_year(rec.get("publication_date")),
            rec["source_institution"],
            helper_department_text(rec),
            serialize_json(rec.get("section_path") or []),
            " ".join(str(item) for item in (rec.get("section_path") or [])),
            rec["chunk_index"],
            rec.get("content") or "",
            rec.get("retrieval_text") or rec.get("content") or "",
            rec.get("chunk_type") or "other",
            rec.get("text_for_embedding") or rec.get("retrieval_text") or rec.get("content") or "",
            rec.get("recommendation") or "",
            serialize_json(rec.get("evidence") or []),
            serialize_json(rec.get("metadata") or {}),
            rec["retrieval_key"],
            rec.get("source_file") or "",
            str(helper_data_artifact_path(data_dir, rec.get("markdown_clean_path"), "markdown_clean", rec["doc_id"])),
            bool(rec.get("is_reference_section")),
        )


def helper_document_card_rows(data_dir: Path, allowed_doc_ids: set[str] | None = None) -> list[tuple[Any, ...]]:
    return list(iter_document_card_rows(data_dir, allowed_doc_ids))


def helper_document_view_rows(data_dir: Path, allowed_doc_ids: set[str] | None = None) -> list[tuple[Any, ...]]:
    return list(iter_document_view_rows(data_dir, allowed_doc_ids))


def helper_section_rows(data_dir: Path) -> list[tuple[Any, ...]]:
    return list(iter_section_rows(data_dir))


def helper_chunk_rows(data_dir: Path) -> list[tuple[Any, ...]]:
    return list(iter_chunk_rows(data_dir))


def helper_insert_batches(conn: Any, sql: str, rows: Iterable[tuple[Any, ...]], batch_size: int) -> int:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    count = 0
    batch: list[tuple[Any, ...]] = []
    with conn.cursor() as cur:
        for row in rows:
            batch.append(row)
            if len(batch) < batch_size:
                continue
            cur.executemany(sql, batch)
            conn.commit()
            count += len(batch)
            batch.clear()
        if batch:
            cur.executemany(sql, batch)
            conn.commit()
            count += len(batch)
    return count


def ingest_data(dsn: str | None = None, data_dir: str | Path = DATA_DIR, batch_size: int = 500) -> dict[str, Any]:
    data_path = Path(data_dir)
    allowed_doc_ids = {rec["doc_id"] for rec in read_jsonl(data_path / "documents.jsonl")}
    with connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks")
            cur.execute("DELETE FROM sections")
            cur.execute("DELETE FROM document_views")
            cur.execute("DELETE FROM document_cards")
            cur.execute("DELETE FROM documents")
        conn.commit()

        counts = {
            "documents": helper_insert_batches(
                conn,
                """
                INSERT INTO documents
                (doc_id, title, abstract, publication_date, publication_year, source_institution, clinical_department, document_kind, source_file,
                 markdown_raw_path, markdown_clean_path, content_sha256, content_md)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                iter_document_rows(data_path),
                batch_size,
            ),
            "document_cards": helper_insert_batches(
                conn,
                """
                INSERT INTO document_cards
                (doc_id, title, publication_date, publication_year, source_institution, clinical_department,
                 markdown_clean_path, cleaning_quality, cleaning_flags,
                 source_pdf_text_quality, source_pdf_needs_ocr, source_pdf_is_scanned,
                 pdf_text_quality, pdf_needs_ocr, pdf_is_scanned, ocr_engine, ocr_applied, ocr_status, ocr_error,
                 card_text, fields_json)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                """,
                iter_document_card_rows(data_path, allowed_doc_ids),
                batch_size,
            ),
            "document_views": helper_insert_batches(
                conn,
                """
                INSERT INTO document_views
                (view_id, doc_id, view_type, priority, title, publication_date, publication_year,
                 source_institution, clinical_department, markdown_clean_path, cleaning_quality, cleaning_flags,
                 source_pdf_text_quality, source_pdf_needs_ocr, source_pdf_is_scanned,
                 pdf_text_quality, pdf_needs_ocr, pdf_is_scanned, ocr_engine, ocr_applied, ocr_status, ocr_error,
                 text)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                iter_document_view_rows(data_path, allowed_doc_ids),
                batch_size,
            ),
            "sections": helper_insert_batches(
                conn,
                """
                INSERT INTO sections
                (doc_id, section_index, title, publication_date, publication_year, source_institution, clinical_department, section_path,
                 heading, heading_level, char_start, char_end, is_reference_section, content)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s,%s)
                """,
                iter_section_rows(data_path),
                batch_size,
            ),
            "chunks": helper_insert_batches(
                conn,
                """
                INSERT INTO chunks
                (chunk_id, doc_id, title, publication_date, publication_year, source_institution, clinical_department, section_path,
                 section_path_text, chunk_index, content, retrieval_text, chunk_type, text_for_embedding, recommendation, evidence,
                 chunk_metadata, retrieval_key, source_file, markdown_clean_path, is_reference_section)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s,%s)
                """,
                iter_chunk_rows(data_path),
                batch_size,
            ),
        }
    return counts

def helper_kind_counts(cur: Any, table: str) -> dict[str, int]:
    if table == "documents":
        cur.execute("SELECT document_kind, count(*) FROM documents GROUP BY document_kind ORDER BY document_kind")
    else:
        cur.execute(
            f"""
            SELECT d.document_kind, count(*)
            FROM {table} item
            JOIN documents d ON d.doc_id = item.doc_id
            GROUP BY d.document_kind
            ORDER BY d.document_kind
            """
        )
    return {str(kind): int(count) for kind, count in cur.fetchall()}


def database_stats(dsn: str | None = None) -> dict[str, Any]:
    with connect(dsn) as conn:
        with conn.cursor() as cur:
            result: dict[str, Any] = {}
            for table in ["documents", "document_cards", "document_views", "sections", "chunks"]:
                cur.execute("SELECT to_regclass(%s) IS NOT NULL", (table,))
                if cur.fetchone()[0]:
                    cur.execute(f"SELECT count(*) FROM {table}")
                    result[table] = int(cur.fetchone()[0])
                    result[f"{table}_by_document_kind"] = helper_kind_counts(cur, table)
                else:
                    result[table] = 0
            cur.execute("SELECT source_institution, count(*) FROM documents GROUP BY source_institution ORDER BY count(*) DESC")
            result["source_distribution"] = cur.fetchall()
            cur.execute("SELECT count(*) FROM documents WHERE publication_date='unknown'")
            result["unknown_publication_date_documents"] = int(cur.fetchone()[0])
            cur.execute("SELECT count(*) FROM chunks WHERE publication_date='unknown'")
            result["unknown_publication_date_chunks"] = int(cur.fetchone()[0])
            for table in ["document_card_embeddings", "document_view_embeddings", "chunk_embeddings"]:
                cur.execute("SELECT to_regclass(%s) IS NOT NULL", (table,))
                exists = bool(cur.fetchone()[0])
                result[f"{table}_table"] = exists
                if exists:
                    cur.execute(f"SELECT count(*) FROM {table}")
                    result[table] = int(cur.fetchone()[0])
    return result


def verify_retrieval_snapshot(dsn: str | None = None, model_name: str = "BAAI/bge-m3") -> dict[str, Any]:
    artifact_tables = ("documents", "document_cards", "document_views", "sections", "chunks")
    vector_targets = (
        ("document_cards", "document_card_embeddings"),
        ("document_views", "document_view_embeddings"),
        ("chunks", "chunk_embeddings"),
    )
    with connect(dsn) as conn:
        with conn.cursor() as cur:
            by_kind = {table: helper_kind_counts(cur, table) for table in artifact_tables}
            for kind in ("guideline", "consensus"):
                document_count = by_kind["documents"].get(kind, 0)
                if document_count < 1:
                    raise RuntimeError(f"No {kind} documents were ingested")
                if by_kind["document_cards"].get(kind, 0) != document_count:
                    raise RuntimeError(f"{kind} document_cards do not match documents")
                for table in ("document_views", "sections", "chunks"):
                    if by_kind[table].get(kind, 0) < 1:
                        raise RuntimeError(f"No {kind} rows were ingested into {table}")

            vectors: dict[str, dict[str, int]] = {}
            for source_table, embedding_table in vector_targets:
                cur.execute(f"SELECT count(*) FROM {source_table}")
                source_count = int(cur.fetchone()[0])
                cur.execute(f"SELECT count(*) FROM {embedding_table} WHERE model=%s", (model_name,))
                embedding_count = int(cur.fetchone()[0])
                if source_count != embedding_count:
                    raise RuntimeError(
                        f"{embedding_table} is incomplete for {model_name}: "
                        f"expected {source_count}, found {embedding_count}"
                    )
                vectors[source_table] = {"source": source_count, "embeddings": embedding_count}
    return {"status": "ok", "model": model_name, "by_document_kind": by_kind, "vectors": vectors}

def helper_time_filter_sql(time_range: str | dict[str, str] | None, params: list[Any]) -> str:
    if not time_range:
        return ""
    if isinstance(time_range, dict):
        start = time_range.get("start") or time_range.get("start_date")
        end = time_range.get("end") or time_range.get("end_date")
    else:
        match = re.match(r"^\s*(\d{4})(?:-\d{2}-\d{2})?\s*[-~:]\s*(\d{4})(?:-\d{2}-\d{2})?\s*$", str(time_range))
        start = f"{match.group(1)}-01-01" if match else str(time_range)
        end = f"{match.group(2)}-12-31" if match else None
    clauses = []
    if start:
        clauses.append("publication_date >= %s")
        params.append(start)
    if end:
        clauses.append("publication_date <= %s")
        params.append(end)
    return " AND " + " AND ".join(clauses) if clauses else ""


def helper_fuse_document_results(result_lists: list[list[dict[str, Any]]], topk: int) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    by_id: dict[str, dict[str, Any]] = {}
    channels: dict[str, set[str]] = {}
    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            doc_id = item["doc_id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (60 + rank)
            by_id.setdefault(doc_id, item)
            channel = item.get("text_channel")
            if channel:
                channels.setdefault(doc_id, set()).add(channel)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:topk]
    output = []
    for doc_id, score in ranked:
        item = dict(by_id[doc_id])
        item["score"] = score
        item["retrieval_channels"] = sorted(channels.get(doc_id, set()))
        output.append(item)
    return output


def search_document_cards_pg(
    query: str,
    dsn: str | None = None,
    source_institution: str | None = None,
    clinical_department: str | None = None,
    time_range: str | dict[str, str] | None = None,
    publication_date: str | None = None,
    topk: int = 50,
    document_kind: str | None = None,
) -> list[dict[str, Any]]:
    params: list[Any] = [query, query]
    where = ["websearch_to_tsquery('simple', %s) @@ dc.content_tsv"]
    if source_institution:
        where.append("dc.source_institution ILIKE %s")
        params.append(f"%{source_institution}%")
    if clinical_department:
        where.append("dc.clinical_department ILIKE %s")
        params.append(f"%{clinical_department}%")
    if document_kind:
        where.append("d.document_kind = %s")
        params.append(document_kind)
    if publication_date:
        where.append("dc.publication_date = %s")
        params.append(publication_date)
    time_sql = helper_time_filter_sql(time_range, params).replace("publication_date", "dc.publication_date")
    sql = f"""
        SELECT dc.doc_id, dc.title, d.abstract, dc.publication_date, dc.source_institution, dc.clinical_department,
               ts_rank_cd(dc.content_tsv, websearch_to_tsquery('simple', %s)) AS score
        FROM document_cards dc
        JOIN documents d ON d.doc_id = dc.doc_id
        WHERE {' AND '.join(where)} {time_sql}
        ORDER BY score DESC, dc.publication_date DESC
        LIMIT %s
    """
    params.append(topk)
    with connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    return [
        {
            "doc_id": row[0],
            "title": row[1],
            "abstract": row[2],
            "publication_date": row[3],
            "source_institution": row[4],
            "clinical_department": row[5],
            "score": float(row[6] or 0.0),
            "text_channel": "document_card_text",
        }
        for row in rows
    ]


def search_document_views_pg(
    query: str,
    dsn: str | None = None,
    source_institution: str | None = None,
    clinical_department: str | None = None,
    time_range: str | dict[str, str] | None = None,
    publication_date: str | None = None,
    topk: int = 50,
    document_kind: str | None = None,
) -> list[dict[str, Any]]:
    params: list[Any] = [query, query]
    where = ["websearch_to_tsquery('simple', %s) @@ v.content_tsv"]
    if source_institution:
        where.append("v.source_institution ILIKE %s")
        params.append(f"%{source_institution}%")
    if clinical_department:
        where.append("v.clinical_department ILIKE %s")
        params.append(f"%{clinical_department}%")
    if document_kind:
        where.append("d.document_kind = %s")
        params.append(document_kind)
    if publication_date:
        where.append("v.publication_date = %s")
        params.append(publication_date)
    time_sql = helper_time_filter_sql(time_range, params).replace("publication_date", "v.publication_date")
    sql = f"""
        SELECT v.doc_id, v.title, d.abstract, v.publication_date, v.source_institution, v.clinical_department,
               v.view_id, v.view_type, ts_rank_cd(v.content_tsv, websearch_to_tsquery('simple', %s)) AS score
        FROM document_views v
        JOIN documents d ON d.doc_id = v.doc_id
        WHERE {' AND '.join(where)} {time_sql}
        ORDER BY score DESC, v.priority DESC, v.publication_date DESC
        LIMIT %s
    """
    # 一个文档可能有多个视图，先扩大候选池，再在 Python 端按 doc_id 去重至 topk。
    params.append(max(topk * 3, topk))
    with connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    results = []
    seen: set[str] = set()
    for row in rows:
        if row[0] in seen:
            continue
        seen.add(row[0])
        results.append(
            {
                "doc_id": row[0],
                "title": row[1],
                "abstract": row[2],
                "publication_date": row[3],
                "source_institution": row[4],
                "clinical_department": row[5],
                "view_id": row[6],
                "view_type": row[7],
                "score": float(row[8] or 0.0),
                "text_channel": "document_view_text",
            }
        )
        if len(results) >= topk:
            break
    return results


def search_documents_pg(
    query: str,
    dsn: str | None = None,
    source_institution: str | None = None,
    clinical_department: str | None = None,
    time_range: str | dict[str, str] | None = None,
    publication_date: str | None = None,
    topk: int = 10,
) -> list[dict[str, Any]]:
    pool_size = max(50, topk)
    cards = search_document_cards_pg(query, dsn, source_institution, clinical_department, time_range, publication_date, pool_size)
    views = search_document_views_pg(query, dsn, source_institution, clinical_department, time_range, publication_date, pool_size)
    return helper_fuse_document_results([cards, views], topk)


def retrieve_chunks_pg(
    query: str,
    dsn: str | None = None,
    source_institution: str | None = None,
    clinical_department: str | None = None,
    time_range: str | dict[str, str] | None = None,
    publication_date: str | None = None,
    topk: int = 5,
    document_kind: str | None = None,
) -> list[dict[str, Any]]:
    params: list[Any] = [query, query]
    where = ["websearch_to_tsquery('simple', %s) @@ c.content_tsv", "c.is_reference_section = false"]
    if source_institution:
        where.append("c.source_institution ILIKE %s")
        params.append(f"%{source_institution}%")
    if clinical_department:
        where.append("c.clinical_department ILIKE %s")
        params.append(f"%{clinical_department}%")
    if document_kind:
        where.append("d.document_kind = %s")
        params.append(document_kind)
    if publication_date:
        where.append("c.publication_date = %s")
        params.append(publication_date)
    time_sql = helper_time_filter_sql(time_range, params).replace("publication_date", "c.publication_date")
    sql = f"""
        SELECT c.chunk_id, c.doc_id, c.content, c.title, c.publication_date, c.source_institution, c.clinical_department, c.section_path,
               c.chunk_index, c.source_file, c.markdown_clean_path, c.chunk_type, c.retrieval_text,
               ts_rank_cd(c.content_tsv, websearch_to_tsquery('simple', %s)) AS score,
               c.retrieval_key
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id
        WHERE {' AND '.join(where)} {time_sql}
        ORDER BY score DESC
        LIMIT %s
    """
    params.append(topk)
    with connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    return [
        {
            "chunk_id": row[0],
            "doc_id": row[1],
            "content": row[2],
            "title": row[3],
            "publication_date": row[4],
            "source_institution": row[5],
            "clinical_department": row[6],
            "section_path": row[7],
            "chunk_index": row[8],
            "source_file": row[9],
            "markdown_clean_path": row[10],
            "chunk_type": row[11],
            "retrieval_text": row[12],
            "score": float(row[13] or 0.0),
            "retrieval_key": row[14],
        }
        for row in rows
    ]


def read_document_pg(
    doc_id: str | None = None,
    dsn: str | None = None,
    title: str | None = None,
    max_chars: int | None = None,
) -> dict[str, Any]:
    if not doc_id and not title:
        raise ValueError("read_document_pg requires either doc_id or title")
    with connect(dsn) as conn:
        with conn.cursor() as cur:
            if doc_id:
                cur.execute(
                    """
                    SELECT doc_id, title, publication_date, source_institution, clinical_department, document_kind,
                           source_file, markdown_clean_path, content_md
                    FROM documents
                    WHERE doc_id=%s
                    """,
                    (doc_id,),
                )
            else:
                query_title = (title or "").strip()
                cur.execute(
                    """
                    SELECT doc_id, title, publication_date, source_institution, clinical_department, document_kind,
                           source_file, markdown_clean_path, content_md
                    FROM documents
                    WHERE title = %s OR title ILIKE %s
                    ORDER BY CASE WHEN title = %s THEN 0 ELSE 1 END, publication_date DESC, length(title) ASC
                    LIMIT 1
                    """,
                    (query_title, f"%{query_title}%", query_title),
                )
            row = cur.fetchone()
    if not row:
        raise KeyError(f"Document not found: {doc_id or title}")
    content = row[8]
    if max_chars and max_chars > 0:
        content = content[:max_chars]
    return {
        "doc_id": row[0],
        "title": row[1],
        "publication_date": row[2],
        "source_institution": row[3],
        "clinical_department": row[4],
        "document_kind": row[5],
        "source_file": row[6],
        "markdown_clean_path": row[7],
        "content": content,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["init", "reset", "ingest", "index-vectors", "stats", "verify"])
    parser.add_argument("--dsn")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--with-vector", action="store_true")
    parser.add_argument("--model", default=os.getenv("PG_VECTOR_MODEL", "BAAI/bge-m3"))
    args = parser.parse_args()
    if args.command == "init":
        payload = init_schema(args.dsn, args.with_vector)
    elif args.command == "reset":
        payload = reset_schema(args.dsn, args.with_vector)
    elif args.command == "ingest":
        payload = ingest_data(args.dsn, args.data_dir, args.batch_size)
    elif args.command == "index-vectors":
        payload = create_vector_indexes(args.dsn)
    elif args.command == "verify":
        payload = verify_retrieval_snapshot(args.dsn, args.model)
    else:
        payload = database_stats(args.dsn)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
