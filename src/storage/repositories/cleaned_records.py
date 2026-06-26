from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

from src.storage.connection import get_connection
from src.storage.row_mappers import cleaned_record_row, guideline_row, paper_row
from src.storage.schema import create_schema
from src.storage.utils import DEFAULT_BATCH_SIZE, batches, execute_many, iter_jsonl


JsonDict = Dict[str, Any]
connection = Any


def insert_guidelines(conn: connection, rows: Sequence[JsonDict]) -> int:
    sql = """
        INSERT INTO guidelines (
            guideline_id, title, disease_area, guideline_type, status, source, issuer,
            publication_url, pdf_url, current_version, update_frequency, published_date,
            last_updated_at, created_at, updated_at
        )
        VALUES (
            %(guideline_id)s, %(title)s, %(disease_area)s, %(guideline_type)s, %(status)s,
            %(source)s, %(issuer)s, %(publication_url)s, %(pdf_url)s, %(current_version)s,
            %(update_frequency)s, %(published_date)s, %(last_updated_at)s,
            COALESCE(NULLIF(%(created_at)s, ''), CURRENT_TIMESTAMP::text),
            COALESCE(NULLIF(%(updated_at)s, ''), CURRENT_TIMESTAMP::text)
        )
        ON CONFLICT (guideline_id) DO UPDATE SET
            title = EXCLUDED.title,
            disease_area = COALESCE(EXCLUDED.disease_area, guidelines.disease_area),
            guideline_type = EXCLUDED.guideline_type,
            status = EXCLUDED.status,
            source = EXCLUDED.source,
            issuer = EXCLUDED.issuer,
            publication_url = EXCLUDED.publication_url,
            pdf_url = EXCLUDED.pdf_url,
            current_version = COALESCE(EXCLUDED.current_version, guidelines.current_version),
            update_frequency = COALESCE(EXCLUDED.update_frequency, guidelines.update_frequency),
            published_date = EXCLUDED.published_date,
            last_updated_at = COALESCE(EXCLUDED.last_updated_at, guidelines.last_updated_at),
            updated_at = CURRENT_TIMESTAMP::text
    """
    return execute_many(conn, sql, rows)


def insert_papers(conn: connection, rows: Sequence[JsonDict]) -> int:
    sql = """
        INSERT INTO papers (
            paper_id, title, pmid, doi, abstract, authors, journal, publication_date,
            article_type, source_database, url, first_seen_at, screening_status
        )
        VALUES (
            %(paper_id)s, %(title)s, %(pmid)s, %(doi)s, %(abstract)s, %(authors)s::jsonb,
            %(journal)s, %(publication_date)s, %(article_type)s, %(source_database)s,
            %(url)s, COALESCE(NULLIF(%(first_seen_at)s, ''), CURRENT_TIMESTAMP::text),
            %(screening_status)s
        )
        ON CONFLICT (paper_id) DO UPDATE SET
            title = EXCLUDED.title,
            pmid = COALESCE(EXCLUDED.pmid, papers.pmid),
            doi = COALESCE(EXCLUDED.doi, papers.doi),
            abstract = COALESCE(EXCLUDED.abstract, papers.abstract),
            authors = EXCLUDED.authors,
            journal = COALESCE(EXCLUDED.journal, papers.journal),
            publication_date = EXCLUDED.publication_date,
            article_type = COALESCE(EXCLUDED.article_type, papers.article_type),
            source_database = EXCLUDED.source_database,
            url = EXCLUDED.url,
            screening_status = EXCLUDED.screening_status
    """
    return execute_many(conn, sql, rows)


def insert_cleaned_records(conn: connection, rows: Sequence[JsonDict]) -> int:
    sql = """
        INSERT INTO cleaned_records (
            record_id, guideline_id, paper_id, source, issuer, title, url, published_year,
            raw_pdf_path, url_provenance, content, tables, table_count, table_row_count,
            table_cell_count, guideline_seed, paper_seed, direct_extraction, raw_record
        )
        VALUES (
            %(record_id)s, %(guideline_id)s, %(paper_id)s, %(source)s, %(issuer)s,
            %(title)s, %(url)s, %(published_year)s, %(raw_pdf_path)s, %(url_provenance)s,
            %(content)s, %(tables)s::jsonb, %(table_count)s, %(table_row_count)s,
            %(table_cell_count)s, %(guideline_seed)s::jsonb, %(paper_seed)s::jsonb,
            %(direct_extraction)s::jsonb, %(raw_record)s::jsonb
        )
        ON CONFLICT (record_id) DO UPDATE SET
            guideline_id = EXCLUDED.guideline_id,
            paper_id = EXCLUDED.paper_id,
            source = EXCLUDED.source,
            issuer = EXCLUDED.issuer,
            title = EXCLUDED.title,
            url = EXCLUDED.url,
            published_year = EXCLUDED.published_year,
            raw_pdf_path = EXCLUDED.raw_pdf_path,
            url_provenance = EXCLUDED.url_provenance,
            content = EXCLUDED.content,
            tables = EXCLUDED.tables,
            table_count = EXCLUDED.table_count,
            table_row_count = EXCLUDED.table_row_count,
            table_cell_count = EXCLUDED.table_cell_count,
            guideline_seed = EXCLUDED.guideline_seed,
            paper_seed = EXCLUDED.paper_seed,
            direct_extraction = EXCLUDED.direct_extraction,
            raw_record = EXCLUDED.raw_record
    """
    return execute_many(conn, sql, rows)


def ingest_cleaned_records(jsonl_path: str | Path, batch_size: int = DEFAULT_BATCH_SIZE) -> int:
    create_schema(recreate=False)
    total = 0
    with get_connection() as conn:
        for batch in batches(iter_jsonl(jsonl_path), batch_size):
            guideline_rows = [guideline_row(item["guideline_seed"]) for item in batch if item.get("guideline_seed")]
            paper_rows = [paper_row(item["paper_seed"]) for item in batch if item.get("paper_seed")]
            record_rows = [cleaned_record_row(item) for item in batch if item.get("record_id")]
            try:
                insert_guidelines(conn, guideline_rows)
                insert_papers(conn, paper_rows)
                insert_cleaned_records(conn, record_rows)
                conn.commit()
            except Exception as exc:  # noqa: BLE001
                conn.rollback()
                raise RuntimeError(f"Failed to ingest cleaned records after {total} rows: {exc}") from exc
            total += len(record_rows)
            print(f"Ingested {total} cleaned records")
    return total
