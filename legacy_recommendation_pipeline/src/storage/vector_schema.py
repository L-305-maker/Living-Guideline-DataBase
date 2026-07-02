"""存储层文件：定义 PostgreSQL schema、行映射、入库流程和完整性检查。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

from typing import Any

from src.storage.connection import get_connection


DEFAULT_VECTOR_DIMENSIONS = 1024
EMBEDDING_TABLE = "embedding_records"


def create_vector_schema(*, dimensions: int = DEFAULT_VECTOR_DIMENSIONS, recreate: bool = False) -> None:
    """Create pgvector-backed embedding index tables.

    This schema is intentionally separate from the authoritative Living-Guideline
    storage schema. PostgreSQL remains the source of truth; this table is a
    rebuildable retrieval index.
    """

    if dimensions <= 0:
        raise ValueError("Vector dimensions must be positive.")
    with get_connection() as conn:
        with conn.cursor() as cur:
            create_vector_schema_with_cursor(cur, dimensions=dimensions, recreate=recreate)
        conn.commit()
    print(f"pgvector embedding schema is ready. dimensions={dimensions}")


def create_vector_schema_with_cursor(cur: Any, *, dimensions: int = DEFAULT_VECTOR_DIMENSIONS, recreate: bool = False) -> None:
    if dimensions <= 0:
        raise ValueError("Vector dimensions must be positive.")
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    if recreate:
        cur.execute(f"DROP TABLE IF EXISTS {EMBEDDING_TABLE}")
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {EMBEDDING_TABLE} (
            vector_id TEXT PRIMARY KEY,
            entity_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            collection TEXT NOT NULL,
            text_hash TEXT NOT NULL,
            embedding_model TEXT NOT NULL,
            embedding_version TEXT NOT NULL,
            embedding vector({dimensions}) NOT NULL,
            text_for_embedding TEXT NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            source_file TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_{EMBEDDING_TABLE}_collection
        ON {EMBEDDING_TABLE}(collection)
        """
    )
    cur.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_{EMBEDDING_TABLE}_entity
        ON {EMBEDDING_TABLE}(entity_type, entity_id)
        """
    )
    cur.execute(
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_{EMBEDDING_TABLE}_hash_model
        ON {EMBEDDING_TABLE}(entity_type, entity_id, text_hash, embedding_model, embedding_version)
        """
    )

