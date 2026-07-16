"""Vectorize document cards, document views, and chunks with BGE-M3 into pgvector."""

from __future__ import annotations

import argparse
import json
from typing import Any

from src.storage.postgres_store import connect
from src.storage.query_embedding import DEFAULT_MODEL, load_model as helper_load_model, vector_literal as _vector_literal


def ensure_vector_schema(dsn: str | None = None) -> None:
    from src.storage.postgres_store import VECTOR_SCHEMA_SQL

    with connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(VECTOR_SCHEMA_SQL)
        conn.commit()


def helper_fetch_document_cards(
    dsn: str | None,
    limit: int | None,
    offset: int,
    missing_only: bool,
    model_name: str,
    max_text_chars: int,
) -> list[tuple[str, str]]:
    sql = """
        SELECT dc.doc_id, left(dc.card_text, %s) AS content
        FROM document_cards dc
    """
    params: list[Any] = [max_text_chars]
    if missing_only:
        # 缺失判断包含模型名：更换模型后会为同一文档生成一套新的向量记录。
        sql += " LEFT JOIN document_card_embeddings e ON e.doc_id=dc.doc_id AND e.model=%s WHERE e.doc_id IS NULL"
        params.append(model_name)
    # OFFSET 必须配合稳定排序，否则分批执行时可能重复或漏掉记录。
    sql += " ORDER BY dc.doc_id OFFSET %s"
    params.append(offset)
    if limit:
        sql += " LIMIT %s"
        params.append(limit)
    with connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def helper_fetch_document_views(
    dsn: str | None,
    limit: int | None,
    offset: int,
    missing_only: bool,
    model_name: str,
    max_text_chars: int,
) -> list[tuple[str, str]]:
    sql = """
        SELECT v.view_id, left(v.text, %s) AS content
        FROM document_views v
    """
    params: list[Any] = [max_text_chars]
    if missing_only:
        sql += " LEFT JOIN document_view_embeddings e ON e.view_id=v.view_id AND e.model=%s WHERE e.view_id IS NULL"
        params.append(model_name)
    sql += " ORDER BY v.view_id OFFSET %s"
    params.append(offset)
    if limit:
        sql += " LIMIT %s"
        params.append(limit)
    with connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def helper_fetch_chunks(dsn: str | None, limit: int | None, offset: int, missing_only: bool, model_name: str) -> list[tuple[str, str]]:
    sql = """
        SELECT c.chunk_id, coalesce(nullif(c.text_for_embedding, ''), nullif(c.retrieval_text, ''), c.content) AS content
        FROM chunks c
    """
    params: list[Any] = []
    if missing_only:
        sql += " LEFT JOIN chunk_embeddings e ON e.chunk_id=c.chunk_id AND e.model=%s WHERE e.chunk_id IS NULL"
        params.append(model_name)
    sql += " ORDER BY c.chunk_id OFFSET %s"
    params.append(offset)
    if limit:
        sql += " LIMIT %s"
        params.append(limit)
    with connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def helper_vectorize_rows(
    rows: list[tuple[str, str]],
    dsn: str | None,
    model_name: str,
    batch_size: int,
    kind: str,
    table: str,
    id_column: str,
    model: Any | None = None,
) -> dict[str, Any]:
    print(f"[vectorize] selected {len(rows)} {kind}", flush=True)
    if model is None:
        print(f"[vectorize] loading model {model_name}...", flush=True)
        model = helper_load_model(model_name)
        print(f"[vectorize] model loaded on {getattr(model, 'device', 'unknown')}", flush=True)
    inserted = 0
    with connect(dsn) as conn:
        with conn.cursor() as cur:
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                texts = [row[1] for row in batch]
                print(f"[vectorize] encoding {kind} {start + 1}-{start + len(batch)}...", flush=True)
                embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
                payload = [
                    (row[0], model_name, int(embeddings.shape[1]), _vector_literal(embedding))
                    for row, embedding in zip(batch, embeddings)
                ]
                cur.executemany(
                    f"""
                    INSERT INTO {table} ({id_column}, model, dim, embedding)
                    VALUES (%s, %s, %s, %s::vector)
                    ON CONFLICT ({id_column}, model) DO UPDATE SET
                        dim=EXCLUDED.dim,
                        embedding=EXCLUDED.embedding,
                        created_at=now()
                    """,
                    payload,
                )
                conn.commit()
                inserted += len(payload)
                print(f"[vectorize] upserted {kind} {inserted}/{len(rows)}", flush=True)
    return {"kind": kind, "model": model_name, "selected": len(rows), "upserted": inserted}


def helper_vectorize_pages(
    fetch_rows: Any,
    fetch_kwargs: dict[str, Any],
    dsn: str | None,
    model_name: str,
    batch_size: int,
    kind: str,
    table: str,
    id_column: str,
    limit: int | None,
    offset: int,
    missing_only: bool,
    page_size: int,
) -> dict[str, Any]:
    if batch_size < 1 or page_size < 1:
        raise ValueError("batch_size and page_size must be positive")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    total = 0
    fetch_offset = offset
    model = None
    while limit is None or total < limit:
        fetch_limit = page_size if limit is None else min(page_size, limit - total)
        page_offset = offset if missing_only else fetch_offset
        rows = fetch_rows(dsn, fetch_limit, page_offset, missing_only, model_name, **fetch_kwargs)
        if not rows:
            break
        if model is None:
            print(f"[vectorize] loading model {model_name}...", flush=True)
            model = helper_load_model(model_name)
            print(f"[vectorize] model loaded on {getattr(model, 'device', 'unknown')}", flush=True)
        result = helper_vectorize_rows(rows, dsn, model_name, batch_size, kind, table, id_column, model)
        total += int(result["upserted"])
        if not missing_only:
            fetch_offset += len(rows)
        if len(rows) < fetch_limit:
            break
    return {"kind": kind, "model": model_name, "selected": total, "upserted": total}


def vectorize_document_cards(
    dsn: str | None = None,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 16,
    limit: int | None = None,
    offset: int = 0,
    missing_only: bool = True,
    skip_schema: bool = False,
    max_text_chars: int = 12000,
    page_size: int = 2000,
) -> dict[str, Any]:
    if not skip_schema:
        ensure_vector_schema(dsn)
    return helper_vectorize_pages(
        helper_fetch_document_cards,
        {"max_text_chars": max_text_chars},
        dsn,
        model_name,
        batch_size,
        "document_cards",
        "document_card_embeddings",
        "doc_id",
        limit,
        offset,
        missing_only,
        page_size,
    )


def vectorize_document_views(
    dsn: str | None = None,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 16,
    limit: int | None = None,
    offset: int = 0,
    missing_only: bool = True,
    skip_schema: bool = False,
    max_text_chars: int = 8000,
    page_size: int = 2000,
) -> dict[str, Any]:
    if not skip_schema:
        ensure_vector_schema(dsn)
    return helper_vectorize_pages(
        helper_fetch_document_views,
        {"max_text_chars": max_text_chars},
        dsn,
        model_name,
        batch_size,
        "document_views",
        "document_view_embeddings",
        "view_id",
        limit,
        offset,
        missing_only,
        page_size,
    )


def vectorize_chunks(
    dsn: str | None = None,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    limit: int | None = None,
    offset: int = 0,
    missing_only: bool = True,
    skip_schema: bool = False,
    page_size: int = 2000,
) -> dict[str, Any]:
    if not skip_schema:
        ensure_vector_schema(dsn)
    return helper_vectorize_pages(
        helper_fetch_chunks,
        {},
        dsn,
        model_name,
        batch_size,
        "chunks",
        "chunk_embeddings",
        "chunk_id",
        limit,
        offset,
        missing_only,
        page_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=["document_cards", "document_views", "chunks"])
    parser.add_argument("--dsn")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--page-size", type=int, default=2000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--include-existing", action="store_true")
    parser.add_argument("--skip-schema", action="store_true")
    parser.add_argument("--max-text-chars", type=int)
    args = parser.parse_args()
    common = {
        "dsn": args.dsn,
        "model_name": args.model,
        "batch_size": args.batch_size,
        "page_size": args.page_size,
        "limit": args.limit,
        "offset": args.offset,
        "missing_only": not args.include_existing,
        "skip_schema": args.skip_schema,
    }
    if args.target == "document_cards":
        if args.max_text_chars is not None:
            common["max_text_chars"] = args.max_text_chars
        payload = vectorize_document_cards(**common)
    elif args.target == "document_views":
        if args.max_text_chars is not None:
            common["max_text_chars"] = args.max_text_chars
        payload = vectorize_document_views(**common)
    else:
        payload = vectorize_chunks(**common)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
