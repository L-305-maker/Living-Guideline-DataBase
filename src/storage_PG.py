from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence

import numpy as np
import psycopg
import torch
from transformers import AutoModel, AutoTokenizer


TABLE_NAME = "medical_guidelines_chunks"
DOCUMENT_MODEL_NAME = "ncbi/MedCPT-Article-Encoder"
QUERY_MODEL_NAME = "ncbi/MedCPT-Query-Encoder"
VECTOR_SIZE = 768
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres")

DEFAULT_EMBED_BATCH_SIZE = 32
DEFAULT_INSERT_BATCH_SIZE = 64
DOCUMENT_MAX_LENGTH = 512
QUERY_MAX_LENGTH = 64


_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DOC_TOKENIZER: Optional[Any] = None
_DOC_MODEL: Optional[torch.nn.Module] = None
_QUERY_TOKENIZER: Optional[Any] = None
_QUERY_MODEL: Optional[torch.nn.Module] = None


def get_connection() -> psycopg.Connection:
    """连接 PostgreSQL，并给出清晰错误信息。"""
    try:
        return psycopg.connect(DATABASE_URL)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"无法连接 PostgreSQL。请检查 DATABASE_URL={DATABASE_URL!r}。原始错误: {exc}"
        ) from exc


def _load_document_encoder() -> tuple[Any, torch.nn.Module]:
    """文档入库必须使用 MedCPT Article Encoder。"""
    global _DOC_TOKENIZER, _DOC_MODEL
    if _DOC_TOKENIZER is None or _DOC_MODEL is None:
        print(f"Loading document encoder on {_DEVICE}: {DOCUMENT_MODEL_NAME}", file=sys.stderr)
        _DOC_TOKENIZER = AutoTokenizer.from_pretrained(DOCUMENT_MODEL_NAME)
        _DOC_MODEL = AutoModel.from_pretrained(DOCUMENT_MODEL_NAME).to(_DEVICE).eval()
    return _DOC_TOKENIZER, _DOC_MODEL


def _load_query_encoder() -> tuple[Any, torch.nn.Module]:
    """查询检索必须使用 MedCPT Query Encoder，不能和文档编码器混用。"""
    global _QUERY_TOKENIZER, _QUERY_MODEL
    if _QUERY_TOKENIZER is None or _QUERY_MODEL is None:
        print(f"Loading query encoder on {_DEVICE}: {QUERY_MODEL_NAME}", file=sys.stderr)
        _QUERY_TOKENIZER = AutoTokenizer.from_pretrained(QUERY_MODEL_NAME)
        _QUERY_MODEL = AutoModel.from_pretrained(QUERY_MODEL_NAME).to(_DEVICE).eval()
    return _QUERY_TOKENIZER, _QUERY_MODEL


def _encode_texts(
    texts: Sequence[str],
    tokenizer: Any,
    model: torch.nn.Module,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    vectors: List[np.ndarray] = []
    clean_texts = [str(text or "") for text in texts]

    with torch.no_grad():
        for start in range(0, len(clean_texts), batch_size):
            batch = clean_texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(_DEVICE)
            outputs = model(**encoded)
            cls_vec = outputs.last_hidden_state[:, 0, :]
            cls_vec = torch.nn.functional.normalize(cls_vec, p=2, dim=1)
            vectors.append(cls_vec.cpu().numpy().astype(np.float32))

    if not vectors:
        return np.empty((0, VECTOR_SIZE), dtype=np.float32)
    return np.vstack(vectors)


def encode_documents(texts: Sequence[str], batch_size: int = DEFAULT_EMBED_BATCH_SIZE) -> np.ndarray:
    tokenizer, model = _load_document_encoder()
    return _encode_texts(texts, tokenizer, model, DOCUMENT_MAX_LENGTH, batch_size)


def encode_query(text: str) -> np.ndarray:
    tokenizer, model = _load_query_encoder()
    return _encode_texts([text], tokenizer, model, QUERY_MAX_LENGTH, 1)[0]


def _sql_identifier(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"非法 SQL 标识符: {name}")
    return name


def create_schema(recreate: bool = False) -> None:
    """创建 pgvector 扩展、数据表和常用过滤索引。"""
    table = _sql_identifier(TABLE_NAME)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            if recreate:
                cur.execute(f"DROP TABLE IF EXISTS {table}")

            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    id UUID PRIMARY KEY,
                    text TEXT NOT NULL,
                    title TEXT,
                    url TEXT,
                    issuer TEXT,
                    source TEXT,
                    published_date TEXT,
                    effective_date TEXT,
                    year INTEGER,
                    medical_topic JSONB,
                    chunk_index INTEGER,
                    quality JSONB,
                    recommendation JSONB,
                    metadata JSONB,
                    embedding vector({VECTOR_SIZE}) NOT NULL
                )
                """
            )
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_year ON {table} (year)")
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_issuer ON {table} (issuer)")
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_source ON {table} (source)")
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_effective_date ON {table} (effective_date)")
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_metadata_gin ON {table} USING GIN (metadata)")
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{table}_embedding_cosine
                ON {table}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
                """
            )
        conn.commit()
    print(f"PostgreSQL table ready: {TABLE_NAME}")


def load_chunks(path: str | Path) -> Generator[Dict[str, Any], None, None]:
    jsonl_path = Path(path)
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"跳过 JSON 解析失败行 {line_no}: {exc}", file=sys.stderr)
                continue
            if isinstance(item, dict):
                yield item


def _extract_year(value: Any) -> Optional[int]:
    if value is None:
        return None
    match = re.search(r"(19|20)\d{2}", str(value))
    return int(match.group(0)) if match else None


def _first_non_empty(*values: Any) -> str:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _normalize_medical_topic(value: Any) -> List[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _make_row(item: Dict[str, Any], vector: np.ndarray) -> Dict[str, Any]:
    metadata = item.get("metadata") or {}
    text = str(item.get("page_content") or "")
    published_date = str(metadata.get("published_date") or "")
    effective_date = _first_non_empty(metadata.get("effective_date"), published_date)
    medical_topic = _normalize_medical_topic(metadata.get("medical_topic"))
    chunk_index = metadata.get("chunk_index")
    text_digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    raw_id = "|".join(
        [
            str(metadata.get("url") or ""),
            str(metadata.get("title") or ""),
            str(chunk_index or ""),
            text_digest,
        ]
    )

    return {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, raw_id)),
        "text": text,
        "title": str(metadata.get("title") or ""),
        "url": str(metadata.get("url") or ""),
        "issuer": str(metadata.get("issuer") or ""),
        "source": str(metadata.get("source") or ""),
        "published_date": published_date,
        "effective_date": effective_date,
        "year": _extract_year(effective_date),
        "medical_topic": medical_topic,
        "chunk_index": chunk_index if isinstance(chunk_index, int) else _safe_int(chunk_index),
        "quality": metadata.get("quality"),
        "recommendation": metadata.get("recommendation"),
        "metadata": metadata,
        "embedding": _vector_to_pg(vector),
    }


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _vector_to_pg(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in vector) + "]"


def _iter_batches(items: Iterable[Dict[str, Any]], batch_size: int) -> Generator[List[Dict[str, Any]], None, None]:
    batch: List[Dict[str, Any]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _insert_rows(conn: psycopg.Connection, rows: List[Dict[str, Any]]) -> None:
    table = _sql_identifier(TABLE_NAME)
    sql = f"""
        INSERT INTO {table} (
            id, text, title, url, issuer, source, published_date, effective_date, year,
            medical_topic, chunk_index, quality, recommendation, metadata, embedding
        )
        VALUES (
            %(id)s, %(text)s, %(title)s, %(url)s, %(issuer)s, %(source)s,
            %(published_date)s, %(effective_date)s, %(year)s,
            %(medical_topic)s::jsonb, %(chunk_index)s, %(quality)s::jsonb,
            %(recommendation)s::jsonb, %(metadata)s::jsonb, %(embedding)s::vector
        )
        ON CONFLICT (id) DO UPDATE SET
            text = EXCLUDED.text,
            title = EXCLUDED.title,
            url = EXCLUDED.url,
            issuer = EXCLUDED.issuer,
            source = EXCLUDED.source,
            published_date = EXCLUDED.published_date,
            effective_date = EXCLUDED.effective_date,
            year = EXCLUDED.year,
            medical_topic = EXCLUDED.medical_topic,
            chunk_index = EXCLUDED.chunk_index,
            quality = EXCLUDED.quality,
            recommendation = EXCLUDED.recommendation,
            metadata = EXCLUDED.metadata,
            embedding = EXCLUDED.embedding
    """
    params = []
    for row in rows:
        params.append(
            {
                **row,
                "medical_topic": json.dumps(row["medical_topic"], ensure_ascii=False),
                "quality": json.dumps(row["quality"], ensure_ascii=False),
                "recommendation": json.dumps(row["recommendation"], ensure_ascii=False),
                "metadata": json.dumps(row["metadata"], ensure_ascii=False),
            }
        )
    with conn.cursor() as cur:
        cur.executemany(sql, params)


def ingest(jsonl_path: str | Path, batch_size: int = DEFAULT_INSERT_BATCH_SIZE) -> None:
    create_schema(recreate=False)
    total = 0

    with get_connection() as conn:
        for batch in _iter_batches(load_chunks(jsonl_path), batch_size):
            valid_items = [item for item in batch if str(item.get("page_content") or "").strip()]
            if not valid_items:
                continue

            vectors = encode_documents([str(item.get("page_content") or "") for item in valid_items])
            rows = [_make_row(item, vector) for item, vector in zip(valid_items, vectors)]
            try:
                _insert_rows(conn, rows)
                conn.commit()
            except Exception as exc:  # noqa: BLE001
                conn.rollback()
                raise RuntimeError(f"PostgreSQL 入库失败，已处理 {total} 条后中断: {exc}") from exc

            total += len(rows)
            print(f"已入库 {total} chunks")

    print(f"PostgreSQL 入库完成，总计 {total} chunks")


def _build_where(
    year: Optional[int] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    issuer: Optional[str] = None,
    source: Optional[str] = None,
) -> tuple[str, Dict[str, Any]]:
    clauses: List[str] = []
    params: Dict[str, Any] = {}

    if year is not None:
        clauses.append("year = %(year)s")
        params["year"] = year
    else:
        if year_from is not None:
            clauses.append("year >= %(year_from)s")
            params["year_from"] = year_from
        if year_to is not None:
            clauses.append("year <= %(year_to)s")
            params["year_to"] = year_to
    if issuer:
        clauses.append("issuer = %(issuer)s")
        params["issuer"] = issuer
    if source:
        clauses.append("source = %(source)s")
        params["source"] = source

    return ("WHERE " + " AND ".join(clauses), params) if clauses else ("", params)


def search(
    query: str,
    top_k: int = 5,
    year: Optional[int] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    issuer: Optional[str] = None,
    source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    table = _sql_identifier(TABLE_NAME)
    query_vector = _vector_to_pg(encode_query(query))
    where_sql, params = _build_where(year=year, year_from=year_from, year_to=year_to, issuer=issuer, source=source)
    params["query_vector"] = query_vector
    params["limit"] = top_k

    sql = f"""
        SELECT
            id,
            1 - (embedding <=> %(query_vector)s::vector) AS score,
            text,
            title,
            url,
            issuer,
            source,
            published_date,
            effective_date,
            year,
            medical_topic,
            chunk_index,
            quality,
            recommendation,
            metadata
        FROM {table}
        {where_sql}
        ORDER BY embedding <=> %(query_vector)s::vector
        LIMIT %(limit)s
    """

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"PostgreSQL 检索失败: {exc}") from exc

    results: List[Dict[str, Any]] = []
    for row in rows:
        (
            row_id,
            score,
            text,
            title,
            url,
            issuer_value,
            source_value,
            published_date,
            effective_date,
            row_year,
            medical_topic,
            chunk_index,
            quality,
            recommendation,
            metadata,
        ) = row
        results.append(
            {
                "id": str(row_id),
                "score": float(score),
                "payload": {
                    "text": text,
                    "title": title,
                    "url": url,
                    "issuer": issuer_value,
                    "source": source_value,
                    "published_date": published_date,
                    "effective_date": effective_date,
                    "year": row_year,
                    "medical_topic": medical_topic,
                    "chunk_index": chunk_index,
                    "quality": quality,
                    "recommendation": recommendation,
                    "metadata": metadata,
                },
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="MedCPT + PostgreSQL/pgvector 入库与检索")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="读取 JSONL chunks 并写入 PostgreSQL")
    ingest_parser.add_argument("--input", default="outputs/chunk.jsonl", help="chunk JSONL 路径")
    ingest_parser.add_argument("--batch-size", type=int, default=DEFAULT_INSERT_BATCH_SIZE, help="insert 批大小")
    ingest_parser.add_argument("--recreate", action="store_true", help="重建表后再入库")

    search_parser = subparsers.add_parser("search", help="检索医学指南 chunks")
    search_parser.add_argument("query", help="查询文本")
    search_parser.add_argument("--top-k", type=int, default=5, help="返回条数")
    search_parser.add_argument("--year", type=int, default=None, help="仅检索指定年份")
    search_parser.add_argument("--year-from", type=int, default=None, help="仅检索该年份及之后的数据")
    search_parser.add_argument("--year-to", type=int, default=None, help="仅检索该年份及之前的数据")
    search_parser.add_argument("--issuer", default=None, help="按 issuer 精确过滤")
    search_parser.add_argument("--source", default=None, help="按 source 精确过滤")

    args = parser.parse_args()

    if args.command == "ingest":
        if args.recreate:
            create_schema(recreate=True)
        ingest(args.input, batch_size=args.batch_size)
        return

    if args.command == "search":
        results = search(
            args.query,
            top_k=args.top_k,
            year=args.year,
            year_from=args.year_from,
            year_to=args.year_to,
            issuer=args.issuer,
            source=args.source,
        )
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
