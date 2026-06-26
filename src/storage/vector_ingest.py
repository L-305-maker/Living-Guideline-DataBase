from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Protocol, Sequence

from src.common.process_jsonl import iter_jsonl
from src.storage.connection import get_connection
from src.storage.vector_schema import DEFAULT_VECTOR_DIMENSIONS, EMBEDDING_TABLE, create_vector_schema


JsonDict = Dict[str, Any]


class EmbeddingProvider(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        ...


def print_json(value: Any) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2)
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(text.encode("utf-8") + b"\n")


@dataclass(frozen=True)
class FakeEmbeddingProvider:
    """Deterministic local embeddings for database-path smoke tests.

    The vectors are not semantically meaningful enough for production RAG. They
    exist so pgvector schema, ingest, upsert, and search can be tested without
    network access, API keys, or embedding cost.
    """

    dimensions: int = DEFAULT_VECTOR_DIMENSIONS

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        return [fake_embedding(text, dimensions=self.dimensions) for text in texts]


@dataclass
class BGEM3EmbeddingProvider:
    model_name: str = "BAAI/bge-m3"
    batch_size: int = 12
    max_length: int = 8192
    use_fp16: bool = True

    def __post_init__(self) -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "BGE-M3 embeddings require FlagEmbedding. Install it with: pip install -U FlagEmbedding"
            ) from exc
        self._model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16)

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        output = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return [list(map(float, vector)) for vector in output["dense_vecs"]]


def text_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def fake_embedding(text: str, *, dimensions: int = DEFAULT_VECTOR_DIMENSIONS) -> List[float]:
    if dimensions <= 0:
        raise ValueError("Vector dimensions must be positive.")
    vector = [0.0] * dimensions
    tokens = text_tokens(text) or [hashlib.sha256(text.encode("utf-8")).hexdigest()]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        weight = 1.0 + (digest[5] / 255.0)
        vector[index] += sign * weight
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in values) + "]"


def require_queue_field(row: JsonDict, field: str) -> str:
    value = row.get(field)
    if value in (None, ""):
        raise ValueError(f"Embedding queue row missing required field: {field}")
    return str(value)


def prepare_embedding_record(row: JsonDict, embedding: Sequence[float], *, dimensions: int = DEFAULT_VECTOR_DIMENSIONS) -> JsonDict:
    if len(embedding) != dimensions:
        raise ValueError(f"Embedding dimension mismatch: expected {dimensions}, got {len(embedding)}")
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "vector_id": require_queue_field(row, "vector_id"),
        "entity_type": require_queue_field(row, "entity_type"),
        "entity_id": require_queue_field(row, "entity_id"),
        "collection": require_queue_field(row, "collection"),
        "text_hash": require_queue_field(row, "text_hash"),
        "embedding_model": require_queue_field(row, "embedding_model"),
        "embedding_version": require_queue_field(row, "embedding_version"),
        "embedding": vector_literal(embedding),
        "text_for_embedding": require_queue_field(row, "text_for_embedding"),
        "metadata": json.dumps(metadata, ensure_ascii=False),
        "source_file": row.get("source_file"),
    }


def load_queue(path: str | Path) -> List[JsonDict]:
    return list(iter_jsonl(path))


def upsert_embedding_records(conn: Any, rows: Sequence[JsonDict]) -> int:
    if not rows:
        return 0
    sql = f"""
        INSERT INTO {EMBEDDING_TABLE} (
            vector_id,
            entity_type,
            entity_id,
            collection,
            text_hash,
            embedding_model,
            embedding_version,
            embedding,
            text_for_embedding,
            metadata,
            source_file
        )
        VALUES (
            %(vector_id)s,
            %(entity_type)s,
            %(entity_id)s,
            %(collection)s,
            %(text_hash)s,
            %(embedding_model)s,
            %(embedding_version)s,
            %(embedding)s::vector,
            %(text_for_embedding)s,
            %(metadata)s::jsonb,
            %(source_file)s
        )
        ON CONFLICT (vector_id) DO UPDATE SET
            entity_type = EXCLUDED.entity_type,
            entity_id = EXCLUDED.entity_id,
            collection = EXCLUDED.collection,
            text_hash = EXCLUDED.text_hash,
            embedding_model = EXCLUDED.embedding_model,
            embedding_version = EXCLUDED.embedding_version,
            embedding = EXCLUDED.embedding,
            text_for_embedding = EXCLUDED.text_for_embedding,
            metadata = EXCLUDED.metadata,
            source_file = EXCLUDED.source_file,
            updated_at = CURRENT_TIMESTAMP
    """
    with conn.cursor() as cur:
        cur.executemany(sql, list(rows))
    return len(rows)


def ingest_embedding_queue(
    queue_path: str | Path,
    *,
    provider: EmbeddingProvider | None = None,
    dimensions: int = DEFAULT_VECTOR_DIMENSIONS,
    create_schema_first: bool = True,
) -> JsonDict:
    if create_schema_first:
        create_vector_schema(dimensions=dimensions)
    queue_rows = load_queue(queue_path)
    provider = provider or FakeEmbeddingProvider(dimensions=dimensions)
    embeddings = provider.embed_texts([str(row.get("text_for_embedding") or "") for row in queue_rows])
    db_rows = [
        prepare_embedding_record(row, embedding, dimensions=dimensions)
        for row, embedding in zip(queue_rows, embeddings)
    ]
    with get_connection() as conn:
        try:
            count = upsert_embedding_records(conn, db_rows)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return {
        "status": "ingested",
        "queue_path": str(Path(queue_path)),
        "input_rows": len(queue_rows),
        "upserted_rows": count,
        "dimensions": dimensions,
    }


def search_embeddings(
    query_text: str,
    *,
    collection: str,
    provider: EmbeddingProvider | None = None,
    dimensions: int = DEFAULT_VECTOR_DIMENSIONS,
    top_k: int = 5,
) -> List[JsonDict]:
    provider = provider or FakeEmbeddingProvider(dimensions=dimensions)
    query_embedding = provider.embed_texts([query_text])[0]
    sql = f"""
        SELECT
            vector_id,
            entity_type,
            entity_id,
            collection,
            text_hash,
            embedding_model,
            embedding_version,
            text_for_embedding,
            metadata,
            source_file,
            embedding <-> %s::vector AS distance
        FROM {EMBEDDING_TABLE}
        WHERE collection = %s
        ORDER BY embedding <-> %s::vector
        LIMIT %s
    """
    literal = vector_literal(query_embedding)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (literal, collection, literal, top_k))
            rows = cur.fetchall()
    return [
        {
            "vector_id": row[0],
            "entity_type": row[1],
            "entity_id": row[2],
            "collection": row[3],
            "text_hash": row[4],
            "embedding_model": row[5],
            "embedding_version": row[6],
            "text_for_embedding": row[7],
            "metadata": row[8],
            "source_file": row[9],
            "distance": float(row[10]),
        }
        for row in rows
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pgvector ingest and smoke-search for embedding_queue.jsonl.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create pgvector extension and embedding_records table")
    init_parser.add_argument("--dimensions", type=int, default=DEFAULT_VECTOR_DIMENSIONS)
    init_parser.add_argument("--recreate", action="store_true", help="Drop and recreate only the embedding_records index table")

    ingest_parser = subparsers.add_parser("ingest", help="Embed and ingest an embedding queue")
    ingest_parser.add_argument("--queue", required=True)
    ingest_parser.add_argument("--dimensions", type=int, default=DEFAULT_VECTOR_DIMENSIONS)
    ingest_parser.add_argument("--provider", choices=("bge-m3", "fake"), default="bge-m3")
    ingest_parser.add_argument("--fake-embeddings", action="store_true", help="Use deterministic local fake embeddings")
    ingest_parser.add_argument("--bge-model", default="BAAI/bge-m3")
    ingest_parser.add_argument("--batch-size", type=int, default=12)
    ingest_parser.add_argument("--max-length", type=int, default=8192)
    ingest_parser.add_argument("--no-fp16", action="store_true")

    search_parser = subparsers.add_parser("search", help="Search pgvector embedding records")
    search_parser.add_argument("--query-text", required=True)
    search_parser.add_argument("--collection", required=True)
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.add_argument("--dimensions", type=int, default=DEFAULT_VECTOR_DIMENSIONS)
    search_parser.add_argument("--provider", choices=("bge-m3", "fake"), default="bge-m3")
    search_parser.add_argument("--fake-embeddings", action="store_true", help="Use deterministic local fake embeddings")
    search_parser.add_argument("--bge-model", default="BAAI/bge-m3")
    search_parser.add_argument("--batch-size", type=int, default=12)
    search_parser.add_argument("--max-length", type=int, default=8192)
    search_parser.add_argument("--no-fp16", action="store_true")

    return parser.parse_args()


def provider_from_args(args: argparse.Namespace) -> EmbeddingProvider:
    provider_name = "fake" if getattr(args, "fake_embeddings", False) else args.provider
    if provider_name == "fake":
        return FakeEmbeddingProvider(dimensions=args.dimensions)
    return BGEM3EmbeddingProvider(
        model_name=args.bge_model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_fp16=not args.no_fp16,
    )


def main() -> None:
    args = parse_args()
    if args.command == "init":
        create_vector_schema(dimensions=args.dimensions, recreate=args.recreate)
        return
    if args.command == "ingest":
        summary = ingest_embedding_queue(
            args.queue,
            provider=provider_from_args(args),
            dimensions=args.dimensions,
        )
        print_json(summary)
        return
    if args.command == "search":
        results = search_embeddings(
            args.query_text,
            collection=args.collection,
            provider=provider_from_args(args),
            dimensions=args.dimensions,
            top_k=args.top_k,
        )
        print_json(results)


if __name__ == "__main__":
    main()
