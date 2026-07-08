"""Shared query embedding helpers for PostgreSQL vector retrieval."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable


DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_DIM = 1024


@lru_cache(maxsize=4)
def load_model(model_name: str = DEFAULT_MODEL):
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is required for bge-m3: python -m pip install sentence-transformers") from exc
    return SentenceTransformer(model_name)


def vector_literal(values: Iterable[float]) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in values) + "]"


def query_vector_literal(query: str, model_name: str = DEFAULT_MODEL) -> str:
    model = load_model(model_name)
    embedding = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    return vector_literal(embedding)
