"""Minimal HTTP client for the embedding and reranking vLLM services."""

from __future__ import annotations

import math
import os
import threading
from typing import Any

import requests


DEFAULT_EMBEDDING_BASE_URL = "http://127.0.0.1:8001"
DEFAULT_RERANKER_BASE_URL = "http://127.0.0.1:8002"

_THREAD_LOCAL = threading.local()


def embed_texts(texts: list[str], *, model_name: str, dimensions: int) -> list[list[float]]:
    """Return normalized embeddings in input order through vLLM's OpenAI API."""

    if not texts:
        return []
    served_model = os.getenv("VLLM_EMBEDDING_MODEL", model_name).strip() or model_name
    payload = {
        "model": served_model,
        "input": texts,
        "encoding_format": "float",
        "dimensions": dimensions,
    }
    response = helper_post_json(
        os.getenv("VLLM_EMBEDDING_BASE_URL", DEFAULT_EMBEDDING_BASE_URL),
        "/v1/embeddings",
        payload,
        api_key_env="VLLM_EMBEDDING_API_KEY",
    )
    items = response.get("data")
    if not isinstance(items, list) or len(items) != len(texts):
        raise RuntimeError(f"vLLM embedding response count mismatch: expected {len(texts)}")

    vectors: list[list[float] | None] = [None] * len(texts)
    for item in items:
        if not isinstance(item, dict):
            raise RuntimeError("vLLM embedding response contains a non-object item")
        index = item.get("index")
        raw_vector = item.get("embedding")
        if not isinstance(index, int) or not 0 <= index < len(texts) or not isinstance(raw_vector, list):
            raise RuntimeError("vLLM embedding response has an invalid index or embedding")
        vector = [float(value) for value in raw_vector]
        if len(vector) != dimensions:
            raise RuntimeError(
                f"vLLM embedding dimension mismatch: expected {dimensions}, got {len(vector)}"
            )
        if not all(math.isfinite(value) for value in vector):
            raise RuntimeError("vLLM embedding response contains a non-finite value")
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            raise RuntimeError("vLLM embedding response contains a zero vector")
        vectors[index] = [value / norm for value in vector]

    if any(vector is None for vector in vectors):
        raise RuntimeError("vLLM embedding response is missing an input index")
    return [vector for vector in vectors if vector is not None]


def rerank_texts(query: str, documents: list[str], *, model_name: str) -> list[float]:
    """Return one relevance score per document in the original input order."""

    if not documents:
        return []
    served_model = os.getenv("VLLM_RERANKER_MODEL", model_name).strip() or model_name
    response = helper_post_json(
        os.getenv("VLLM_RERANKER_BASE_URL", DEFAULT_RERANKER_BASE_URL),
        "/v1/rerank",
        {
            "model": served_model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
        },
        api_key_env="VLLM_RERANKER_API_KEY",
    )
    items = response.get("results")
    if not isinstance(items, list) or len(items) != len(documents):
        raise RuntimeError(f"vLLM rerank response count mismatch: expected {len(documents)}")

    scores: list[float | None] = [None] * len(documents)
    for item in items:
        if not isinstance(item, dict):
            raise RuntimeError("vLLM rerank response contains a non-object item")
        index = item.get("index")
        raw_score = item.get("relevance_score")
        if not isinstance(index, int) or not 0 <= index < len(documents):
            raise RuntimeError("vLLM rerank response has an invalid index")
        score = float(raw_score)
        if not math.isfinite(score):
            raise RuntimeError("vLLM rerank response contains a non-finite score")
        scores[index] = score

    if any(score is None for score in scores):
        raise RuntimeError("vLLM rerank response is missing an input index")
    return [score for score in scores if score is not None]


def helper_post_json(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    *,
    api_key_env: str,
) -> dict[str, Any]:
    """POST JSON with a per-thread pooled session and bounded timeouts."""

    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        # 模型服务属于本机/内网基础设施，不应继承进程级 HTTP(S)_PROXY。
        session.trust_env = False
        _THREAD_LOCAL.session = session
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv(api_key_env) or os.getenv("VLLM_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    connect_timeout = helper_env_positive_float("VLLM_CONNECT_TIMEOUT_SECONDS", 5.0)
    request_timeout = helper_env_positive_float("VLLM_REQUEST_TIMEOUT_SECONDS", 120.0)
    try:
        http_response = session.post(
            f"{base_url.rstrip('/')}{path}",
            json=payload,
            headers=headers,
            timeout=(connect_timeout, request_timeout),
        )
        http_response.raise_for_status()
        body = http_response.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(f"vLLM request failed for {path}: {exc}") from exc
    if not isinstance(body, dict):
        raise RuntimeError(f"vLLM response for {path} is not a JSON object")
    return body


def helper_env_positive_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default
