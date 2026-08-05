"""Shared query embedding helpers for PostgreSQL vector retrieval."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Iterable


# 查询向量模型必须与数据库向量记录中的 model 字段一致，否则相似度不可比较。
# 默认使用 Qwen3-Embedding-8B；经 PG_VECTOR_MODEL 可切换为其他 sentence-transformers 模型。
DEFAULT_MODEL = os.getenv("PG_VECTOR_MODEL", "Qwen/Qwen3-Embedding-8B")
DEFAULT_DIM = 1024


def _parse_matryoshka_dim() -> int:
    raw = os.getenv("PG_VECTOR_MATRYOSHKA_DIM", "1024")
    try:
        return int(raw)
    except ValueError:
        return 1024


# Qwen3-Embedding 支持 MRL(Matryoshka Representation Learning)，可输出 32-4096 维。
# 默认 1024 与现有 vector(1024) 表结构一致；若经 PG_VECTOR_MATRYOSHKA_DIM 修改维度，
# 必须同步修改 postgres_store.py VECTOR_SCHEMA_SQL 中三处 embedding vector(N)。
DEFAULT_MATRYOSHKA_DIM = _parse_matryoshka_dim()
# 远程服务器首次部署通常需要联网下载；稳定运行后可用环境变量切换为仅本地缓存。
DEFAULT_LOCAL_FILES_ONLY = os.getenv("PG_VECTOR_LOCAL_ONLY", "0").strip().lower() in {"1", "true", "yes", "y"}


@lru_cache(maxsize=4)
def load_model(model_name: str = DEFAULT_MODEL):
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is required for embedding models: python -m pip install sentence-transformers") from exc
    # 缓存模型实例，避免每次请求都重复加载权重并额外占用显存。
    model_kwargs: dict[str, Any] = {}
    dtype = os.getenv("PG_VECTOR_MODEL_DTYPE", "").strip()
    if dtype:
        model_kwargs["torch_dtype"] = dtype
    attn = os.getenv("PG_VECTOR_MODEL_ATTN", "").strip()
    if attn:
        model_kwargs["attn_implementation"] = attn
    return SentenceTransformer(
        model_name,
        local_files_only=DEFAULT_LOCAL_FILES_ONLY,
        model_kwargs=model_kwargs or None,
        tokenizer_kwargs={"padding_side": "left"},
    )


def encode_with_model(model: Any, texts: list[str], *, model_name: str = DEFAULT_MODEL):
    """Encode texts with the shared normalization + matryoshka settings.

    ``matryoshka_dim`` is passed for Qwen3-Embedding by default. Setting
    PG_VECTOR_MATRYOSHKA_DIM explicitly forces it for any model (intended for
    other MRL-capable models; non-MRL models may reject the argument).
    """
    kwargs: dict[str, Any] = dict(normalize_embeddings=True, convert_to_numpy=True)
    if "Qwen3-Embedding" in model_name or os.getenv("PG_VECTOR_MATRYOSHKA_DIM"):
        kwargs["matryoshka_dim"] = DEFAULT_MATRYOSHKA_DIM
    return model.encode(texts, **kwargs)


def vector_literal(values: Iterable[float]) -> str:
    # pgvector 接受方括号文本格式；固定小数位可减小 SQL 参数体积并保持结果稳定。
    return "[" + ",".join(f"{float(value):.8f}" for value in values) + "]"


def query_vector_literal(query: str, model_name: str = DEFAULT_MODEL) -> str:
    model = load_model(model_name)
    # 与建库阶段保持相同的 L2 归一化设置，使余弦距离计算口径一致。
    embedding = encode_with_model(model, [query], model_name=model_name)[0]
    return vector_literal(embedding)
