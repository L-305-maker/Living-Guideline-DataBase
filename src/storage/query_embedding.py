"""Shared query embedding helpers for PostgreSQL vector retrieval."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable


# 查询向量模型必须与数据库向量记录中的 model 字段一致，否则相似度不可比较。
DEFAULT_MODEL = os.getenv("PG_VECTOR_MODEL", "BAAI/bge-m3")
DEFAULT_DIM = 1024
# 远程服务器首次部署通常需要联网下载；稳定运行后可用环境变量切换为仅本地缓存。
DEFAULT_LOCAL_FILES_ONLY = os.getenv("PG_VECTOR_LOCAL_ONLY", "0").strip().lower() in {"1", "true", "yes", "y"}


@lru_cache(maxsize=4)
def load_model(model_name: str = DEFAULT_MODEL):
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is required for bge-m3: python -m pip install sentence-transformers") from exc
    # 缓存模型实例，避免每次请求都重复加载权重并占用额外显存。
    return SentenceTransformer(model_name, local_files_only=DEFAULT_LOCAL_FILES_ONLY)


def vector_literal(values: Iterable[float]) -> str:
    # pgvector 接受方括号文本格式；固定小数位可减小 SQL 参数体积并保持结果稳定。
    return "[" + ",".join(f"{float(value):.8f}" for value in values) + "]"


def query_vector_literal(query: str, model_name: str = DEFAULT_MODEL) -> str:
    model = load_model(model_name)
    # 与建库阶段保持相同的 L2 归一化设置，使余弦距离计算口径一致。
    embedding = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    return vector_literal(embedding)
