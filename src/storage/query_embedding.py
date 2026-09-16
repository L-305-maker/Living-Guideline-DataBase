# 查询向量化的共享工具，被 PostgreSQL 检索链路复用。
#
# 关键不变量：
#   查询向量所用模型必须与数据库中已有的 embedding 行 model 字段一致，
#   否则余弦距离与原向量的口径不同，结果不可比较。

from __future__ import annotations

import os
from typing import Iterable

from src.models.vllm_client import embed_texts


# 默认使用 Qwen3-Embedding-8B；服务端模型可通过 VLLM_EMBEDDING_MODEL 覆盖。
# 注意：换模型必须重新向量化，否则旧向量记录不会被使用，但 PG 中仍会残留。
DEFAULT_MODEL = os.getenv("PG_VECTOR_MODEL", "Qwen/Qwen3-Embedding-8B")

# PostgreSQL 的 vector 列与编码器共同使用这一固定维度。
EMBEDDING_DIM = 1024

def encode_texts(texts: list[str], *, model_name: str = DEFAULT_MODEL) -> list[list[float]]:
    """通过独立 vLLM 服务编码并返回归一化的 1024 维向量。"""

    return embed_texts(texts, model_name=model_name, dimensions=EMBEDDING_DIM)


def vector_literal(values: Iterable[float]) -> str:
    """把 Python 数值序列渲染成 pgvector 接受的方括号文本格式。

    固定 8 位小数的原因：
    - 控制 SQL 参数体积，便于复用 prepared statement；
    - 避免不同进程/不同 float 实现产生的微小差异导致 hash 抖动（影响缓存与调试可复现性）。
    """
    return "[" + ",".join(f"{float(value):.8f}" for value in values) + "]"


def query_vector_literal(query: str, model_name: str = DEFAULT_MODEL) -> str:
    """通过 vLLM 编码单条查询并返回 pgvector 字面量。"""

    embedding = encode_texts([query], model_name=model_name)[0]
    return vector_literal(embedding)
