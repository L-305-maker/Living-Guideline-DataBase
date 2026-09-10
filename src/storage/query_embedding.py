# 查询向量化的共享工具，被 PostgreSQL 检索链路复用。
#
# 关键不变量：
#   查询向量所用模型必须与数据库中已有的 embedding 行 model 字段一致，
#   否则余弦距离与原向量的口径不同，结果不可比较。

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Iterable


# 默认使用 Qwen3-Embedding-8B；可经 PG_VECTOR_MODEL 切换到其它 sentence-transformers 模型。
# 注意：换模型必须重新向量化，否则旧向量记录不会被使用，但 PG 中仍会残留。
DEFAULT_MODEL = os.getenv("PG_VECTOR_MODEL", "Qwen/Qwen3-Embedding-8B")

# PostgreSQL 的 vector 列与编码器共同使用这一固定维度。
EMBEDDING_DIM = 1024

# 远程服务器稳定运行后通常只读本地缓存；首次部署或换模型需要联网拉权重。
# 用环境变量 PG_VECTOR_LOCAL_ONLY=1 切换，避免每次启动都触发网络请求。
DEFAULT_LOCAL_FILES_ONLY = os.getenv("PG_VECTOR_LOCAL_ONLY", "0").strip().lower() in {"1", "true", "yes", "y"}


@lru_cache(maxsize=4)
def load_model(model_name: str = DEFAULT_MODEL):
    """加载 sentence-transformers 模型实例并按 lru_cache 缓存。

    - maxsize=4：同时支持 4 种不同模型名共存；切换模型不需要重启进程。
    - dtype / attn_implementation 由环境变量透传：分别对应 PG_VECTOR_MODEL_DTYPE
      （bf16 / float16 / float32）和 PG_VECTOR_MODEL_ATTN（flash_attention_2 等），
      避免硬编码，方便远端部署调优。
    - tokenizer padding_side=left：BGE/Qwen 系模型对齐左填充以稳定检索结果。
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for embedding models: "
            "python -m pip install sentence-transformers"
        ) from exc

    # 把 dtype / attention 实现通过 model_kwargs 透传给底层 transformers AutoModel。
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
    """用已加载的模型对文本列表做向量化。

    关键设置：
    - normalize_embeddings=True：输出 L2 归一化向量，使余弦距离退化为内积，
      与 pgvector 的 vector_cosine_ops 索引计算口径一致。
    - convert_to_numpy=True：返回 numpy.ndarray，方便后续写入 pgvector。
    - truncate_dim：Qwen3-Embedding 固定输出数据库列要求的 1024 维；
      非 MRL 模型不传此参数。
    """
    kwargs: dict[str, Any] = dict(normalize_embeddings=True, convert_to_numpy=True)
    if "Qwen3-Embedding" in model_name:
        kwargs["truncate_dim"] = EMBEDDING_DIM
    return model.encode(texts, **kwargs)


def vector_literal(values: Iterable[float]) -> str:
    """把 Python 数值序列渲染成 pgvector 接受的方括号文本格式。

    固定 8 位小数的原因：
    - 控制 SQL 参数体积，便于复用 prepared statement；
    - 避免不同进程/不同 float 实现产生的微小差异导致 hash 抖动（影响缓存与调试可复现性）。
    """
    return "[" + ",".join(f"{float(value):.8f}" for value in values) + "]"


def query_vector_literal(query: str, model_name: str = DEFAULT_MODEL) -> str:
    """将单条查询文本编码为可直接嵌入 SQL 的 pgvector 字面量字符串。

    与建库时使用的 encode_with_model 完全对齐: 相同的 normalize_embeddings、
    相同的 truncate_dim，保证查询向量与库内向量在同一向量空间中可计算余弦距离。

    高并发 (P2): 默认走 GPUQueue 单 producer 入口, 跨请求合并成 batch 一次调
    SentenceTransformer.encode; 仅在 GPUQueue 缺席时 (如 env GPU_QUEUE_DISABLED=1)
    回退到直接 model.encode([query])。
    """
    try:
        from src.mcp.gpu_queue import get_gpu_queue  # 延迟导入避免环依赖
        gq = get_gpu_queue()
        if gq.is_registered(model_name, "embed"):
            vec = gq.submit_embed(model_name, query)
            return vector_literal(vec)
    except RuntimeError:
        # GPUQueue 未注册或禁用, 直接调 model (与原行为一致)
        pass
    model = load_model(model_name)
    embedding = encode_with_model(model, [query], model_name=model_name)[0]
    return vector_literal(embedding)


def _register_gpu_queue_runner(model_name: str = DEFAULT_MODEL) -> None:
    """在 GPUQueue 上注册 (model_name, "embed") runner。

    把多个 producer 的单条 query 合并成一次 model.encode(batch),
    拆结果按请求顺序回填。会在以下时机调用:
    - 模块顶部 lazy 注册 (首次 import 时, 仅在 sentence-transformers 可用);
    - 失败时 caller 回退到直接 model.encode, 不再尝试 queue。
    """
    try:
        from src.mcp.gpu_queue import get_gpu_queue
        gq = get_gpu_queue()
        if gq.is_registered(model_name, "embed"):
            return  # 已注册

        def runner(payloads: list[str]) -> list[list[float]]:
            model = load_model(model_name)
            arr = encode_with_model(model, payloads, model_name=model_name)
            return [list(map(float, row)) for row in arr]

        gq.register(model_name, "embed", runner)
    except Exception:
        # 任何注册失败都吞掉: 实际 producer 在 submit 时仍会感知
        # 并回退直接调用 model.encode, 不影响主调用链。
        pass


try:
    _register_gpu_queue_runner()  # 模块顶层一次性注册默认模型
except Exception:
    pass
