# retrieval 检索辅助模块

该目录提供检索链路中的无状态辅助逻辑：查询解析、chunk 规范化、RRF 融合、规则重排和 BGE reranker 封装。真正的 PostgreSQL 查询入口在 `src/storage/pg_hybrid_retrieval.py`，MCP 对外入口在 `src/mcp/api_pg.py`。

## 文件

| 文件 | 作用 |
| --- | --- |
| `common.py` | 时间范围解析、query term 提取、chunk 元数据补齐和来源上下文拼接 |
| `chunk_normalizer.py` | 将旧版和新版 chunk JSONL 收敛为 PostgreSQL 入库契约 |
| `rrf.py` | Reciprocal Rank Fusion 和加权 RRF |
| `reranker.py` | 文档级和 chunk 级重排接口、规则 fallback 和 Qwen3-Reranker cross-encoder |
| `document_repr/` | 文档 card 与多视图表示构建 |

## 检索约定

- 召回通道可以分别产出候选，最终用 RRF 或加权 RRF 合并。
- `match_reason` 记录命中字段、boost、base score、reranker 和质量惩罚，便于审计。
- chunk 返回应保留 `doc_id`、`chunk_id`、标题路径和字符范围。
- 查询侧模型名必须与 pgvector 表中的 `model` 一致，避免不同模型向量混用。

## Reranker 配置

默认 reranker 使用 `Qwen/Qwen3-Reranker-4B`，本地缓存优先。可用环境变量切换：

- `DOCUMENT_RERANKER`：`bge_m3`、`rule` 或 `none`。
- `CHUNK_RERANKER` 或 `RERANKER`：`bge_m3`、`rule` 或 `none`。
- `BGE_RERANKER_MODEL`、`BGE_RERANKER_DEVICE`、`BGE_RERANKER_BATCH_SIZE`、`BGE_RERANKER_MAX_LENGTH`。
- `BGE_RERANKER_LOCAL_ONLY=1` 时只读本地 Hugging Face 缓存。
- `BGE_RERANKER_DTYPE`、`BGE_RERANKER_ATTN`：传给 CrossEncoder 的 `torch_dtype`/`attn_implementation`（大模型建议 `BGE_RERANKER_DTYPE=bfloat16`）。

## 验证

```powershell
python -B -m compileall -q src/retrieval src/storage src/mcp
python -B -m pytest tests/test_chunk_normalizer.py tests/test_pg_document_multiview_storage.py -q -p no:cacheprovider
```

修改 chunk 契约或 reranker 输出后，同步检查 PostgreSQL 入库、混合检索和 MCP 返回字段。
