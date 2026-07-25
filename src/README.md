# src 代码目录

`src` 保存从 PDF 处理到 PostgreSQL 检索和 MCP 服务的主要实现。各层通过 Markdown、JSONL 和 PostgreSQL 表交换数据，避免在模块之间共享隐式状态。

`PDF -> OCR/抽取 -> 清洗与质检 -> 分块/多视图 -> PostgreSQL 入库 -> 向量化/混合检索 -> MCP 服务`

| 目录 | 职责 | 主要产物或接口 |
| --- | --- | --- |
| `pipeline/` | PDF 转 Markdown、OCR、清洗、质检和编排 | Markdown、documents、sections、chunks |
| `guideline_chunking/` | 结构化指南切分、表格块、链接和本地 BM25 检索 | section、atomic、table chunk 与链接 |
| `retrieval/` | 检索共享逻辑、chunk 规范化、RRF 和重排 | 查询解析、召回融合、reranker 输出 |
| `storage/` | PostgreSQL 建表、入库、全文检索和 pgvector 向量 | 数据表、向量表和混合检索结果 |
| `mcp/` | 暴露 search/read/retrieve 的 MCP 服务 | stdio、SSE 或 streamable HTTP 服务 |
| `models/` | 跨阶段共享的数据模型 | Pydantic schema 与包入口 |
| `utils/` | JSONL、front matter、ID、元数据和文本工具 | 公共辅助函数 |

## 数据约定

- `doc_id` 是跨 Markdown、JSONL、PostgreSQL 和 MCP 输出的稳定主键。
- `chunk_id`、`view_id` 等业务 ID 不应随展示文本变化而随意改变。
- 检索和向量化优先使用 `text_for_embedding` 或 `retrieval_text`，原文保留在 `content`。
- `heading_path`、页码、字符范围和来源路径属于证据追踪字段，不能为简化输出而删除。
- PostgreSQL 向量表使用业务 ID 加 `model` 区分版本，查询侧模型名必须与入库模型一致。

## 验证

```powershell
python -B -m compileall -q src
python -B -m pytest tests/test_mcp_call_logger.py tests/test_chunk_normalizer.py tests/test_pg_document_multiview_storage.py -q -p no:cacheprovider
```

涉及大批量数据写入、OCR 或数据库重建时，先查看 dry-run 或审计报告，再执行实际替换。

## 修改边界

1. 修改数据协议时同步更新生产者、入库、检索输出和测试。
2. 修改 ID、分块或向量文本规则后，重建下游 JSONL、PostgreSQL 数据和向量。
3. 调整 MCP 行为时同时检查 `src/mcp/server.py`、`src/mcp/api_pg.py` 和 `src/storage/pg_hybrid_retrieval.py`。
