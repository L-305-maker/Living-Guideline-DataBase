# src 源码维护指南

## 目录职责

`src` 包含从 PDF 证据处理到检索服务的全部运行时代码。推荐按以下数据流理解：

`PDF → OCR/清洗 → 文档与分块产物 → SQLite 或 PostgreSQL 入库 → 词法/向量混合召回 → 重排 → MCP 服务`

| 目录 | 职责 | 主要产物或接口 |
| --- | --- | --- |
| `pipeline/` | PDF 转换、OCR、清洗、质量审计和流水线编排 | Markdown、documents.jsonl、sections、chunks |
| `guideline_chunking/` | 保留标题层级、表格和块关系的结构化切分 | section、atomic、table chunk 及链接 |
| `retrieval/` | SQLite/FAISS 本地检索、RRF 融合和重排 | 文档候选、证据块和上下文 |
| `storage/` | PostgreSQL、全文检索、pgvector 建库与混合检索 | 数据表、向量表和 SQL 检索函数 |
| `mcp/` | 将搜索、读取和检索能力暴露为 MCP 工具 | stdio、SSE 或 streamable HTTP 服务 |
| `models/` | 跨模块共享的数据模型 | 文档、章节和分块结构 |
| `utils/` | 路径、JSONL、元数据、ID 和科室分类工具 | 无独立服务 |

## 关键数据约束

- `doc_id` 是文档、卡片、视图、章节和分块之间的主关联键。
- `chunk_id`、`view_id` 与向量映射必须稳定；变更 ID 规则后必须重建对应索引。
- 检索文本优先使用 `text_for_embedding` 或 `retrieval_text`，展示内容仍使用原始 `content`。
- `heading_path`、页码、字符范围和来源路径属于可追溯字段，不应在清洗或入库时丢弃。
- PostgreSQL 向量记录的 `model` 必须与查询阶段的 `PG_VECTOR_MODEL` 一致。

## 常用验证

```powershell
python -B -m compileall -q src
python -B -m pytest tests -q -p no:cacheprovider
```

涉及数据库、OCR 或模型的测试可能依赖外部服务和本地模型缓存，应先确认环境变量与依赖已准备好。

## 修改顺序

1. 修改数据模型时，检查生产者、JSONL 产物、数据库表和检索消费者。
2. 修改切分或 ID 规则时，重新生成派生产物并重建词法及向量索引。
3. 修改混合检索权重时，保留各通道名次和 `match_reason`，便于回归分析。
4. 修改服务接口时，同时检查 SQLite 与 PostgreSQL 两种后端。