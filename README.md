# Data Splitting Ingestion

本项目把医学指南 PDF 和清洗后的 Markdown 处理为可检索、可审计的证据库，并通过 PostgreSQL、pgvector 和 MCP 提供文档搜索、全文读取和分块检索能力。

## 主流程

`PDF -> Markdown -> 文本清洗 -> 结构化切分 -> JSONL 产物 -> PostgreSQL 入库 -> BGE-M3 向量化 -> 混合检索 -> MCP 服务`

默认数据目录为 `data/evidence`。大批量处理时先保留中间产物和审计报告，再执行入库或替换主数据。

## 目录

- `src/pipeline/`：PDF 转换、OCR 调度、清洗、质检和流水线编排。
- `src/guideline_chunking/`：结构优先的 Markdown 指南切分、表格处理、链接和 BM25 检索。
- `src/retrieval/`：检索通用逻辑、chunk 规范化、RRF 融合和 reranker。
- `src/storage/`：PostgreSQL 表结构、全文检索、pgvector 向量化和混合召回。
- `src/mcp/`：PostgreSQL-only 的 MCP search/read/retrieve 服务入口。
- `scripts/`：当前数据生产、OCR 修复、远程 PostgreSQL 建库和运维脚本。
- `deploy/`：远程服务部署样例和环境变量模板。

## 常用命令

```powershell
python -m pytest -q
python -B -m compileall -q src scripts
python -B -m src.mcp
```

数据库连接、OCR 凭据和模型路径只通过环境变量或服务器配置文件提供，不写入仓库。

## PostgreSQL 与向量

检索主链路以 PostgreSQL 和 pgvector 为准。修改文档正文、分块规则、`text_for_embedding` 或模型名后，应重新入库并补齐对应模型的向量；仅重建索引不会生成缺失向量。
