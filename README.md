# Guideline Evidence RAG

本项目当前任务是：把指南 PDF 转成 Markdown，清洗、切分、建库，为循证医学 Agent 提供可追溯证据。

## 当前结构

```text
src/
  pipeline/       PDF->Markdown、clean、block、chunk、质量处理
  retrieval/      SQLite FTS、BM25、vector、RRF 融合检索
  mcp/            MCP 工具 search/read/retrieve
  storage/        PostgreSQL/pgvector 后端
  models/         数据 schema
  utils/          通用工具
data/evidence/    默认运行数据目录
deploy/           远程部署示例
legacy_recommendation_pipeline/
                  旧 Recommendation/PICO/GRADE 链路归档
```

## 构建证据库

```powershell
python -m src.pipeline.orchestration.run_pipeline --data-dir data/evidence --skip-vector
```

## 启动 MCP

```powershell
$env:RAG_BACKEND="local"
$env:RAG_DATA_DIR="D:\python\Data_splitting_ingestion\data\evidence"
python -m src.mcp
```

## MCP 工具

`search`：文档级检索。

参数：`query` 必填；`source_institution`、`clinical_department`、`time_range`、`publication_date`、`recency_boost`、`topk` 可选。

`read`：读取完整 clean Markdown 文档。

参数：`doc_id` 或 `title` 至少填一个；`max_chars` 可选。

`retrieve`：chunk 级证据召回。

参数：`query` 必填；`source_institution`、`clinical_department`、`time_range`、`publication_date`、`topk` 可选。

## 医学指南 chunk 切分模块

该模块位于 `src/guideline_chunking/`，只负责 Markdown 结构保真切分：heading path、page/source span、section parent、atomic chunk、table row chunk 和 chunk link。它不抽取或编造 PICO、GRADE、effect estimate、recommendation strength。

```powershell
python scripts/parse_markdown.py --markdown-dir data/markdown --metadata-dir data/metadata --output build/parsed_blocks.jsonl
python scripts/build_section_chunks.py --blocks build/parsed_blocks.jsonl --output build/section_chunks.jsonl
python scripts/build_atomic_chunks.py --blocks build/parsed_blocks.jsonl --sections build/section_chunks.jsonl --output build/atomic_chunks.jsonl
python scripts/build_table_chunks.py --blocks build/parsed_blocks.jsonl --sections build/section_chunks.jsonl --output build/table_chunks.jsonl
python scripts/build_chunk_links.py --sections build/section_chunks.jsonl --atomic build/atomic_chunks.jsonl --tables build/table_chunks.jsonl --output build/chunk_links.jsonl
python scripts/build_chunk_index.py --atomic build/atomic_chunks.jsonl --tables build/table_chunks.jsonl --index-dir build/chunk_index
python scripts/retrieve_chunks.py --index-dir build/chunk_index --query "糖尿病肾病 ACEI 证据" --top-k 10
python scripts/evaluate_chunk_retrieval.py --index-dir build/chunk_index --queries eval/chunk_queries.jsonl --top-k 50
```

主要输出：

```text
build/parsed_blocks.jsonl
build/section_chunks.jsonl
build/atomic_chunks.jsonl
build/table_chunks.jsonl
build/chunk_links.jsonl
build/chunk_build_warnings.jsonl
```
