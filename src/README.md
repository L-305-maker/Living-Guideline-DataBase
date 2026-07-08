# Evidence RAG Source Layout

当前源码已经统一收敛到 `src/`。`project/` 里的实现已按职责迁入这里，运行数据放在 `data/evidence/`。

## 目录结构

```text
src/
  pipeline/
    cleaning/        PDF->Markdown、清洗、block、chunk
    ocr/             OCR 辅助
    quality/         质量审计和修复
    orchestration/   一键构建入口
  retrieval/         SQLite FTS、BM25、vector、RRF、hybrid retrieval
  mcp/               search/read/retrieve MCP API 和 server
  storage/           PostgreSQL/pgvector 后端
  models/            JSON-facing schemas
  utils/             front matter、ID、IO、metadata 等通用工具
  common/            兼容入口
```

## 运行

```powershell
python -m src.pipeline.orchestration.run_pipeline --data-dir data/evidence --skip-vector
python -m src.mcp
```

默认数据目录来自 `src.utils.io.DATA_DIR`：

```text
data/evidence/
  raw_pdf/
  markdown_raw/
  markdown_clean/
  sections/
  chunks/
  index/
```

## 检索入口

MCP 只暴露三个工具：

- `search`：指南/文档级检索。
- `read`：读取完整 clean Markdown 证据文档。
- `retrieve`：chunk 级证据召回，使用 BM25/vector RRF 融合。
