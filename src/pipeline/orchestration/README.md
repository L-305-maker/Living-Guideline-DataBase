# orchestration 阶段

`src/pipeline/orchestration/` 是当前证据库构建的一键入口，负责调用 cleaning 阶段并生成可供 MCP 检索使用的数据目录。

## 数据流

```text
run_pipeline.py
  -> PDF to Markdown
  -> clean Markdown
  -> encode blocks
  -> split chunks
  -> build SQLite FTS
  -> optionally build BM25 JSON/vector index
  -> manifest.json
```

## 入口

```powershell
python -m src.pipeline.orchestration.run_pipeline --data-dir project/data --skip-vector
```

常用参数：

| 参数 | 作用 |
| --- | --- |
| `--data-dir` | 数据根目录，默认 `project/data`。 |
| `--raw-pdf-dir` | 原始 PDF 目录；不传则使用 `data_dir/raw_pdfs`。 |
| `--skip-pdf-to-markdown` | 已经有 Markdown 时跳过 PDF 转换。 |
| `--skip-vector` | 跳过向量索引构建，只保留 SQLite/FTS。 |
| `--legacy-json-bm25` | 额外生成旧 JSON BM25 索引。 |
| `--embedding-model` | 构建向量索引时使用的 embedding 模型。 |

## 设计边界

`run_pipeline.py` 不再调度 Recommendation/PICO/GRADE 抽取，也不做人工审核或推荐发布。它只产出面向循证医学 Agent 的证据检索库。
