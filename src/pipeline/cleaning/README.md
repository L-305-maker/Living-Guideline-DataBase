# cleaning 数据清洗流水线

该目录负责把原始 PDF 或 raw Markdown 转换成可追踪、可检索的中间产物。清洗层只生产文档、章节、分块、卡片和视图，不在这里执行 PostgreSQL 向量化。

## 文件

| 文件 | 作用 |
| --- | --- |
| `pdf_to_md.py` | PDF 转 raw Markdown，生成 front matter、稳定 `doc_id`、文本质量报告和可选 OCR manifest |
| `pdf_to_markdown.py` | `pdf_to_md.py` 的流水线适配入口 |
| `cleaner.py`、`markdown_cleaner.py` | 清洗 Markdown、保留 front matter、输出文档元数据 |
| `encoder.py`、`block_encoder.py` | 将 clean Markdown 编码为完整源章节块 |
| `chunker.py`、`semantic_chunker.py` | 生成可检索的小 chunk 和检索文本 |
| `evidence_pipeline.py` | 串联 PDF 转换、清洗、章节块、chunk 和文档多视图构建 |
| `add_clinical_departments.py` | 根据标题、摘要和正文信号补齐临床科室 |
| `compact_retrieval_artifacts.py` | 将 cards、views、chunks 重写为最小检索契约 |
| `reclassify_unknown_departments.py` | 复核并处理未分类科室 |
| `relabel_chunk_departments.py` | 根据文档和章节信号重标 chunk 科室 |

## 主要产物

默认输入为 `data/evidence/raw_pdf/**/*.pdf`，主要输出包括：

- `data/evidence/markdown_raw/*.md`：带 front matter 的原始 Markdown。
- `data/evidence/markdown_clean/*.md`：清洗后的 Markdown 正文。
- `data/evidence/documents.jsonl`：文档级元数据。
- `data/evidence/sections/*.jsonl`：完整源章节块。
- `data/evidence/chunks/*.jsonl` 和 `data/evidence/chunks/all_chunks.jsonl`：检索 chunk。
- `data/evidence/document_cards.jsonl` 和 `data/evidence/document_views.jsonl`：文档级召回表示。

## 维护原则

- 批量写入前先保留 manifest 或 dry-run 报告，失败样本进入报告而不是中断整批。
- OCR 凭据只从环境变量读取，不写入源码、README 或日志。
- 修改正文后更新内容哈希；修改 `doc_id`、分块规则或检索文本后重建下游产物。
- `markdown_clean` 是后续回填和入库的主要文本来源，不从多个目录猜测最新版本。
- `cleaning_quality` 和 `cleaning_flags` 用于检索质量惩罚，应随修复结果同步更新。

## 验证

```powershell
python -B -m pytest tests/test_markdown_cleaner.py tests/test_pdf_ocr_ingestion.py -q -p no:cacheprovider
python -B -m compileall -q src/pipeline/cleaning
```

入库和向量化由 `src/storage` 负责，cleaning 层完成后再执行对应 PostgreSQL 命令。
