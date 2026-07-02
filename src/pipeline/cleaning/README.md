# cleaning 阶段

`src/pipeline/cleaning/` 现在只服务于“指南证据库”任务：把原始 PDF 转为 Markdown，清洗 Markdown，再按完整 block 和小 chunk 两层结构建库。

本阶段不再抽取 `Recommendation`、`PICOQuestion`、`GradeCandidate` 等结构化推荐对象。block 被视为可追溯的原文证据单元，chunk 是面向检索和向量化的更小文本单元。

## 数据流

```text
raw_pdfs/
  -> pdf_to_markdown.py
  -> markdown_raw/
  -> markdown_cleaner.py
  -> markdown_clean/
  -> block_encoder.py
  -> sections/
  -> block_chunker.py
  -> chunks.jsonl
```

## 核心文件

| 文件 | 作用 |
| --- | --- |
| `pdf_to_markdown.py` | 调用 `project.pipeline.pdf_to_md`，把 PDF 批量转换为 Markdown。 |
| `markdown_cleaner.py` | 调用 `project.pipeline.cleaner`，清理页眉页脚、目录、参考文献等噪声。 |
| `block_encoder.py` | 调用 `project.pipeline.encoder`，把清洗后的 Markdown 编码为完整 block。 |
| `block_chunker.py` | 调用 `project.pipeline.chunker`，把单个 block 切分成检索 chunk。 |
| `evidence_pipeline.py` | 串联 PDF->Markdown->clean->block->chunk->SQLite/FTS/vector 的证据库构建流程。 |

## 设计边界

- cleaning 只生成证据检索需要的 Markdown、block、chunk 和索引。
- 推荐语句、PICO、GRADE、人工审核、发布版本等旧链路已经迁移到 `legacy_recommendation_pipeline/`。
- 默认向量化粒度是 chunk，不是整篇 document；document/block 主要承担溯源和展示。
