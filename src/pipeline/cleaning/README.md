# cleaning 阶段

`src/pipeline/cleaning/` 现在只服务于“指南证据库”任务：把原始 PDF 转为 Markdown，清洗 Markdown，再按完整 block 和小 chunk 两层结构建库。

本阶段不再抽取 `Recommendation`、`PICOQuestion`、`GradeCandidate` 等结构化推荐对象。block 被视为可追溯的原文证据单元，chunk 是面向检索和向量化的更小文本单元。

## 数据流

```text
raw_pdf/
  -> pdf_to_markdown.py
     -> inspect PDF text layer / image pages
     -> optional OCRmyPDF for scanned or low-text PDFs
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
| `pdf_to_markdown.py` | 调用 `src.pipeline.cleaning.pdf_to_md`，把 PDF 批量转换为 Markdown。 |
| `markdown_cleaner.py` | 调用 `src.pipeline.cleaning.cleaner`，清理页眉页脚、目录、参考文献等噪声。 |
| `block_encoder.py` | 调用 `src.pipeline.cleaning.encoder`，把清洗后的 Markdown 编码为完整 block。 |
| `block_chunker.py` | 调用 `src.pipeline.cleaning.chunker`，把单个 block 切分成检索 chunk。 |
| `evidence_pipeline.py` | 串联 PDF->Markdown->clean->block->chunk->SQLite/FTS/vector 的证据库构建流程。 |

## 扫描版 PDF

`pdf_to_md.py` 会先检查每个 PDF 的文本层和图片页比例，并在 front matter / manifest 中写入：

- `source_pdf_text_quality`, `source_pdf_needs_ocr`, `source_pdf_is_scanned`: 原始 PDF 的质量和扫描件判断
- `pdf_text_quality`: `ok` / `warning` / `poor`
- `pdf_needs_ocr`: 是否需要 OCR
- `pdf_is_scanned`: 是否像扫描件
- `ocr_engine`, `ocr_applied`, `ocr_status`, `ocr_error`: OCR 尝试结果

默认 `--ocr-mode auto`。当检测到扫描件或文本层过少时，如果本机安装了 OCRmyPDF，会自动生成 OCR 后 PDF 再转 Markdown；如果未安装，则不中断流水线，只标记 `ocr_status=needed_unavailable` 和 `ocr_error`，方便后续重跑。

注意：`pdf_*` 表示实际用于抽取 Markdown 的 PDF。如果 OCR 成功，它会反映 OCR 输出 PDF 的质量；`source_pdf_*` 始终保留原始 PDF 的状态。

检索阶段不会硬过滤低质量文档，而是在最终 rerank 时降权：

- `cleaning_quality=poor`
- `pdf_text_quality=poor`
- `pdf_needs_ocr=true`
- `ocr_status=needed_unavailable/failed/needed_but_disabled`
- `cleaning_flags` 中出现 `likely_ocr_failure`、`low_text_signal`、`noisy_ocr_lines`

降权明细会写入搜索结果的 `match_reason.quality_penalties` 和 `match_reason.quality_multiplier`，便于排查为什么某篇扫描件文档排名靠后。

```powershell
python -m src.pipeline.cleaning.pdf_to_md --input-dir data/evidence/raw_pdf --output-dir data/evidence/markdown_raw --ocr-mode auto --ocr-languages chi_sim+eng
```

强制重跑 OCR：

```powershell
python -m src.pipeline.cleaning.pdf_to_md --input-dir data/evidence/raw_pdf --output-dir data/evidence/markdown_raw --ocr-mode force
```

## 设计边界

- cleaning 只生成证据检索需要的 Markdown、block、chunk 和索引。
- 推荐语句、PICO、GRADE、人工审核、发布版本等旧链路已经迁移到 `legacy_recommendation_pipeline/`。
- 默认向量化粒度是 chunk，不是整篇 document；document/block 主要承担溯源和展示。
