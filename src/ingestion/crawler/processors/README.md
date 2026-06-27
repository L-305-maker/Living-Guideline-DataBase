# ingestion/crawler/processors

processors 把 spider 抓到的原始内容整理成项目可追溯输入。

## 数据流动

```text
spider raw results / raw PDF
  -> processors
  -> origin JSONL
  -> provenance / raw_pdf manifest
  -> pipeline/cleaning
```

## 文件

| 文件 | 作用 |
| --- | --- |
| `origin.py` | 生成或整理 origin JSONL |
| `raw_pdf_incremental.py` | 增量下载和登记 PDF |
| `backfill_provenance.py` | 补充来源和下载 provenance |

processor 的输出应尽量保留 URL、标题、来源、PDF 路径和原始文本字段。
