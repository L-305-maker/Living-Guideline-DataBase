# ingestion/crawler/common

这里是爬虫阶段的公共工具层，为不同来源 spider 和 processor 提供基础能力。

## 数据流动

```text
spiders / processors
  -> common/http.py
  -> common/pdf_parser.py
  -> common/jsonl.py
  -> origin JSONL / raw PDF
```

## 文件

| 文件 | 作用 |
| --- | --- |
| `http.py` | HTTP 请求、重试、下载辅助 |
| `pdf_parser.py` | PDF 文本提取 |
| `jsonl.py` | ingestion 阶段 JSONL 读写 |
| `records.py` | 来源记录整理 |
| `text.py` | 爬虫阶段文本清理 |
