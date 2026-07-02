# ingestion 模块

`ingestion/` 负责把外部医学指南、论文、PDF 和网页内容收集成项目能处理的原始输入。

## 数据如何流动

```text
外部来源网页 / PDF
  -> crawler/spiders/
  -> crawler/common/
  -> crawler/processors/
  -> data/origin/*.jsonl
  -> data/raw_pdf/
  -> pipeline/cleaning/
```

## 主要部分

| 目录 | 作用 |
| --- | --- |
| `crawler/spiders/` | 面向具体来源的抓取逻辑，如 WHO、PMC、IDSA |
| `crawler/common/` | HTTP、JSONL、PDF、文本公共处理 |
| `crawler/processors/` | 把抓取结果整理成 origin JSONL 或补全 provenance |
| `crawler/crawl.py` | 爬虫命令入口 |

## 输入输出

输入通常是 URL、来源名称、PDF 链接或已有下载文件。

输出通常是：

```text
data/origin/<source>_origin.jsonl
data/raw_pdf/<source>_pdf/...
```

后续 `pipeline/cleaning/source_cleaner.py` 会读取这些 origin JSONL，把它们统一成 cleaned record。
