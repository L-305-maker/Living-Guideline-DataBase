# ingestion/crawler/spiders

spiders 面向具体医学资料来源，负责抓取列表页、详情页或 PDF 链接。

## 数据流动

```text
source website
  -> source-specific spider
  -> raw item
  -> processors
  -> data/origin
```

## 文件

| 文件 | 作用 |
| --- | --- |
| `base.py` | spider 基类和共享接口 |
| `registry.py` | 来源 spider 注册 |
| `who.py`、`idsa.py`、`pmc.py` 等 | 具体来源适配 |

新增来源时，优先复用 `base.py` 和 `common/` 工具，避免每个 spider 自己实现下载和 JSONL 逻辑。
