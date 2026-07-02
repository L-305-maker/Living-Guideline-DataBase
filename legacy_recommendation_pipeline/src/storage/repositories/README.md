# storage/repositories

`repositories/` 封装特定表或业务对象的 PostgreSQL 读写逻辑，避免所有 SQL 都堆在 CLI 或通用入库器里。

## 数据如何流动

```text
storage_PG.py / generic_ingest.py
  -> repositories/*
  -> PostgreSQL
```

## 文件职责

| 文件 | 作用 |
| --- | --- |
| `cleaned_records.py` | cleaned_records、guidelines、papers 的入库辅助 |
| `recommendation_version_reviews.py` | recommendation_version_review_queue 相关读写 |

如果某个表有复杂业务读写逻辑，可以放到这里；纯批量 JSONL 入库优先走 `generic_ingest.py`。
