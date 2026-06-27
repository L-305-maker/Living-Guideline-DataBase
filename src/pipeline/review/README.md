# review 阶段

`review/` 处理人工或规则辅助复核，把弱关联、冲突、缺失和 backlog 数据变成可继续发布的候选。

## 数据如何流动

```text
candidate / version / evidence backlog
  -> review_batch.py
  -> manual_review_gate.py
  -> rule_assisted_backfill.py
  -> association_review.py
  -> reviewed JSONL
  -> publish/
```

## 核心文件

| 文件 | 作用 |
| --- | --- |
| `manual_review_gate.py` | 通用人工复核 gate 和回写 |
| `review_batch.py` | 构建和处理复核批次 |
| `rule_assisted_backfill.py` | 用规则补全缺失关联和字段 |
| `association_review.py` | 处理 recommendation、PICO、evidence 的关联复核 |

## 维护重点

复核阶段的目标不是“让所有候选都发布”，而是把不确定原因显式化：哪些能修、哪些要留在 backlog。
