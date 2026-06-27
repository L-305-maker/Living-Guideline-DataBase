# orchestration 阶段

`orchestration/` 是端到端候选生成流水线的调度层。它负责串联各阶段，但不把候选发布成正式推荐。

## 数据如何流动

```text
origin JSONL
  -> run_pipeline.py
  -> cleaned.jsonl
  -> cleaned.ready.jsonl
  -> blocks.jsonl
  -> routed queues
  -> candidates / picos / evidence
  -> recommendation_versions.jsonl
  -> update_logs.jsonl
  -> llm_review_queue.jsonl
  -> quality_reports/
  -> run_manifest.json
```

## 核心文件

| 文件 | 作用 |
| --- | --- |
| `run_pipeline.py` | 候选生成流水线主入口 |

## 设计边界

`run_pipeline.py` 只生成候选和报告，不做正式发布，不直接写 PostgreSQL。

正式发布应走：

```text
pipeline/publish/
  -> storage/
```
