# llm_review/queue

这里负责构建和运行复核队列。

## 数据流动

```text
recommendation_candidates + grade_candidates
  -> builder.py
  -> llm_review_queue.jsonl
  -> runner.py
  -> LLM response / review output
```

队列项应该包含足够上下文，让人工或模型能判断候选是否可接受。
