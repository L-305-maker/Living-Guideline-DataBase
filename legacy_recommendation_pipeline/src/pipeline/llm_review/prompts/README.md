# llm_review/prompts

这里负责把候选和上下文组织成 LLM 可读的复核 prompt。

## 数据流动

```text
review queue item
  -> prompt builder
  -> LLM request text/messages
  -> clients
```

Prompt 应清楚区分来源文本、候选字段、需要模型判断的问题和期望输出格式。
