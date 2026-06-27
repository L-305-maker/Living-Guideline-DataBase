# llm_review 阶段

`llm_review/` 为人工或 LLM 辅助复核准备队列、prompt、响应解析和自动质检。

## 数据如何流动

```text
recommendation_candidates.jsonl
grade_candidates.jsonl
  -> queue/builder.py
  -> llm_review_queue.jsonl
  -> prompts/builder.py
  -> clients/llm_client.py
  -> parsing/response_parser.py
  -> auto_qc/
  -> reviewed result
```

## 子目录

| 子目录 | 作用 |
| --- | --- |
| `queue/` | 构建和运行复核队列 |
| `prompts/` | 生成 LLM 复核提示 |
| `clients/` | LLM 客户端封装 |
| `parsing/` | 解析和校验 LLM 输出 |
| `auto_qc/` | 对推荐和 GRADE 复核结果做自动质量检查 |

## 维护提醒

LLM 复核输出不能直接进入正式库。它应先被解析、校验、写回候选或 review 结果，再经过发布流程。
