# extraction/recommendation

`recommendation/` 从推荐队列中识别临床推荐语句，生成 `RecommendationCandidate` 和对应 `ModelTrace`。

## 数据如何流动

```text
routes/*_recommendation_blocks.jsonl
  -> candidate_extractor.py
  -> recommendation_candidates.jsonl
  -> recommendation_traces.jsonl
```

## 数据变化

```text
SourceBlock.text
  -> split_recommendation_statements()
  -> evaluate_recommendation_statement()
  -> RecommendationCandidate
```

`RecommendationCandidate` 会包含：

```text
candidate_id
recommendation_text
direction
strength
certainty
status
source_text
source_section
normalized_payload
raw_payload
```

这里的候选还不是正式推荐。`status=needs_review` 的候选一般需要人工或 LLM 复核。
