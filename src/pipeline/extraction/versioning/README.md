# extraction/versioning

`versioning/` 把推荐、GRADE、PICO 和证据候选组装成不可变的 `RecommendationVersion`。

## 数据如何流动

```text
recommendation_candidates.jsonl
grade_candidates.jsonl
pico_questions.jsonl
evidence_items.jsonl
  -> recommendation_version_builder.py
  -> recommendation_versions.jsonl
  -> recommendation_versions_report.jsonl
```

## 数据变化

```text
accepted RecommendationCandidate
  -> 匹配 GradeCandidate
  -> 匹配 PicoQuestion
  -> 匹配 EvidenceItem
  -> evaluate_publish_gate()
  -> RecommendationVersion
```

## publish_gate

`publish_gate` 会把版本分为：

```text
publishable   可以进入正式发布候选
needs_review  缺少软条件，需要复核
blocked       有硬性问题，不能发布
```

注意：`RecommendationVersion` 是版本账本，不等于 `recommendations` 当前态。当前态由发布流程生成。
