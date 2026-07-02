# domain/knowledge

这里定义接近正式知识库的实体。它们通常由多个候选组合而成，并会进入发布和入库流程。

## 数据流动

```text
RecommendationCandidate + GradeCandidate + PicoQuestion + EvidenceItem
  -> RecommendationVersion
  -> publish
  -> Recommendation 当前态
```

## 实体

| 文件 | 作用 |
| --- | --- |
| `pico_question.py` | PICO 临床问题结构 |
| `evidence_item.py` | 证据条目 |
| `recommendation_version.py` | 不可变推荐版本 |
| `recommendation.py` | 当前正式推荐状态 |

`RecommendationVersion` 是版本账本，`Recommendation` 是当前态，两者不要混淆。
