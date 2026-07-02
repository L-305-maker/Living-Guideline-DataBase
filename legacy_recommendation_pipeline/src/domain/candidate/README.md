# domain/candidate

这里定义抽取候选层实体。候选数据来自规则或模型抽取，但还没有成为正式知识。

## 数据流动

```text
SourceBlock
  -> RecommendationCandidate / GradeCandidate / ModelTrace
  -> review
  -> versioning
```

## 实体

| 文件 | 作用 |
| --- | --- |
| `recommendation_candidate.py` | 推荐语句候选 |
| `grade_candidate.py` | GRADE 和推荐强度候选 |
| `model_trace.py` | 记录抽取过程、输入、输出和置信度 |

候选层的核心原则：保留来源和 trace，允许 `needs_review`，不要直接当正式知识使用。
