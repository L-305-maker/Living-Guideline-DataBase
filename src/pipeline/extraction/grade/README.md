# extraction/grade

`grade/` 从 GRADE 队列中抽取证据确定性、推荐强度、降级原因，并尝试关联到推荐候选。

## 数据如何流动

```text
routes/*_grade_blocks.jsonl
recommendation_candidates.jsonl
  -> candidate_extractor.py
  -> grade_candidates.jsonl
  -> grade_traces.jsonl
```

## 数据变化

```text
GRADE block
  -> infer_grade_system()
  -> infer_certainty()
  -> infer_strength()
  -> downgrade_reasons()
  -> RecommendationIndex.find()
  -> GradeCandidate
```

## 关联逻辑

GRADE 候选需要尽量挂到某条 `RecommendationCandidate`：

```text
same_block
  -> nearest_order
  -> section_overlap
  -> not_found
```

关联质量会进入 `association_quality`，后续版本构建会参考它。
