# extraction 阶段

`pipeline/extraction/` 是候选抽取核心层。它从 `SourceBlock` 中抽取推荐、GRADE、PICO、证据，并最终构建 `RecommendationVersion`。

## 数据如何流动

```text
blocks.jsonl
  -> routing/candidate_router.py
  -> routes/*_recommendation_blocks.jsonl
  -> routes/*_grade_blocks.jsonl
  -> routes/*_pico_blocks.jsonl
  -> routes/*_evidence_blocks.jsonl
  -> recommendation_candidates.jsonl
  -> grade_candidates.jsonl
  -> pico_questions.jsonl
  -> evidence_items.jsonl
  -> versioning/recommendation_version_builder.py
  -> recommendation_versions.jsonl
```

## 子模块

| 子目录 | 作用 |
| --- | --- |
| `routing/` | 把 SourceBlock 分配到抽取队列 |
| `recommendation/` | 抽取推荐语句候选 |
| `grade/` | 抽取 GRADE、证据确定性和推荐强度 |
| `pico/` | 抽取 PICO 临床问题结构 |
| `evidence/` | 抽取证据条目和研究信息 |
| `versioning/` | 把候选组装成 RecommendationVersion |
| `enhancement/` | 对候选做增强和归一化 |
| `source_span/` | source span 优先的实验抽取路径 |
| `common/` | 抽取共享枚举、映射和校验工具 |

## 重要边界

抽取结果仍然是候选。`recommendation_candidates` 不等于正式推荐，`recommendation_versions` 也需要经过 publish gate 和发布流程。
