# extraction/evidence

`evidence/` 从证据队列中识别研究和证据条目，并尝试关联 PICO 与推荐候选。

## 数据如何流动

```text
routes/*_evidence_blocks.jsonl
pico_questions.jsonl
recommendation_candidates.jsonl
  -> item_extractor.py
  -> evidence_items.jsonl
  -> evidence_traces.jsonl
```

## 数据变化

```text
Evidence block
  -> infer_study_design()
  -> infer_sample_size()
  -> infer_effect_size()
  -> infer_confidence_interval()
  -> infer_effect_direction()
  -> PicoIndex.match()
  -> RecommendationIndex.match()
  -> EvidenceItem
```

## 输出状态

```text
pending             有较可信关联，后续可复核发布
association_review  有证据信号但关联不完整
uncertain           证据信号弱或结构不足
```

正式发布包只会接收 publish-ready 范围内的证据。
