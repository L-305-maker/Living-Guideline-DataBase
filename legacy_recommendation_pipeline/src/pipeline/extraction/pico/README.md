# extraction/pico

`pico/` 从 PICO 队列中抽取临床问题结构：Population、Intervention、Comparator、Outcomes。

## 数据如何流动

```text
routes/*_pico_blocks.jsonl
  -> question_extractor.py
  -> pico_questions.jsonl
  -> pico_traces.jsonl
```

## 数据变化

```text
SourceBlock.text
  -> split_pico_units()
  -> infer_population()
  -> infer_intervention()
  -> infer_comparator()
  -> infer_outcomes()
  -> PicoQuestion row
```

## 输出状态

```text
confidence >= 0.8 -> active
confidence < 0.8  -> under_review
```

版本构建阶段会尝试把推荐候选匹配到最合适的 PICO。
