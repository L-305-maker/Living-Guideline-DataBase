# extraction/routing

`routing/` 根据 `SourceBlock.candidate_hints`、章节质量和记录类型，把文本块分发到具体抽取支线。

## 数据如何流动

```text
blocks.jsonl
  -> candidate_router.py
  -> recommendation queue
  -> grade queue
  -> pico queue
  -> evidence queue
  -> background / repair / skipped queue
```

## 关键输入

```text
candidate_hints
quality.skip_candidate_extraction
section_path
record_type
block_type
text
```

## 输出变化

每个 block 会被复制并附加 `route` 字段：

```text
route.task_types
route.primary_task
route.queues
route.priority
route.route_action
route.reason
```

后续抽取器只读取自己队列里的 block，避免每个抽取器重复判断所有文本。
