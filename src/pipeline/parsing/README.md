# parsing 阶段

`pipeline/parsing/` 负责把 cleaned record 切成可追溯的 `SourceBlock`。它不做正式抽取，只恢复文档结构。

## 数据如何流动

```text
cleaned.ready.jsonl
  -> structure_parser.py
  -> blocks.jsonl
  -> pipeline/extraction/routing/
```

## 核心文件

| 文件 | 作用 |
| --- | --- |
| `structure_parser.py` | 按标题、段落、列表、表格切分 SourceBlock，并保留 section_path、order、char_start/char_end |
| `guideline_profiler.py` | 识别指南 profile 和来源上下文，为抽取提供辅助线索 |

## SourceBlock 的意义

`SourceBlock` 是抽取前的最小工作单元。它通常包含：

```text
block_id
record_id
guideline_id / paper_id
section_path
block_type
text
order
char_start / char_end
candidate_hints
quality
```

后续 router 根据 `candidate_hints` 和 `quality` 决定这个 block 进入哪条抽取支线。
