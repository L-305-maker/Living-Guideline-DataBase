# domain/origin

这里定义原始来源层实体，用来描述数据从外部进入项目后的基础形状。

## 数据流动

```text
ingestion 输出
  -> SourceRecord
  -> cleaning
  -> SourceBlock
  -> extraction
```

## 实体

| 文件 | 作用 |
| --- | --- |
| `source_record.py` | 原始或清洗后的来源记录 |
| `source_block.py` | 结构解析后的文本块 |
| `guideline.py` | 指南来源元数据 |
| `paper.py` | 论文或证据来源元数据 |

这些实体偏“来源和位置”，重点是保留可追溯性。
