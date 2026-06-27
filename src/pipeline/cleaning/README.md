# cleaning 阶段

`pipeline/cleaning/` 是主流水线第一阶段。它把来源格式不稳定、PDF 噪声较多的原始记录整理成可追溯的 cleaned record。

## 数据如何流动

```text
origin JSONL
  -> source_cleaner.py
  -> cleaned.jsonl
  -> quality_gate.py
  -> cleaned.ready.jsonl
  -> cleaned.needs_layout_repair.jsonl
  -> cleaned.parse_failed.jsonl
  -> cleaned.skipped.jsonl
```

## 核心文件

| 文件 | 作用 |
| --- | --- |
| `source_cleaner.py` | 清洗单条来源记录，生成 `raw_content`、`clean_content`、sections、tables、cleaning_log |
| `quality_gate.py` | 根据长度、章节、版式、表格、来源类型等质量信号分流 cleaned records |
| `pdf_noise.py` | 增强 PDF 噪声修复 |
| `offset_mapping.py` | 维护 raw/clean/source span 的 offset 追踪 |
| `source_normalizers.py` | 标准化不同来源的字段和 seed |
| `base_cleaner.py` | 文本基础清理工具 |
| `deduplicator.py` | 去重辅助 |

## 数据变化

```text
raw record
  -> raw_content 保留原文
  -> clean_content 作为后续工作文本
  -> references_text / affiliations_text 从正文拆出
  -> tables / recommendation_boxes / source_span_boundaries 成为辅助线索
  -> cleaning_log 记录清洗动作
```

只有 `cleaned.ready.jsonl` 会进入 `pipeline/parsing/`。
