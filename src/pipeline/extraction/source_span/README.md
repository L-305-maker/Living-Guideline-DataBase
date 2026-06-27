# extraction/source_span

这里是 source span 优先的实验抽取路径，目标是强化结构化结果与原文片段的对应关系。

## 数据流动

```text
SourceBlock / clean_content offsets
  -> first_pass.py
  -> source-span-first candidates
  -> quality / review
```

这个目录里的逻辑更偏实验和评估。稳定主流程仍以 routing 后的四类抽取器为主。
