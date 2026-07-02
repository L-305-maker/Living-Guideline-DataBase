# llm_review/parsing

这里解析和校验 LLM 复核输出。

## 数据流动

```text
raw LLM response
  -> response_parser.py
  -> validators.py
  -> output_records.py
  -> reviewed result
```

解析层要尽量把错误显式化，不要默默把格式错误的输出当成通过。
