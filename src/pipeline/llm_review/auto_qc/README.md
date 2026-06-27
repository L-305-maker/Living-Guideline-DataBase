# llm_review/auto_qc

这里对 LLM 或人工复核结果做自动质量检查。

## 数据流动

```text
reviewed recommendation / grade result
  -> auto_qc
  -> pass / warning / fail
  -> review or publish decision input
```

auto QC 是保护层，不是正式发布层。最终是否进入发布仍要看 review 和 publish gate。
