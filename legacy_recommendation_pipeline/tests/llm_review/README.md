# tests/llm_review

这里测试 LLM 复核相关逻辑，重点看 prompt 构建、响应解析和复核输出约束。

## 测试数据流

```text
候选样例
  -> prompt builder
  -> 模拟/解析 LLM 输出
  -> 校验结构和字段
```

这些测试帮助确认 LLM 辅助复核不会输出无法被后续 review/publish 消费的格式。
