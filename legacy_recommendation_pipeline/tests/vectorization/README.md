# tests/vectorization

这里测试向量化队列逻辑，确保正式知识对象能被转换成后续 embedding/RAG 使用的任务。

## 测试数据流

```text
RecommendationVersion / EvidenceItem / PicoQuestion 样例
  -> embedding_queue
  -> queue item
  -> 断言对象类型、ID、文本和元数据
```

向量化测试不验证真实 embedding 服务，只验证入队数据结构是否稳定。
