# tests/fixtures

这里存放小型、稳定、可重复的测试样本。

## 数据如何使用

```text
fixture JSONL
  -> tests/*
  -> 调用 src 模块
  -> 断言输出
```

fixture 应保持小而清楚。大型运行产物应放在 `data/processed/`，不要混进这里。
