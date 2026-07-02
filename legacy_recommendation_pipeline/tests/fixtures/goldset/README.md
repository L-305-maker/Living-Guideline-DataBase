# tests/fixtures/goldset

这里存放 goldset 评估测试使用的小型人工标注样本。

## 数据流动

```text
gold labels
predictions
  -> quality/goldset.py
  -> precision / recall / match report
```

goldset fixture 的重点是稳定评估逻辑，不追求覆盖完整真实数据规模。
