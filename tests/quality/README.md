# quality 测试组

## 覆盖范围

本组测试质量报告和 goldset 评估：

- 数据产物 manifest 是否能正确统计文件。
- goldset evaluation 是否能计算 precision、recall、F1、字段准确率和 offset 准确率。
- goldset 阈值门禁是否能明确报告失败指标。

## 什么时候必须运行

修改以下模块时必须运行：

```powershell
python -m tests.test_main --group quality
```

- `src/pipeline/quality/`
- `scripts/evaluate_goldset.py`
- `scripts/data_artifact_manifest.py`
- `tests/fixtures/goldset/`

## 常见失败含义

- goldset 指标变化：检查预测实体 ID、字段名和 raw offset。
- manifest 测试失败：检查 data artifact 分类规则。
