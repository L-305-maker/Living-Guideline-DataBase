# cleaning 测试组

## 覆盖范围

本组测试清洗阶段和记录质量上下文：

- 清洗质量门如何把记录分到 ready、layout repair、parse failed、skipped。
- 指南文本清洗是否保留正文、表格、references 和 affiliations。
- clean span 到 raw span 的 offset 映射是否稳定。
- record-level quality context 是否能传递到后续 block。
- record type routing 是否能区分 guideline、paper、mixed、unknown。

## 什么时候必须运行

修改以下模块时必须运行：

```powershell
python -m tests.test_main --group cleaning
```

- `src/pipeline/cleaning/`
- `src/common/record_quality.py`
- 清洗字段契约，例如 `raw_content`、`content`、`cleaning_quality_flags`

## 常见失败含义

- ready/skipped 数量变化：检查质量门阈值和 hard flag。
- offset 测试失败：检查清洗是否删除或合并了原文字符。
- record type 失败：检查 guideline/paper 判定关键词和 source metadata。
