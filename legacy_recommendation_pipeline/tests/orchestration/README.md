# orchestration 测试组

## 覆盖范围

本组测试端到端候选生成流水线：

- `run_pipeline.py` 是否能串起清洗、解析、路由、抽取、版本构建和报告。
- pipeline manifest 是否记录关键产物。
- 包级公开 API 是否仍可导入。

## 什么时候必须运行

修改以下模块时必须运行：

```powershell
python -m tests.test_main --group orchestration
```

- `src/pipeline/orchestration/`
- 任意会改变 pipeline 产物路径或字段契约的模块
- 包级 `__init__.py` 导出

## 常见失败含义

- e2e 失败：优先看最早失败阶段的 stage summary。
- manifest 缺字段：检查 `PipelinePaths` 和 `artifact_counts`。
- 公开 API 失败：检查包导出是否改名。
