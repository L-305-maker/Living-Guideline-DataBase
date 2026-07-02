# versioning 测试组

## 覆盖范围

本组测试推荐版本与当前态：

- `RecommendationVersion` 是否正确生成版本号和 previous version 链。
- publish gate 是否能拦截缺少 source span、GRADE 冲突、弱上下文版本。
- `Recommendation` 当前态是否只由 publishable version 生成。
- `UpdateLog` 是否能识别新增、修改、撤回、重申等变化。

## 什么时候必须运行

修改以下模块时必须运行：

```powershell
python -m tests.test_main --group versioning
```

- `src/pipeline/extraction/versioning/`
- `src/pipeline/update/`
- `src/domain/knowledge/recommendation.py`
- `src/domain/knowledge/recommendation_version.py`

## 常见失败含义

- 版本号失败：检查 `recommendation_id` 是否稳定。
- publish gate 失败：检查阻断原因是否被误删。
- update log 失败：检查比较字段和 evidence id 读取逻辑。
