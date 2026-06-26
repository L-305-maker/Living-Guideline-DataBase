# review_publish 测试组

## 覆盖范围

本组测试人工审核和正式发布：

- 通用 manual review queue 的 build/apply 往返。
- recommendation version review queue 的 approve/reject/published 状态转换。
- `recommendation_publisher` 是否能生成发布 bundle。
- 发布 bundle 是否包含 version、current recommendation、update log、review queue 状态。

## 什么时候必须运行

修改以下模块时必须运行：

```powershell
python -m tests.test_main --group review_publish
```

- `src/pipeline/review/`
- `src/pipeline/publish/`
- `src/storage/repositories/recommendation_version_reviews.py`

## 常见失败含义

- approved item 不能发布：检查 `review_status` 和 `version_payload`。
- update log 缺失：检查 previous_versions 输入和 change_type。
- review 状态失败：检查 published 状态字段是否同步。
