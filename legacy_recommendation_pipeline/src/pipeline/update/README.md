# update 阶段

`update/` 负责比较新旧推荐版本，生成 `UpdateLog`。

## 数据如何流动

```text
previous recommendation_versions
current recommendation_versions
  -> version_diff.py
  -> update_logs.jsonl
  -> update_logs_report.jsonl
```

## 数据变化

```text
old version + new version
  -> 判断新增 / 修改 / 无变化
  -> 生成 update_type
  -> 记录 old_version_id / new_version_id
```

`UpdateLog` 是正式发布和后续 Living-Guideline 更新解释的重要依据。
