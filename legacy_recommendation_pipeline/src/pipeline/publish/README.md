# publish 阶段

`publish/` 把复核后、闭包完整、质量合格的候选版本导出为正式发布包。

## 数据如何流动

```text
reviewed run directory
  -> release_exporter.py
  -> publish_ready_v1/
  -> review_backlog_v1/
  -> storage/storage_PG.py validate-run
  -> storage/storage_PG.py ingest-release
```

## 核心文件

| 文件 | 作用 |
| --- | --- |
| `release_exporter.py` | 根据 formal gate 导出 publish-ready 闭包和 backlog |
| `recommendation_publisher.py` | 把 approved 版本发布为当前推荐态 |

## 发布包必须闭合

`publish_ready_v1` 需要包含：

```text
cleaned_records
guidelines / papers
model_traces
recommendation_candidates
grade_candidates
pico_questions
recommendation_versions
recommendations
evidence_items
recommendation_version_evidence_links
update_logs
```

不满足发布条件的数据进入 `review_backlog_v1`，不进入默认 RAG 和正式入库。
