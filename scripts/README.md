# scripts 脚本目录

`scripts/` 把 `src/` 中的能力包装成便于本地运行的命令行脚本。它们通常不应该承载核心业务规则。

## 数据如何流动

```text
开发者运行脚本
  -> 调用 src 中的模块
  -> 读取 data/ 或 pipeline 产物
  -> 输出报告、审计结果或新 JSONL
```

## 常见脚本类型

| 类型 | 例子 | 作用 |
| --- | --- | --- |
| 流水线入口 | `run_pipeline.py` | 调用主候选生成流水线 |
| 质量检查 | `check_quality.py` | 聚合测试和 ruff 等质量门 |
| 审计脚本 | `audit_cleaned_records.py`、`initial_association_audit.py` | 检查数据质量和关联问题 |
| 包装入口 | `prepare_review_batch.py`、`rule_assisted_backfill.py` | 调用 review/publish 相关模块 |
| 数据治理 | `data_artifact_manifest.py` | 生成 data 目录产物清单 |

维护时优先把业务逻辑放回 `src/`，脚本只负责参数解析和调用。
