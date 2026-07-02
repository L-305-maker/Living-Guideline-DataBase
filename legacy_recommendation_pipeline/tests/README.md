# tests 测试目录

`tests/` 是项目行为的可执行说明书。学习项目时，测试经常比长代码更容易说明“输入是什么、期望输出是什么”。

## 测试数据如何流动

```text
小样本输入 / fixture
  -> 调用 src 模块
  -> 断言输出字段、状态、计数或报告
```

## 测试分组

| 目录 | 验证内容 |
| --- | --- |
| `cleaning/` | 清洗、质量门、offset、record_type |
| `extraction/` | 推荐、GRADE、PICO、证据、source span |
| `orchestration/` | 端到端流水线和公开 API |
| `review_publish/` | 复核、backfill、release export、发布 |
| `storage/` | schema、contract、入库和完整性 |
| `quality/` | goldset、manifest、质量报告 |
| `versioning/` | RecommendationVersion 和 update log |
| `vectorization/` | embedding queue |
| `fixtures/` | 小型稳定测试样本 |

运行全部测试：

```powershell
python -B -X utf8 -m unittest discover -s tests
```
