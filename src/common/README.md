# common 模块

`src/common/` 现在只保留当前证据库任务需要的公共入口。

## 文件职责

| 文件 | 作用 |
| --- | --- |
| `mcp_server.py` | 对外提供 MCP 工具：`search`、`read`、`retrieve`。 |

## MCP 工具边界

- `search`：混合检索入口，返回候选 chunk/block。
- `read`：按文档、block 或 chunk 标识读取原文证据。
- `retrieve`：面向 Agent 的完整召回接口，组合检索结果和可读证据内容。

旧的 JSONL 工具、抽取公共类型、质量字段传播等模块已经迁移到 `legacy_recommendation_pipeline/src/common/`。
