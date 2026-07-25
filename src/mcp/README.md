# mcp PostgreSQL 服务

该目录暴露 PostgreSQL-backed 的 MCP 工具：`search`、`read` 和 `retrieve`。服务不读取本地 FAISS 文件，检索逻辑来自 `src.storage.pg_hybrid_retrieval` 和 `src.storage.postgres_store`。

## 文件

| 文件 | 作用 |
| --- | --- |
| `server.py` | 定义 MCP 工具、默认 topk 和服务实例 |
| `api_pg.py` | 将 MCP payload 转为 PostgreSQL Search/Read/Retrieve API 调用 |
| `call_logger.py` | 记录工具调用参数、状态、耗时、摘要和结果，并脱敏敏感字段 |
| `server_common.py` | 共享环境变量读取、payload 构造和服务启动逻辑 |
| `smoke_test.py` | 直接调用 PostgreSQL API 做 search/read/retrieve 冒烟检查 |
| `__main__.py` | `python -m src.mcp` 的启动入口 |

## 工具语义

- `search` 返回候选指南文档，支持机构、科室、时间范围、发布日期和 topk 过滤。
- `read` 通过 `doc_id` 或标题读取文档正文，可用 `max_chars` 限制返回长度。
- `retrieve` 返回命中的 chunk，并带上来源上下文、相邻块和匹配理由。

## 配置

- 数据库连接由 `POSTGRES_DSN`、`DATABASE_URL` 或标准 `PGHOST`、`PGPORT`、`PGDATABASE`、`PGUSER`、`PGPASSWORD` 提供。
- `MCP_TRANSPORT` 支持 `stdio`、`sse` 和 `streamable-http`。
- HTTP 模式可设置 `MCP_HOST`、`MCP_PORT`、`MCP_STREAMABLE_HTTP_PATH`、`MCP_SSE_PATH`、`MCP_MESSAGE_PATH` 和 `MCP_MOUNT_PATH`。
- 设置 `PG_VECTOR_RETRIEVAL_REQUIRED=1` 时，向量缺失会直接失败，不做静默退化。

## 调用日志

`MCP_CALL_LOG_ENABLED` 默认开启。未设置 `MCP_CALL_LOG_PATH` 时，日志写入 `data/evidence/logs/mcp_calls.jsonl`。日志会保留 payload、状态、耗时、`result_summary` 和按长度限制裁剪后的 `result`；密码、Token、密钥等字段会写成 `***`。

## 验证

```powershell
python -B -m src.mcp
python -B -m src.mcp.smoke_test --query "diabetes hypertension guideline" --topk 3
python -B -m pytest tests/test_mcp_call_logger.py -q -p no:cacheprovider
```

冒烟测试依赖可连接的 PostgreSQL 数据库和已入库的 evidence 数据。
