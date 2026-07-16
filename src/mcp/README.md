# mcp 检索服务接口

## 模块职责

| 文件 | 作用 |
| --- | --- |
| `server.py` | 根据 `RAG_BACKEND` 选择 SQLite 或 PostgreSQL 后端并注册 MCP 工具 |
| `server_pg.py` | PostgreSQL 专用、依赖更少的服务入口 |
| `api_search.py`、`api_retrieve.py`、`api_read.py` | 本地后端的搜索、证据检索和全文读取接口 |
| `api_pg.py` | PostgreSQL 后端 JSON API |
| `call_logger.py` | 记录工具名、参数、实际结果摘要、耗时和错误 |
| `smoke_test.py` | 基础服务冒烟检查 |
| `__main__.py` | 包命令行入口 |

## 暴露能力

服务主要提供三类操作：搜索候选文档、读取指定文档、检索带来源信息的证据块。接口输出应保留 `doc_id`、`chunk_id`、标题路径、来源文件和匹配原因，便于调用方追踪证据。

## 后端配置

- `RAG_BACKEND`：`sqlite` 或 `postgres`/`postgresql`。
- PostgreSQL 连接使用 `POSTGRES_DSN`、`DATABASE_URL` 或标准 `PG*` 变量。
- `MCP_TRANSPORT`：`stdio`、`sse` 或 `streamable-http`。
- HTTP 配置包括 `MCP_HOST`、`MCP_PORT`、`MCP_STREAMABLE_HTTP_PATH`、`MCP_SSE_PATH` 和 `MCP_MESSAGE_PATH`。

远程最终服务使用 PostgreSQL 时，优先使用 `server_pg.py`，并设置 `PG_VECTOR_RETRIEVAL_REQUIRED=1` 以防缺失向量时静默退化。

## 调用日志

`MCP_CALL_LOG_ENABLED` 控制日志开关，`MCP_CALL_LOG_PATH` 设置输出路径。结果模式和字段截断由 `MCP_CALL_LOG_RESULT_MODE` 及对应 MAX 变量控制。日志可能包含查询和结果摘要，应按业务数据安全要求保存，不要记录凭据或环境变量值。

## 启动与验证

```powershell
python -B -m src.mcp
python -B -m src.mcp.server_pg
python -B -m pytest tests/test_mcp_call_logger.py -q -p no:cacheprovider
```

部署前先用实际后端执行搜索、读取和检索各一次，确认数据库、模型缓存、向量完整性和日志目录均可用。