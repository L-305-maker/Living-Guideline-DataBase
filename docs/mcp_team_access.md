# 团队 Agent 访问 MCP 部署指南

本文档用于把远程 PostgreSQL `living_guideline` 数据库通过 MCP 以只读工具形式提供给团队成员的 Agent 使用。

## 1. 权限模型

推荐分成两类账号：

- 团队成员：使用 PostgreSQL 只读账号，例如 `lg_readonly`，只能通过 MCP 查询。
- 维护者：使用写入账号，例如 `postgres`，通过项目 CLI 入库、备份或重建表。

MCP 服务本身只暴露安全只读工具，不暴露任意 SQL、删除、更新、入库或 `--recreate`。

## 2. 远程创建只读数据库账号

在远程服务器执行：

```bash
sudo -u postgres psql -d living_guideline
```

进入 psql 后执行，把密码换成团队只读密码：

```sql
CREATE USER lg_readonly WITH PASSWORD 'replace_with_readonly_password';

GRANT CONNECT ON DATABASE living_guideline TO lg_readonly;
GRANT USAGE ON SCHEMA public TO lg_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO lg_readonly;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO lg_readonly;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT SELECT ON TABLES TO lg_readonly;
```

退出：

```sql
\q
```

## 3. 远程安装依赖

在远程服务器项目目录执行：

```bash
cd /home/lhj/project
python -m pip install -r requirements.txt
```

如果远程服务器使用虚拟环境，先激活虚拟环境再安装。

## 4. 远程启动 MCP 服务

建议只监听本机地址 `127.0.0.1`，不要直接暴露公网端口。

```bash
cd /home/lhj/project

export DATABASE_URL="postgresql://lg_readonly:replace_with_readonly_password@127.0.0.1:5432/living_guideline"
export MCP_TRANSPORT="streamable-http"
export MCP_HOST="127.0.0.1"
export MCP_PORT="8000"

python -m src.common.mcp_server
```

服务启动后，MCP 地址为：

```text
http://127.0.0.1:8000/mcp
```

这个地址是在服务器本机可见。团队成员需要通过 SSH 隧道访问。

## 5. 团队成员本地连接 MCP

团队成员在本地开一个终端：

```powershell
ssh -L 18000:127.0.0.1:8000 my-server
```

保持这个窗口打开。

然后在 Agent 客户端里配置 MCP HTTP 地址：

```text
http://127.0.0.1:18000/mcp
```

## 6. 暴露给团队 Agent 的工具

当前 MCP 只提供这些只读工具：

- `health_check`：检查 MCP 是否连上数据库。
- `table_counts`：查看项目表行数。
- `database_overview`：查看质量状态、证据状态和来源分布。
- `source_stats`：查看指南来源统计。
- `search_recommendations`：按关键词搜索推荐。
- `get_recommendation`：查看单条推荐详情。
- `get_recommendation_evidence`：查看单条推荐关联证据。
- `integrity_report`：查看外键闭合和版本质量检查。

## 7. 维护者入库流程

维护者不要通过团队 MCP 入库。建议本地通过 SSH 隧道连接远程 PostgreSQL，然后使用项目 CLI。

本地打开隧道：

```powershell
ssh -L 15433:127.0.0.1:5432 my-server
```

另开 PowerShell：

```powershell
cd D:\python\Data_splitting_ingestion
$env:DATABASE_URL = "postgresql://postgres:replace_with_admin_password@localhost:15433/living_guideline"
```

入库前先备份远程库：

```powershell
& "D:\PSQL\bin\pg_dump.exe" -h localhost -p 15433 -U postgres -d living_guideline -Fc -f "D:\python\Data_splitting_ingestion\remote_backup.dump"
```

校验发布包：

```powershell
python -B -X utf8 -m src.storage.storage_PG validate-run `
  --run-dir data\processed\runs\codex_all_china_deep_20260627_reviewed_pass2\publish_ready_v1
```

入库：

```powershell
python -B -X utf8 -m src.storage.storage_PG ingest-release `
  --release-dir data\processed\runs\codex_all_china_deep_20260627_reviewed_pass2\publish_ready_v1 `
  --recreate
```

验证：

```powershell
python -B -X utf8 -m src.storage.storage_PG counts
python -B -X utf8 -m src.storage.storage_PG integrity-report
```
