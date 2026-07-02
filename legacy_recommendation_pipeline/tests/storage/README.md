# storage 测试组

## 覆盖范围

本组测试 PostgreSQL schema、通用入库和原子事务边界：

- 多表 bundle 是否只提交一次。
- 任何表失败时是否整体 rollback。
- 非 publishable version 是否分流到 review queue。
- storage 公开 API 和 SQL identifier 安全。
- live PostgreSQL 发布 smoke test。

## 什么时候必须运行

修改以下模块时必须运行：

```powershell
python -m tests.test_main --group storage
```

- `src/storage/`
- `src/pipeline/publish/` 的入库路径
- schema、migration、generic ingest 配置

## live PostgreSQL 测试

默认跳过。需要真实数据库时显式运行：

```powershell
$env:RUN_POSTGRES_SMOKE="1"
python -m tests.test_main --group storage
```

## 常见失败含义

- 原子入库失败：检查表顺序和外键依赖。
- publish gate 分流失败：检查 `normalized_payload.publish_gate`。
- live smoke 失败：检查 `DATABASE_URL` 和 schema 是否最新。
