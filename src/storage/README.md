# storage PostgreSQL 与 pgvector

## 模块职责

| 文件 | 作用 |
| --- | --- |
| `postgres_store.py` | 建表、全量入库、统计、PostgreSQL 全文检索和向量索引创建 |
| `vectorize.py` | 读取卡片、视图、分块，使用 Qwen3-Embedding-8B（1024 维）编码并写入 pgvector 表 |
| `query_embedding.py` | 调用 vLLM embedding 服务并生成 pgvector 参数文本 |
| `pg_hybrid_retrieval.py` | 融合 PostgreSQL 词法通道、pgvector 通道和重排结果 |

## 数据关系

`documents` 是父表；`document_cards`、`document_views`、`sections` 和 `chunks` 通过 `doc_id` 关联。向量表以业务 ID 与 `model` 作为版本键，因此同一数据可并存多套模型向量。

## 建库顺序

```powershell
python -B -m src.storage.postgres_store init --with-vector
python -B -m src.storage.postgres_store ingest
python -B -m src.storage.vectorize document_cards
python -B -m src.storage.vectorize document_views
python -B -m src.storage.vectorize chunks
python -B -m src.storage.postgres_store index-vectors
python -B -m src.storage.postgres_store stats
```

命令参数以各模块 `--help` 为准。入库会按外键依赖清空并重建主数据快照；不要在包含未备份人工修改的数据库上直接执行。

## 连接与模型配置

数据库连接优先读取命令行 `--dsn`，随后读取 `POSTGRES_DSN`、`DATABASE_URL` 或标准 `PGHOST`、`PGPORT`、`PGDATABASE`、`PGUSER`、`PGPASSWORD` 环境变量。

- `PG_VECTOR_MODEL`：查询与就绪检查使用的模型，必须与向量化模型一致。
- `VLLM_EMBEDDING_BASE_URL`：vLLM 根地址，默认 `http://127.0.0.1:8001`。
- `VLLM_EMBEDDING_MODEL`：vLLM 的 served model name，默认沿用 `PG_VECTOR_MODEL`。
- `PG_VECTOR_RETRIEVAL_REQUIRED=1`：要求卡片、视图和分块向量全部齐全，失败时禁止退化。

## 向量完整性

强制模式会比较源表记录数与指定模型的向量数。切换模型、修改检索文本或重新入库后，应重新向量化；仅创建 HNSW/IVFFlat 索引不会生成缺失向量。

## 远程部署

代码、模型缓存和数据库数据是独立资产。PostgreSQL 向量存放在数据库中，迁移时应使用数据库备份/恢复或在服务器重新向量化；不要把本地 FAISS 文件当作 PostgreSQL 的必需产物。
