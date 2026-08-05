# PostgreSQL 存储与向量化辅助模块。
#
# 提供：
# - PostgreSQL 表结构（documents / cards / views / sections / chunks 及 embedding 表）的 DDL
# - JSONL 到 PostgreSQL 的流式入库
# - BGE-M3 / Qwen3-Embedding-8B 向量化入口
# - 混合检索（全文 + 向量 + RRF + Reranker）
# 远程部署时，DSN 由 /etc/pdf-markdown-rag/postgres.env 提供，本地开发也可通过
# POSTGRES_DSN / DATABASE_URL 或 PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD 注入。