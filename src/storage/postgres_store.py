# PostgreSQL 表结构、JSONL 入库、BM25-like 全文检索的集中实现。
#
# 模块职责：
# - 定义 5 张主表 DDL（documents / document_cards / document_views / sections / chunks）
#   以及 3 张向量表（document_card_embeddings / document_view_embeddings / chunk_embeddings）。
# - 提供 init / reset / ingest / ingest-sections / index-vectors / stats / verify
#   七个 CLI 子命令，覆盖建库到检索就绪的全流程。
# - 提供 search_*_pg / retrieve_chunks_pg / read_document_pg 三类 BM25 检索入口，
#   与 pg_hybrid_retrieval.py 配合组成全文 + 向量的混合检索。
#
# 关键约定：
# - 业务 ID（doc_id / view_id / chunk_id）跨 JSONL / PostgreSQL / MCP 全链路保持一致。
# - retrieval_text 字段作为 BM25 检索口径（见 chunks.content_tsv），content 仅作展示。
# - 删除主表会通过 ON DELETE CASCADE 自动清理子表和 embedding 表。

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Iterator
from psycopg_pool import ConnectionPool
import threading
from src.retrieval.chunk_normalizer import iter_normalized_chunks
from src.storage.query_embedding import DEFAULT_MODEL
from src.utils.io import DATA_DIR, read_jsonl
from src.utils.records import (
    department_text as helper_department_text,
    publication_year as helper_year,
    truthy as helper_truthy,
)

POOL = None
POOL_LOCK = threading.Lock()

# 5 张主表 + 16 个普通/全文索引。
# 全文索引使用 GIN(content_tsv)；其中 content_tsv 是 PG 自动生成的 GENERATED 列，
# 由 title(A) + section_path_text(B) + clinical_department(B) + retrieval_text/content(C)
# 加权拼接，便于在 BM25-like 检索中区分关键字段的贡献度。
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL DEFAULT '',
    publication_date TEXT NOT NULL DEFAULT 'unknown',
    publication_year INTEGER,
    source_institution TEXT NOT NULL,
    clinical_department TEXT NOT NULL DEFAULT '未分类',
    document_kind TEXT NOT NULL DEFAULT 'guideline',
    source_file TEXT NOT NULL,
    markdown_raw_path TEXT NOT NULL,
    markdown_clean_path TEXT NOT NULL,
    content_sha256 TEXT NOT NULL DEFAULT '',
    content_md TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS document_cards (
    doc_id TEXT PRIMARY KEY REFERENCES documents(doc_id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    publication_date TEXT NOT NULL DEFAULT 'unknown',
    publication_year INTEGER,
    source_institution TEXT NOT NULL DEFAULT 'Unknown',
    clinical_department TEXT NOT NULL DEFAULT '未分类',
    markdown_clean_path TEXT NOT NULL DEFAULT '',
    cleaning_quality TEXT NOT NULL DEFAULT '',
    cleaning_flags TEXT NOT NULL DEFAULT '',
    source_pdf_text_quality TEXT NOT NULL DEFAULT '',
    source_pdf_needs_ocr BOOLEAN NOT NULL DEFAULT FALSE,
    source_pdf_is_scanned BOOLEAN NOT NULL DEFAULT FALSE,
    pdf_text_quality TEXT NOT NULL DEFAULT '',
    pdf_needs_ocr BOOLEAN NOT NULL DEFAULT FALSE,
    pdf_is_scanned BOOLEAN NOT NULL DEFAULT FALSE,
    ocr_engine TEXT NOT NULL DEFAULT '',
    ocr_applied BOOLEAN NOT NULL DEFAULT FALSE,
    ocr_status TEXT NOT NULL DEFAULT '',
    ocr_error TEXT NOT NULL DEFAULT '',
    card_text TEXT NOT NULL DEFAULT '',
    fields_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    content_tsv TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('simple', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('simple', coalesce(clinical_department, '')), 'B') ||
        setweight(to_tsvector('simple', coalesce(card_text, '')), 'C')
    ) STORED
);

CREATE TABLE IF NOT EXISTS document_views (
    view_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    view_type TEXT NOT NULL,
    priority REAL NOT NULL DEFAULT 1.0,
    title TEXT NOT NULL,
    publication_date TEXT NOT NULL DEFAULT 'unknown',
    publication_year INTEGER,
    source_institution TEXT NOT NULL DEFAULT 'Unknown',
    clinical_department TEXT NOT NULL DEFAULT '未分类',
    markdown_clean_path TEXT NOT NULL DEFAULT '',
    cleaning_quality TEXT NOT NULL DEFAULT '',
    cleaning_flags TEXT NOT NULL DEFAULT '',
    source_pdf_text_quality TEXT NOT NULL DEFAULT '',
    source_pdf_needs_ocr BOOLEAN NOT NULL DEFAULT FALSE,
    source_pdf_is_scanned BOOLEAN NOT NULL DEFAULT FALSE,
    pdf_text_quality TEXT NOT NULL DEFAULT '',
    pdf_needs_ocr BOOLEAN NOT NULL DEFAULT FALSE,
    pdf_is_scanned BOOLEAN NOT NULL DEFAULT FALSE,
    ocr_engine TEXT NOT NULL DEFAULT '',
    ocr_applied BOOLEAN NOT NULL DEFAULT FALSE,
    ocr_status TEXT NOT NULL DEFAULT '',
    ocr_error TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL DEFAULT '',
    content_tsv TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('simple', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('simple', coalesce(view_type, '')), 'B') ||
        setweight(to_tsvector('simple', coalesce(clinical_department, '')), 'B') ||
        setweight(to_tsvector('simple', coalesce(text, '')), 'C')
    ) STORED
);

CREATE TABLE IF NOT EXISTS sections (
    section_id BIGSERIAL PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    section_index INTEGER NOT NULL,
    title TEXT NOT NULL,
    publication_date TEXT NOT NULL DEFAULT 'unknown',
    publication_year INTEGER,
    source_institution TEXT NOT NULL,
    clinical_department TEXT NOT NULL DEFAULT '未分类',
    section_path JSONB NOT NULL DEFAULT '[]'::jsonb,
    heading TEXT,
    heading_level INTEGER,
    char_start INTEGER,
    char_end INTEGER,
    is_reference_section BOOLEAN NOT NULL DEFAULT FALSE,
    content TEXT NOT NULL DEFAULT '',
    UNIQUE (doc_id, section_index)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    publication_date TEXT NOT NULL DEFAULT 'unknown',
    publication_year INTEGER,
    source_institution TEXT NOT NULL,
    clinical_department TEXT NOT NULL DEFAULT '未分类',
    section_path JSONB NOT NULL DEFAULT '[]'::jsonb,
    section_path_text TEXT NOT NULL DEFAULT '',
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    retrieval_text TEXT NOT NULL DEFAULT '',
    chunk_type TEXT NOT NULL DEFAULT 'other',
    text_for_embedding TEXT NOT NULL DEFAULT '',
    recommendation TEXT NOT NULL DEFAULT '',
    evidence JSONB NOT NULL DEFAULT '[]'::jsonb,
    chunk_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    retrieval_key TEXT NOT NULL UNIQUE,
    source_file TEXT NOT NULL DEFAULT '',
    markdown_clean_path TEXT NOT NULL DEFAULT '',
    is_reference_section BOOLEAN NOT NULL DEFAULT FALSE,
    content_tsv TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('simple', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('simple', coalesce(section_path_text, '')), 'B') ||
        setweight(to_tsvector('simple', coalesce(clinical_department, '')), 'B') ||
        setweight(to_tsvector('simple', coalesce(chunk_type, '')), 'B') ||
        setweight(to_tsvector('simple', coalesce(retrieval_text, content, '')), 'C')
    ) STORED
);

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS document_kind TEXT NOT NULL DEFAULT 'guideline';

CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_institution);
CREATE INDEX IF NOT EXISTS idx_documents_department ON documents(clinical_department);
CREATE INDEX IF NOT EXISTS idx_documents_kind_department ON documents(document_kind, clinical_department);
CREATE INDEX IF NOT EXISTS idx_documents_publication_date ON documents(publication_date);
CREATE INDEX IF NOT EXISTS idx_documents_year ON documents(publication_year);
CREATE INDEX IF NOT EXISTS idx_document_cards_source ON document_cards(source_institution);
CREATE INDEX IF NOT EXISTS idx_document_cards_department ON document_cards(clinical_department);
CREATE INDEX IF NOT EXISTS idx_document_cards_date ON document_cards(publication_date);
CREATE INDEX IF NOT EXISTS idx_document_cards_content_tsv ON document_cards USING gin(content_tsv);
CREATE INDEX IF NOT EXISTS idx_document_views_doc ON document_views(doc_id);
CREATE INDEX IF NOT EXISTS idx_document_views_type ON document_views(view_type);
CREATE INDEX IF NOT EXISTS idx_document_views_source ON document_views(source_institution);
CREATE INDEX IF NOT EXISTS idx_document_views_department ON document_views(clinical_department);
CREATE INDEX IF NOT EXISTS idx_document_views_date ON document_views(publication_date);
CREATE INDEX IF NOT EXISTS idx_document_views_content_tsv ON document_views USING gin(content_tsv);
CREATE INDEX IF NOT EXISTS idx_sections_doc ON sections(doc_id);
CREATE INDEX IF NOT EXISTS idx_sections_department ON sections(clinical_department);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_institution);
CREATE INDEX IF NOT EXISTS idx_chunks_department ON chunks(clinical_department);
CREATE INDEX IF NOT EXISTS idx_chunks_publication_date ON chunks(publication_date);
CREATE INDEX IF NOT EXISTS idx_chunks_year ON chunks(publication_year);
CREATE INDEX IF NOT EXISTS idx_chunks_content_tsv ON chunks USING gin(content_tsv);
"""


# pgvector 扩展和 3 张 embedding 表。
# 主键 (业务ID, model)：允许同一文档对应多套向量（不同模型/不同 MRL 维度）。
# dim 列记录实际维度，便于混合维度向量的诊断（实际业务统一 1024）。
VECTOR_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_card_embeddings (
    doc_id TEXT NOT NULL REFERENCES document_cards(doc_id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    embedding vector(1024) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (doc_id, model)
);

CREATE TABLE IF NOT EXISTS document_view_embeddings (
    view_id TEXT NOT NULL REFERENCES document_views(view_id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    embedding vector(1024) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (view_id, model)
);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    embedding vector(1024) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (chunk_id, model)
);

"""


# IVFFlat 向量索引。
# lists=100 是经验值：数据量 1 万级时聚类质量与查询速度较平衡。
# 大规模数据应按 sqrt(N) 调整；过小导致聚类不准，过大导致索引过大、查询慢。
VECTOR_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_document_card_embeddings_cosine
    ON document_card_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_document_view_embeddings_cosine
    ON document_view_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_cosine
    ON chunk_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
"""


def get_dsn(cli_dsn: str | None = None) -> str:
    #按优先级解析 PostgreSQL DSN（最高优先级在前）。
    #解析顺序：
    # 1. CLI 参数 --dsn（最优先，常用于测试或多环境切换）
    # 2. 环境变量 POSTGRES_DSN / DATABASE_URL（远程部署默认走这里）
    # 3. 拆分式环境变量 PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD（云厂商托管 PG 常用）
    #任一命中即返回，否则抛 RuntimeError，避免悄悄连到错误库。
    if cli_dsn:
        return cli_dsn
    for name in ["POSTGRES_DSN", "DATABASE_URL"]:
        value = os.environ.get(name)
        if value:
            return value
    host = os.environ.get("PGHOST")
    db = os.environ.get("PGDATABASE")
    user = os.environ.get("PGUSER")
    password = os.environ.get("PGPASSWORD")
    port = os.environ.get("PGPORT", "5432")
    if host and db and user:
        auth = f"{user}:{password}" if password else user
        return f"postgresql://{auth}@{host}:{port}/{db}"
    raise RuntimeError("PostgreSQL DSN not configured. Set POSTGRES_DSN or DATABASE_URL.")


def get_pool(dsn=None):
    global POOL
    if POOL is not None:
        return POOL
    with POOL_LOCK:
        if POOL is None:
            POOL = ConnectionPool(
                conninfo=get_dsn(dsn),
                min_size=int(os.getenv("PG_POOL_MIN", "2")),
                max_size=int(os.getenv("PG_POOL_MAX", "16")),
                timeout=30.0,
                kwargs={"options": "-c statement_timeout=8000"},
                open=False,                   # 延迟开
                name="mcp-pg-pool",
            )
            POOL.open(wait=True, timeout=10.0)
    return POOL


class PooledConn:
    def __init__(self, pool): self._pool = pool; self._conn = None

    def __enter__(self):
        self._conn = self._pool.getconn(timeout=10)
        return self._conn
    
    def __exit__(self, *a):
        try:
            if a[0] is not None:
                self._conn.rollback()
        finally:
            self._pool.putconn(self._conn)
            self._conn = None

"""
def connect(dsn=None):
    import psycopg

    conn = psycopg.connect(get_dsn(dsn),

        options=f"-c statement_timeout={int(os.getenv('PG_STATEMENT_TIMEOUT_MS', '8000'))}"
                 f" -c idle_in_transaction_session_timeout=10s"
                 f" -c lock_timeout={int(os.getenv('PG_LOCK_TIMEOUT_MS', '3000'))}")
    # 一个慢 query 不再卡死 worker
    return conn
"""

def serialize_json(value: Any) -> str:
    # 将 Python 对象序列化为 JSON 字符串（ensure_ascii=False 保留中文字符）。

    # 主要用于向 PG 的 JSONB 列传值：JSONB 列接受字符串并自动 parse，
    # ensure_ascii=False 让存储原文可读，便于 pg_dump / 查询时直接阅读。
    return json.dumps(value, ensure_ascii=False)


def init_schema(dsn: str | None = None, with_vector: bool = False) -> dict[str, Any]:
    # 幂等创建主表 + pg_trgm 扩展；可选创建向量表。

    # pg_trgm 用于 ILIKE 模糊匹配的 GIN 索引加速（如 source_institution ILIKE '%xxx%'）。
    # 向量表与扩展只在 with_vector=True 时创建，便于分阶段建库（先文本后向量）。
    
    with PooledConn(get_pool(dsn)) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            cur.execute(SCHEMA_SQL)
            if with_vector:
                cur.execute(VECTOR_SCHEMA_SQL)
        conn.commit()
    return {"ok": True, "schema": "created", "vector_schema": with_vector}


def create_vector_indexes(dsn: str | None = None) -> dict[str, Any]:
    # 建 IVFFlat 向量索引 + ANALYZE 收集统计信息。

    # 调用时机：所有 embedding 行写入完成之后立即执行。
    # - IVFFlat 索引是必须的，否则向量召回是顺序扫描，性能不可接受；
    # - ANALYZE 让查询规划器知道表大小与数据分布，避免向量召回选择错误的执行计划。
    
    with PooledConn(get_pool(dsn)) as conn:
        with conn.cursor() as cur:
            cur.execute(VECTOR_INDEX_SQL)
            for table in ("document_card_embeddings", "document_view_embeddings", "chunk_embeddings"):
                cur.execute(f"ANALYZE {table}")
        conn.commit()
    return {"ok": True, "vector_indexes": "created"}


def reset_schema(dsn: str | None = None, with_vector: bool = False) -> dict[str, Any]:
    # DROP 所有 9 张表（先 embeddings 再主表，避开外键依赖）后调用 init_schema 重建。

    # DROP 顺序（重要）：
    #  chunk_embeddings → document_view_embeddings → document_card_embeddings
    #  → document_embeddings（预留位，当前未使用但保留 DROP）
    #  → chunks → sections → document_views → document_cards → documents
    # 必须在有外键依赖的子表（embeddings）先于主表 DROP，否则 PG 会拒绝。
    
    with PooledConn(get_pool(dsn)) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS chunk_embeddings")
            cur.execute("DROP TABLE IF EXISTS document_view_embeddings")
            cur.execute("DROP TABLE IF EXISTS document_card_embeddings")
            cur.execute("DROP TABLE IF EXISTS document_embeddings")
            cur.execute("DROP TABLE IF EXISTS chunks")
            cur.execute("DROP TABLE IF EXISTS sections")
            cur.execute("DROP TABLE IF EXISTS document_views")
            cur.execute("DROP TABLE IF EXISTS document_cards")
            cur.execute("DROP TABLE IF EXISTS documents")
        conn.commit()
    return init_schema(dsn, with_vector)




def helper_data_artifact_path(data_dir: Path, stored_path: str | None, folder: str, doc_id: str) -> Path:
    # 解析 Markdown 工件路径，兼容"JSONL 中的路径来自其他机器"的情况。
    # 优先级：
    #  1. stored_path 指向的文件确实存在 → 直接使用；
    #  2. 否则回退到 data_dir/folder/<doc_id>.md（本地约定的标准布局）。
    # 这一兼容层让多机器协作时 JSONL 携带的旧路径不会让 ingest 失败。

    path = Path(stored_path or "")
    if stored_path and path.is_file():
        return path
    return data_dir / folder / f"{doc_id}.md"


def iter_document_rows(data_dir: Path, allowed_doc_ids: set[str] | None = None) -> Iterator[tuple[Any, ...]]:
    # 流式产出 documents 表的入库行（行序与 INSERT 列定义严格对齐）。
    # 行为要点：
    # - 跳过不在 allowed_doc_ids 集合中的 doc_id（增量/筛选场景）；
    # - clean_path 缺失即抛 FileNotFoundError，避免写入半残数据；
    # - content_md 字段直接读 markdown_clean 全文（用于 read_document_pg 返回正文）。
    
    for rec in read_jsonl(data_dir / "documents.jsonl"):
        doc_id = rec["doc_id"]
        if allowed_doc_ids is not None and doc_id not in allowed_doc_ids:
            continue
        clean_path = helper_data_artifact_path(data_dir, rec.get("markdown_clean_path"), "markdown_clean", doc_id)
        raw_path = helper_data_artifact_path(data_dir, rec.get("markdown_raw_path"), "markdown_raw", doc_id)
        if not clean_path.is_file():
            raise FileNotFoundError(f"Missing clean markdown for {doc_id}: {clean_path}")
        yield (
            doc_id,
            rec["title"],
            rec.get("abstract") or "",
            rec.get("publication_date") or "unknown",
            helper_year(rec.get("publication_date")),
            rec["source_institution"],
            helper_department_text(rec),
            rec.get("document_kind") or "guideline",
            rec["source_file"],
            str(raw_path),
            str(clean_path),
            rec.get("content_sha256") or "",
            clean_path.read_text(encoding="utf-8", errors="replace"),
        )


def helper_document_rows(data_dir: Path) -> list[tuple[Any, ...]]:
    # 把 iter_document_rows 物化为 list；仅供一次性内存场景使用。
    return list(iter_document_rows(data_dir))


def iter_document_card_rows(data_dir: Path, allowed_doc_ids: set[str] | None = None) -> Iterator[tuple[Any, ...]]:
    # 流式产出 document_cards 表的入库行（含 OCR 元信息与清洗审计字段）。
    # 与 documents 入库的区别：
    # - 继承 documents 的 OCR / 清洗字段以支持按 document_kind 检索时的过滤；
    # - 过滤条件为 allowed_doc_ids（增量模式下只入库目标集合）；
    # - fields_json 字段以 JSONB 写入，承载 JSONL 中的 fields 富字段。
    
    path = data_dir / "document_cards.jsonl"
    if not path.exists():
        return
    for rec in read_jsonl(path):
        doc_id = rec.get("doc_id")
        if allowed_doc_ids is not None and doc_id not in allowed_doc_ids:
            continue
        yield (
            doc_id,
            rec.get("title") or "",
            rec.get("publication_date") or "unknown",
            helper_year(rec.get("publication_date")),
            rec.get("source_institution") or "Unknown",
            helper_department_text(rec),
            str(helper_data_artifact_path(data_dir, rec.get("markdown_clean_path"), "markdown_clean", doc_id)),
            rec.get("cleaning_quality") or "",
            rec.get("cleaning_flags") or "",
            rec.get("source_pdf_text_quality") or "",
            helper_truthy(rec.get("source_pdf_needs_ocr")),
            helper_truthy(rec.get("source_pdf_is_scanned")),
            rec.get("pdf_text_quality") or "",
            helper_truthy(rec.get("pdf_needs_ocr")),
            helper_truthy(rec.get("pdf_is_scanned")),
            rec.get("ocr_engine") or "",
            helper_truthy(rec.get("ocr_applied")),
            rec.get("ocr_status") or "",
            rec.get("ocr_error") or "",
            rec.get("card_text") or "",
            serialize_json(rec.get("fields") or {}),
        )


def iter_document_view_rows(data_dir: Path, allowed_doc_ids: set[str | None] = None) -> Iterator[tuple[Any, ...]]:
    # 流式产出 document_views 表的入库行。
    # 关键差异：
    # - 视图行继承文档级质量信息，但保留独立 view_id、view_type 和 priority；
    # - view_id 缺失时由 "<doc_id>__<view_type>" 回填，保证主键非空且唯一。

    path = data_dir / "document_views.jsonl"
    if not path.exists():
        return
    for rec in read_jsonl(path):
        doc_id = rec.get("doc_id")
        if allowed_doc_ids is not None and doc_id not in allowed_doc_ids:
            continue
        yield (
            rec.get("view_id") or f"{doc_id}__{rec.get('view_type')}",
            doc_id,
            rec.get("view_type") or "",
            float(rec.get("priority") or 1.0),
            rec.get("title") or "",
            rec.get("publication_date") or "unknown",
            helper_year(rec.get("publication_date")),
            rec.get("source_institution") or "Unknown",
            helper_department_text(rec),
            str(helper_data_artifact_path(data_dir, rec.get("markdown_clean_path"), "markdown_clean", doc_id)),
            rec.get("cleaning_quality") or "",
            rec.get("cleaning_flags") or "",
            rec.get("source_pdf_text_quality") or "",
            helper_truthy(rec.get("source_pdf_needs_ocr")),
            helper_truthy(rec.get("source_pdf_is_scanned")),
            rec.get("pdf_text_quality") or "",
            helper_truthy(rec.get("pdf_needs_ocr")),
            helper_truthy(rec.get("pdf_is_scanned")),
            rec.get("ocr_engine") or "",
            helper_truthy(rec.get("ocr_applied")),
            rec.get("ocr_status") or "",
            rec.get("ocr_error") or "",
            rec.get("text") or "",
        )


def iter_section_rows(data_dir: Path, allowed_doc_ids: set[str] | None = None) -> Iterator[tuple[Any, ...]]:
    # 流式产出 sections 表的入库行，遍历 sections/*.jsonl 各文件并按文件内顺序索引。
    # section_index 是文件内序号（从 0 起），与 (doc_id, section_index) 唯一约束对齐。
    
    for path in sorted((data_dir / "sections").glob("*.jsonl")):
        for index, rec in enumerate(read_jsonl(path)):
            if allowed_doc_ids is not None and rec["doc_id"] not in allowed_doc_ids:
                continue
            yield (
                rec["doc_id"],
                index,
                rec["title"],
                rec.get("publication_date") or "unknown",
                helper_year(rec.get("publication_date")),
                rec["source_institution"],
                helper_department_text(rec),
                serialize_json(rec.get("section_path") or []),
                rec.get("heading"),
                rec.get("heading_level"),
                rec.get("char_start"),
                rec.get("char_end"),
                bool(rec.get("is_reference_section")),
                rec.get("content") or "",
            )


def iter_chunk_rows(data_dir: Path, allowed_doc_ids: set[str] | None = None) -> Iterator[tuple[Any, ...]]:
    # 流式产出 chunks 表的入库行；通过 iter_normalized_chunks 收敛字段版本。
    # 关键兼容点：
    # - 使用 chunk_normalizer 把不同阶段的 chunk 字段统一为单一契约；
    # - text_for_embedding / retrieval_text / content 的优先级 fallback 也在 normalizer 内处理；
    # - section_path_text 由 list 拼接得到，供 content_tsv 加权检索。
    
    for rec in iter_normalized_chunks(data_dir):
        if allowed_doc_ids is not None and rec["doc_id"] not in allowed_doc_ids:
            continue
        yield (
            rec["chunk_id"],
            rec["doc_id"],
            rec["title"],
            rec.get("publication_date") or "unknown",
            helper_year(rec.get("publication_date")),
            rec["source_institution"],
            helper_department_text(rec),
            serialize_json(rec.get("section_path") or []),
            " ".join(str(item) for item in (rec.get("section_path") or [])),
            rec["chunk_index"],
            rec.get("content") or "",
            rec.get("retrieval_text") or rec.get("content") or "",
            rec.get("chunk_type") or "other",
            rec.get("text_for_embedding") or rec.get("retrieval_text") or rec.get("content") or "",
            rec.get("recommendation") or "",
            serialize_json(rec.get("evidence") or []),
            serialize_json(rec.get("metadata") or {}),
            rec["retrieval_key"],
            rec.get("source_file") or "",
            str(helper_data_artifact_path(data_dir, rec.get("markdown_clean_path"), "markdown_clean", rec["doc_id"])),
            bool(rec.get("is_reference_section")),
        )


def helper_document_card_rows(data_dir: Path, allowed_doc_ids: set[str] | None = None) -> list[tuple[Any, ...]]:
    # 物化卡片行；用于外部脚本调试或单次全量场景。
    return list(iter_document_card_rows(data_dir, allowed_doc_ids))

def helper_document_view_rows(data_dir: Path, allowed_doc_ids: set[str] | None = None) -> list[tuple[Any, ...]]:
    # 物化视图行。
    return list(iter_document_view_rows(data_dir, allowed_doc_ids))

def helper_section_rows(data_dir: Path) -> list[tuple[Any, ...]]:
    # 物化章节行。
    return list(iter_section_rows(data_dir))

def helper_chunk_rows(data_dir: Path) -> list[tuple[Any, ...]]:
    # 物化分块行。
    return list(iter_chunk_rows(data_dir))


def helper_insert_batches(conn: Any,sql: str,rows: Iterable[tuple[Any, ...]],batch_size: int,commit_each_batch: bool = True,) -> int:
    # 把 rows 按 batch_size 切片执行 executemany，返回累计插入行数。
    # 关键设计：
    # - 每批 executemany 后可选 commit（默认 commit），限制单事务大小；
    # - 最后一批 < batch_size 也会被 flush，避免遗漏；
    # - cur.rowcount 在 psycopg v3 对 executemany 返回值不稳定，回退到 batch 长度计数。
    
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    count = 0
    batch: list[tuple[Any, ...]] = []
    with conn.cursor() as cur:
        for row in rows:
            batch.append(row)
            if len(batch) < batch_size:
                continue
            cur.executemany(sql, batch)
            if commit_each_batch:
                conn.commit()
            count += cur.rowcount if cur.rowcount >= 0 else len(batch)
            batch.clear()
        if batch:
            cur.executemany(sql, batch)
            if commit_each_batch:
                conn.commit()
            count += cur.rowcount if cur.rowcount >= 0 else len(batch)
    return count


def ingest_data(dsn: str | None = None,data_dir: str | Path = DATA_DIR,batch_size: int = 500,incremental: bool = False,) -> dict[str, Any]:
    # 将 5 张主表的 JSONL 流式入库 PostgreSQL。

    # 两种模式：
    # - 全量（默认）：DELETE FROM 5 张表后 INSERT；批量提交，断点处可能产生空快照。
    #   因为 DELETE 不在同一事务内，最后一个 commit 之后 SELECT 会看到空数据。
    # - 增量（--incremental）：对 documents 加 SHARE ROW EXCLUSIVE MODE 锁，
    #   用 ON CONFLICT DO NOTHING 跳过已存在的 doc_id，整批完成后再统一 commit。
    # 边界：
    # - documents.jsonl 出现重复 doc_id 直接抛 ValueError，避免静默吞掉；
    # - INSERT 列顺序与各 iter_*_rows 函数 yield 的元组严格对应，重构时必须同步改两边。

    data_path = Path(data_dir)
    input_doc_ids = [rec["doc_id"] for rec in read_jsonl(data_path / "documents.jsonl")]
    allowed_doc_ids = set(input_doc_ids)
    if len(allowed_doc_ids) != len(input_doc_ids):
        raise ValueError("documents.jsonl contains duplicate doc_id values")

    with PooledConn(get_pool(dsn)) as conn:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = 0")
        existing_doc_ids: set[str] = set()
        if incremental:
            if allowed_doc_ids:
                with conn.cursor() as cur:
                    # SHARE ROW EXCLUSIVE MODE：阻止其他事务并发写入 documents，
                    # 但允许读取。配合 ON CONFLICT DO NOTHING 保证增量入库原子性。
                    cur.execute("LOCK TABLE documents IN SHARE ROW EXCLUSIVE MODE")
                    cur.execute("SELECT doc_id FROM documents WHERE doc_id = ANY(%s)", (sorted(allowed_doc_ids),))
                    existing_doc_ids = {row[0] for row in cur.fetchall()}
            target_doc_ids = allowed_doc_ids - existing_doc_ids
        else:
            with conn.cursor() as cur:
                # DELETE 顺序：先 chunks → sections → document_views → document_cards → documents，
                # 避开外键依赖（与 reset_schema 的 DROP 顺序同理）。
                cur.execute("DELETE FROM chunks")
                cur.execute("DELETE FROM sections")
                cur.execute("DELETE FROM document_views")
                cur.execute("DELETE FROM document_cards")
                cur.execute("DELETE FROM documents")
            conn.commit()
            target_doc_ids = allowed_doc_ids

        conflict_clause = " ON CONFLICT DO NOTHING" if incremental else ""
        # 增量模式走单事务提交，避免半成品 doc_id 落到库内；全量模式每批提交以减小事务体积。
        commit_each_batch = not incremental
        counts = {
            "documents": helper_insert_batches(
                conn,
                f"""
                INSERT INTO documents
                (doc_id, title, abstract, publication_date, publication_year, source_institution, clinical_department, document_kind, source_file,
                 markdown_raw_path, markdown_clean_path, content_sha256, content_md)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                {conflict_clause}
                """,
                iter_document_rows(data_path, target_doc_ids),
                batch_size,
                commit_each_batch,
            ),
            "document_cards": helper_insert_batches(
                conn,
                f"""
                INSERT INTO document_cards
                (doc_id, title, publication_date, publication_year, source_institution, clinical_department,
                 markdown_clean_path, cleaning_quality, cleaning_flags,
                 source_pdf_text_quality, source_pdf_needs_ocr, source_pdf_is_scanned,
                 pdf_text_quality, pdf_needs_ocr, pdf_is_scanned, ocr_engine, ocr_applied, ocr_status, ocr_error,
                 card_text, fields_json)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                {conflict_clause}
                """,
                iter_document_card_rows(data_path, target_doc_ids),
                batch_size,
                commit_each_batch,
            ),
            "document_views": helper_insert_batches(
                conn,
                f"""
                INSERT INTO document_views
                (view_id, doc_id, view_type, priority, title, publication_date, publication_year,
                 source_institution, clinical_department, markdown_clean_path, cleaning_quality, cleaning_flags,
                 source_pdf_text_quality, source_pdf_needs_ocr, source_pdf_is_scanned,
                 pdf_text_quality, pdf_needs_ocr, pdf_is_scanned, ocr_engine, ocr_applied, ocr_status, ocr_error,
                 text)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                {conflict_clause}
                """,
                iter_document_view_rows(data_path, target_doc_ids),
                batch_size,
                commit_each_batch,
            ),
            "sections": helper_insert_batches(
                conn,
                f"""
                INSERT INTO sections
                (doc_id, section_index, title, publication_date, publication_year, source_institution, clinical_department, section_path,
                 heading, heading_level, char_start, char_end, is_reference_section, content)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s,%s)
                {conflict_clause}
                """,
                iter_section_rows(data_path, target_doc_ids),
                batch_size,
                commit_each_batch,
            ),
            "chunks": helper_insert_batches(
                conn,
                f"""
                INSERT INTO chunks
                (chunk_id, doc_id, title, publication_date, publication_year, source_institution, clinical_department, section_path,
                 section_path_text, chunk_index, content, retrieval_text, chunk_type, text_for_embedding, recommendation, evidence,
                 chunk_metadata, retrieval_key, source_file, markdown_clean_path, is_reference_section)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s,%s)
                {conflict_clause}
                """,
                iter_chunk_rows(data_path, target_doc_ids),
                batch_size,
                commit_each_batch,
            ),
        }
        if incremental:
            conn.commit()
            counts["skipped_existing_documents"] = len(existing_doc_ids)
    return counts


def ingest_sections(dsn: str | None = None,data_dir: str | Path = DATA_DIR,batch_size: int = 500,) -> dict[str, int]:
    # 仅入库 sections 表，常用于 sections 重建（不影响 documents/cards/views/chunks）。
    # 安全策略：
    #  - 仅入库 documents 表中已存在的 doc_id 对应的 sections（用 existing_doc_ids 限定）；
    #  - 使用 ON CONFLICT DO NOTHING，跳过重复 (doc_id, section_index)，允许多次执行而不出错。

    data_path = Path(data_dir)
    input_doc_ids = [rec["doc_id"] for rec in read_jsonl(data_path / "documents.jsonl")]
    allowed_doc_ids = set(input_doc_ids)
    if len(allowed_doc_ids) != len(input_doc_ids):
        raise ValueError("documents.jsonl contains duplicate doc_id values")

    with PooledConn(get_pool(dsn)) as conn:
        existing_doc_ids: set[str] = set()
        if allowed_doc_ids:
            with conn.cursor() as cur:
                cur.execute("SELECT doc_id FROM documents WHERE doc_id = ANY(%s)", (sorted(allowed_doc_ids),))
                existing_doc_ids = {row[0] for row in cur.fetchall()}
        inserted = helper_insert_batches(
            conn,
            """
            INSERT INTO sections
            (doc_id, section_index, title, publication_date, publication_year, source_institution, clinical_department, section_path,
             heading, heading_level, char_start, char_end, is_reference_section, content)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s,%s)
            ON CONFLICT DO NOTHING
            """,
            iter_section_rows(data_path, existing_doc_ids),
            batch_size,
            commit_each_batch=False,
        )
        conn.commit()
    return {"matched_documents": len(existing_doc_ids), "sections_inserted": inserted}


def helper_kind_counts(cur: Any, table: str) -> dict[str, int]:
    # 统计某张表按 document_kind 分布的行数（guideline / consensus 等）。
    # 对 documents 直接 GROUP BY；其它表需 JOIN documents 取 document_kind 字段。
    # 用于 verify / stats 子命令检查数据完整性。
    
    if table == "documents":
        cur.execute("SELECT document_kind, count(*) FROM documents GROUP BY document_kind ORDER BY document_kind")
    else:
        cur.execute(
            f"""
            SELECT d.document_kind, count(*)
            FROM {table} item
            JOIN documents d ON d.doc_id = item.doc_id
            GROUP BY d.document_kind
            ORDER BY d.document_kind
            """
        )
    return {str(kind): int(count) for kind, count in cur.fetchall()}


def database_stats(dsn: str | None = None) -> dict[str, Any]:
    # 汇总当前数据库的表行数、按 document_kind 分布、source 分布等统计信息。
    # 向量表用 to_regclass() 检查存在性（可能在 with_vector=False 时未建），
    # 仅对存在的表读 count(*)，避免 SQL 报错。
    
    with PooledConn(get_pool(dsn)) as conn:
        with conn.cursor() as cur:
            result: dict[str, Any] = {}
            for table in ["documents", "document_cards", "document_views", "sections", "chunks"]:
                cur.execute("SELECT to_regclass(%s) IS NOT NULL", (table,))
                if cur.fetchone()[0]:
                    cur.execute(f"SELECT count(*) FROM {table}")
                    result[table] = int(cur.fetchone()[0])
                    result[f"{table}_by_document_kind"] = helper_kind_counts(cur, table)
                else:
                    result[table] = 0
            cur.execute("SELECT source_institution, count(*) FROM documents GROUP BY source_institution ORDER BY count(*) DESC")
            result["source_distribution"] = cur.fetchall()
            cur.execute("SELECT count(*) FROM documents WHERE publication_date='unknown'")
            result["unknown_publication_date_documents"] = int(cur.fetchone()[0])
            cur.execute("SELECT count(*) FROM chunks WHERE publication_date='unknown'")
            result["unknown_publication_date_chunks"] = int(cur.fetchone()[0])
            for table in ["document_card_embeddings", "document_view_embeddings", "chunk_embeddings"]:
                cur.execute("SELECT to_regclass(%s) IS NOT NULL", (table,))
                exists = bool(cur.fetchone()[0])
                result[f"{table}_table"] = exists
                if exists:
                    cur.execute(f"SELECT count(*) FROM {table}")
                    result[table] = int(cur.fetchone()[0])
    return result


def verify_retrieval_snapshot(dsn: str | None = None, model_name: str | None = None) -> dict[str, Any]:
    # 校验检索快照完整性，任何不匹配立即抛 RuntimeError。
    # 检查项：
    # 1. documents 中至少存在 guideline 与 consensus 两种 document_kind；
    # 2. document_cards / document_views / sections / chunks 对应每种 document_kind 都非空；
    # 3. 三张源表的行数 == 对应 embedding 表按 model 过滤后的行数。
    #    按 model 过滤是关键：换模型后旧向量记录不会被误判为已完成。
    # 返回值：包含 by_document_kind（各表按 kind 分布）和 vectors（源表 vs embedding 行数）的字典。
    
    if model_name is None:
        model_name = DEFAULT_MODEL
    artifact_tables = ("documents", "document_cards", "document_views", "sections", "chunks")
    vector_targets = (
        ("document_cards", "document_card_embeddings"),
        ("document_views", "document_view_embeddings"),
        ("chunks", "chunk_embeddings"),
    )
    with PooledConn(get_pool(dsn)) as conn:
        with conn.cursor() as cur:
            by_kind = {table: helper_kind_counts(cur, table) for table in artifact_tables}
            for kind in ("guideline", "consensus"):
                document_count = by_kind["documents"].get(kind, 0)
                if document_count < 1:
                    raise RuntimeError(f"No {kind} documents were ingested")
                if by_kind["document_cards"].get(kind, 0) != document_count:
                    raise RuntimeError(f"{kind} document_cards do not match documents")
                for table in ("document_views", "sections", "chunks"):
                    if by_kind[table].get(kind, 0) < 1:
                        raise RuntimeError(f"No {kind} rows were ingested into {table}")

            vectors: dict[str, dict[str, int]] = {}
            for source_table, embedding_table in vector_targets:
                cur.execute(f"SELECT count(*) FROM {source_table}")
                source_count = int(cur.fetchone()[0])
                # WHERE model=%s：避免把旧模型的向量记录误判为当前模型已完成。
                cur.execute(f"SELECT count(*) FROM {embedding_table} WHERE model=%s", (model_name,))
                embedding_count = int(cur.fetchone()[0])
                if source_count != embedding_count:
                    raise RuntimeError(
                        f"{embedding_table} is incomplete for {model_name}: "
                        f"expected {source_count}, found {embedding_count}"
                    )
                vectors[source_table] = {"source": source_count, "embeddings": embedding_count}
    return {"status": "ok", "model": model_name, "by_document_kind": by_kind, "vectors": vectors}


def helper_time_filter_sql(time_range: str | dict[str, str] | None, params: list[Any]) -> str:
    # 把 time_range（'YYYY-YYYY' / dict / 自由字符串）展开成 SQL 片段与 params。
    # 解析规则：
    # - dict：取 start/start_date 与 end/end_date；
    # - 'YYYY-YYYY' 或 'YYYY:YYYY'（含 - ~ : 分隔）：自动补齐为 '-01-01' / '-12-31'；
    # - 其它字符串：作为单边 start。
    # 返回形如 ' AND publication_date >= %s AND publication_date <= %s'。
    # 多个过滤组合时由调用方在 WHERE 后用 AND 直接拼接。
    
    if not time_range:
        return ""
    if isinstance(time_range, dict):
        start = time_range.get("start") or time_range.get("start_date")
        end = time_range.get("end") or time_range.get("end_date")
    else:
        match = re.match(r"^\s*(\d{4})(?:-\d{2}-\d{2})?\s*[-~:]\s*(\d{4})(?:-\d{2}-\d{2})?\s*$", str(time_range))
        start = f"{match.group(1)}-01-01" if match else str(time_range)
        end = f"{match.group(2)}-12-31" if match else None
    clauses = []
    if start:
        clauses.append("publication_date >= %s")
        params.append(start)
    if end:
        clauses.append("publication_date <= %s")
        params.append(end)
    return " AND " + " AND ".join(clauses) if clauses else ""


def helper_fuse_document_results(result_lists: list[list[dict[str, Any]]], topk: int) -> list[dict[str, Any]]:
    # 对多通道文档召回结果做 RRF 融合，取 topk。
    # RRF（Reciprocal Rank Fusion）公式：score = Σ 1/(60 + rank)
    # - k=60 是经验常数，平滑高分项的极端权重；
    # - 多通道按 doc_id 求和，分数与通道数无关；
    # - 同分时按 doc_id 字典序稳定排序。
    # 关键点：
    # - by_id 取最先出现的项作为代表记录，避免被后续通道覆盖；
    # - channels 记录每个 doc_id 命中过的通道名（text_channel），便于溯源。

    scores: dict[str, float] = {}
    by_id: dict[str, dict[str, Any]] = {}
    channels: dict[str, set[str]] = {}
    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            doc_id = item["doc_id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (60 + rank)
            by_id.setdefault(doc_id, item)
            channel = item.get("text_channel")
            if channel:
                channels.setdefault(doc_id, set()).add(channel)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:topk]
    output = []
    for doc_id, score in ranked:
        item = dict(by_id[doc_id])
        item["score"] = score
        item["retrieval_channels"] = sorted(channels.get(doc_id, set()))
        output.append(item)
    return output


def search_document_cards_pg(
    query: str,
    dsn: str | None = None,
    source_institution: str | None = None,
    clinical_department: str | None = None,
    time_range: str | dict[str, str] | None = None,
    publication_date: str | None = None,
    topk: int = 50,
    document_kind: str | None = None,
) -> list[dict[str, Any]]:
    # 基于全文索引的卡片级 BM25-like 检索。
    # 排序策略：score DESC, publication_date DESC —— 同分时优先最新发布的指南。
    # ILIKE 模糊匹配 source_institution / clinical_department 配合 GIN-trgm 索引加速。
    
    params: list[Any] = [query, query]
    where = ["websearch_to_tsquery('simple', %s) @@ dc.content_tsv"]
    if source_institution:
        where.append("dc.source_institution ILIKE %s")
        params.append(f"%{source_institution}%")
    if clinical_department:
        where.append("dc.clinical_department ILIKE %s")
        params.append(f"%{clinical_department}%")
    if document_kind:
        where.append("d.document_kind = %s")
        params.append(document_kind)
    if publication_date:
        where.append("dc.publication_date = %s")
        params.append(publication_date)
    time_sql = helper_time_filter_sql(time_range, params).replace("publication_date", "dc.publication_date")
    sql = f"""
        SELECT dc.doc_id, dc.title, d.abstract, dc.publication_date, dc.source_institution, dc.clinical_department,
               ts_rank_cd(dc.content_tsv, websearch_to_tsquery('simple', %s)) AS score
        FROM document_cards dc
        JOIN documents d ON d.doc_id = dc.doc_id
        WHERE {' AND '.join(where)} {time_sql}
        ORDER BY score DESC, dc.publication_date DESC
        LIMIT %s
    """
    params.append(topk)
    with PooledConn(get_pool(dsn)) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    return [
        {
            "doc_id": row[0],
            "title": row[1],
            "abstract": row[2],
            "publication_date": row[3],
            "source_institution": row[4],
            "clinical_department": row[5],
            "score": float(row[6] or 0.0),
            "text_channel": "document_card_text",
        }
        for row in rows
    ]


def search_document_views_pg(
    query: str,
    dsn: str | None = None,
    source_institution: str | None = None,
    clinical_department: str | None = None,
    time_range: str | dict[str, str] | None = None,
    publication_date: str | None = None,
    topk: int = 50,
    document_kind: str | None = None,
) -> list[dict[str, Any]]:
    # 基于全文索引的视图级 BM25-like 检索。
    # 排序：score DESC, priority DESC, publication_date DESC
    # - priority 来自 builder，重要视图（recommendation_summary=1.35）排在前面；
    # - 同一文档可能有多个视图，先扩大候选池（max(topk*3, topk)），
    #   再在 Python 端按 doc_id 去重至 topk，避免视图数挤占文档数。
    
    params: list[Any] = [query, query]
    where = ["websearch_to_tsquery('simple', %s) @@ v.content_tsv"]
    if source_institution:
        where.append("v.source_institution ILIKE %s")
        params.append(f"%{source_institution}%")
    if clinical_department:
        where.append("v.clinical_department ILIKE %s")
        params.append(f"%{clinical_department}%")
    if document_kind:
        where.append("d.document_kind = %s")
        params.append(document_kind)
    if publication_date:
        where.append("v.publication_date = %s")
        params.append(publication_date)
    time_sql = helper_time_filter_sql(time_range, params).replace("publication_date", "v.publication_date")
    sql = f"""
        SELECT v.doc_id, v.title, d.abstract, v.publication_date, v.source_institution, v.clinical_department,
               v.view_id, v.view_type, ts_rank_cd(v.content_tsv, websearch_to_tsquery('simple', %s)) AS score
        FROM document_views v
        JOIN documents d ON d.doc_id = v.doc_id
        WHERE {' AND '.join(where)} {time_sql}
        ORDER BY score DESC, v.priority DESC, v.publication_date DESC
        LIMIT %s
    """
    params.append(max(topk * 3, topk))
    with PooledConn(get_pool(dsn)) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    results = []
    seen: set[str] = set()
    for row in rows:
        if row[0] in seen:
            continue
        seen.add(row[0])
        results.append(
            {
                "doc_id": row[0],
                "title": row[1],
                "abstract": row[2],
                "publication_date": row[3],
                "source_institution": row[4],
                "clinical_department": row[5],
                "view_id": row[6],
                "view_type": row[7],
                "score": float(row[8] or 0.0),
                "text_channel": "document_view_text",
            }
        )
        if len(results) >= topk:
            break
    return results


def search_documents_pg(query: str,dsn: str, source_institution: str, clinical_department: str, time_range: str | dict[str, str] | None = None, publication_date: str | None = None, topk: int = 10,) -> list[dict[str, Any]]:
    # 文档级融合检索：cards 通道 + views 通道 → RRF 融合 → topk。
    # pool_size = max(50, topk)：单通道召回至少 50 个候选，留足融合空间。
    # 实际调用方通常使用 search_documents_hybrid_pg（混合 BM25 + 向量）。
    
    pool_size = max(50, topk)
    cards = search_document_cards_pg(query, dsn, source_institution, clinical_department, time_range, publication_date, pool_size)
    views = search_document_views_pg(query, dsn, source_institution, clinical_department, time_range, publication_date, pool_size)
    return helper_fuse_document_results([cards, views], topk)


def retrieve_chunks_pg(
    query: str,
    dsn: str | None = None,
    source_institution: str | None = None,
    clinical_department: str | None = None,
    time_range: str | dict[str, str] | None = None,
    publication_date: str | None = None,
    topk: int = 5,
    document_kind: str | None = None,
) -> list[dict[str, Any]]:
    # 基于全文索引的分块级 BM25-like 检索。
    # SQL 层硬性排除 is_reference_section=true 的块（参考文献列表对召回噪声大）；
    # 返回字段保留全部追踪字段（section_path / chunk_type / source_file 等），供 MCP 返回原文引用。
    
    params: list[Any] = [query, query]
    where = ["websearch_to_tsquery('simple', %s) @@ c.content_tsv", "c.is_reference_section = false"]
    if source_institution:
        where.append("c.source_institution ILIKE %s")
        params.append(f"%{source_institution}%")
    if clinical_department:
        where.append("c.clinical_department ILIKE %s")
        params.append(f"%{clinical_department}%")
    if document_kind:
        where.append("d.document_kind = %s")
        params.append(document_kind)
    if publication_date:
        where.append("c.publication_date = %s")
        params.append(publication_date)
    time_sql = helper_time_filter_sql(time_range, params).replace("publication_date", "c.publication_date")
    sql = f"""
        SELECT c.chunk_id, c.doc_id, c.content, c.title, c.publication_date, c.source_institution, c.clinical_department, c.section_path,
               c.chunk_index, c.source_file, c.markdown_clean_path, c.chunk_type, c.retrieval_text,
               ts_rank_cd(c.content_tsv, websearch_to_tsquery('simple', %s)) AS score,
               c.retrieval_key
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id
        WHERE {' AND '.join(where)} {time_sql}
        ORDER BY score DESC
        LIMIT %s
    """
    params.append(topk)
    with PooledConn(get_pool(dsn)) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    return [
        {
            "chunk_id": row[0],
            "doc_id": row[1],
            "content": row[2],
            "title": row[3],
            "publication_date": row[4],
            "source_institution": row[5],
            "clinical_department": row[6],
            "section_path": row[7],
            "chunk_index": row[8],
            "source_file": row[9],
            "markdown_clean_path": row[10],
            "chunk_type": row[11],
            "retrieval_text": row[12],
            "score": float(row[13] or 0.0),
            "retrieval_key": row[14],
        }
        for row in rows
    ]


def read_document_pg(doc_id: str | None = None,dsn: str | None = None,title: str | None = None,max_chars: int | None = None,) -> dict[str, Any]:
    # 按 doc_id 或 title 读取单篇文档正文。
    # - 必须传 doc_id 或 title 之一，否则 ValueError；
    # - title 模式优先精确匹配，再退到 ILIKE '%title%' 模糊匹配；
    #   排序优先级：精确匹配 > publication_date DESC > 标题长度 ASC，
    #   保证选择最具体且最新的文档。
    # - max_chars > 0 时截断 content，避免返回超长文档撑爆 MCP 响应。
    
    if not doc_id and not title:
        raise ValueError("read_document_pg requires either doc_id or title")
    with PooledConn(get_pool(dsn)) as conn:
        with conn.cursor() as cur:
            if doc_id:
                cur.execute(
                    """
                    SELECT doc_id, title, publication_date, source_institution, clinical_department, document_kind,
                           source_file, markdown_clean_path, content_md
                    FROM documents
                    WHERE doc_id=%s
                    """,
                    (doc_id,),
                )
            else:
                query_title = (title or "").strip()
                cur.execute(
                    """
                    SELECT doc_id, title, publication_date, source_institution, clinical_department, document_kind,
                           source_file, markdown_clean_path, content_md
                    FROM documents
                    WHERE title = %s OR title ILIKE %s
                    ORDER BY CASE WHEN title = %s THEN 0 ELSE 1 END, publication_date DESC, length(title) ASC
                    LIMIT 1
                    """,
                    (query_title, f"%{query_title}%", query_title),
                )
            row = cur.fetchone()
    if not row:
        raise KeyError(f"Document not found: {doc_id or title}")
    content = row[8]
    if max_chars and max_chars > 0:
        content = content[:max_chars]
    return {
        "doc_id": row[0],
        "title": row[1],
        "publication_date": row[2],
        "source_institution": row[3],
        "clinical_department": row[4],
        "document_kind": row[5],
        "source_file": row[6],
        "markdown_clean_path": row[7],
        "content": content,
    }


def main() -> None:
    """CLI 入口: python -m src.storage.postgres_store <command> [flags]。

    子命令一览：
      init              建主表（可选 --with-vector 加 pgvector 与三张 embedding 表）
      reset             DROP 所有 9 张表后 init（重置最干净的方式）
      ingest            全量/增量入库 5 张主表（--incremental 进入增量模式）
      ingest-sections   仅入库 sections 表（ON CONFLICT DO NOTHING，可重复执行）
      index-vectors     建 IVFFlat 向量索引（必须在所有 embedding 写入后）
      stats             打印行数 / 按 document_kind 分布 / embedding 表状态
      verify            校验主表与 embedding 表行数一致（按 model 过滤），失败抛 RuntimeError

    注意: reset 会删除所有数据，调用前确认已经备份或接受数据丢失。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["init", "reset", "ingest", "ingest-sections", "index-vectors", "stats", "verify"])
    parser.add_argument("--dsn")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--with-vector", action="store_true")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="For ingest, append only new doc_ids and leave existing documents unchanged.",
    )
    parser.add_argument("--model", default=os.getenv("PG_VECTOR_MODEL", DEFAULT_MODEL))
    args = parser.parse_args()
    if args.command == "init":
        payload = init_schema(args.dsn, args.with_vector)
    elif args.command == "reset":
        payload = reset_schema(args.dsn, args.with_vector)
    elif args.command == "ingest":
        payload = ingest_data(args.dsn, args.data_dir, args.batch_size, args.incremental)
    elif args.command == "ingest-sections":
        payload = ingest_sections(args.dsn, args.data_dir, args.batch_size)
    elif args.command == "index-vectors":
        payload = create_vector_indexes(args.dsn)
    elif args.command == "verify":
        payload = verify_retrieval_snapshot(args.dsn, args.model)
    else:
        payload = database_stats(args.dsn)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()