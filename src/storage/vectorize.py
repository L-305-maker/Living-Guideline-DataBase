# 将 document_cards / document_views / chunks 三类内容向量化为 pgvector。
#
# 关键设计：
# 1. 同一文档可对应多个模型：embedding 表的主键是 (业务ID, model)，
#    换模型不需要先清表，旧向量会原样保留，新向量按新 model 追加。
# 2. 默认 missing_only=True（只补缺失）：利用 LEFT JOIN ... IS NULL 探测缺失向量，
#    避免大规模重算。强制重算请使用 --include-existing。
# 3. 分页读取 + 批量编码：page_size 控制单次从 PG 拉取的行数，
#    batch_size 控制单次送入 GPU 的样本数；二者解耦避免显存峰值。
# 4. UPSERT 而非纯 INSERT：(业务ID, model) 冲突时直接更新 dim/embedding/created_at，
#    保证补缺失时不会因部分记录已存在而中断。

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

from src.storage.postgres_store import get_pool
from src.storage.query_embedding import (
    DEFAULT_MODEL,
    EMBEDDING_DIM,
    encode_with_model as _encode_with_model,
    load_model as helper_load_model,
    vector_literal as _vector_literal,
)


def ensure_vector_schema(dsn: str | None = None) -> None:
    """幂等创建 3 张 embedding 表与 pgvector 扩展（若不存在则建）。

    使用 CREATE EXTENSION / CREATE TABLE IF NOT EXISTS，可重复执行。
    幂等性保证多进程或多轮调用不会因重复建表失败。
    """
    from src.storage.postgres_store import VECTOR_SCHEMA_SQL

    with get_pool(dsn).connection(timeout=10) as conn:
        with conn.cursor() as cur:
            cur.execute(VECTOR_SCHEMA_SQL)
        conn.commit()


@dataclass(frozen=True)
class VectorTarget:
    source_table: str
    alias: str
    id_column: str
    text_expression: str
    embedding_table: str


DOCUMENT_CARDS = VectorTarget(
    "document_cards", "dc", "doc_id", "dc.card_text", "document_card_embeddings"
)
DOCUMENT_VIEWS = VectorTarget(
    "document_views", "v", "view_id", "v.text", "document_view_embeddings"
)
CHUNKS = VectorTarget(
    "chunks",
    "c",
    "chunk_id",
    "coalesce(nullif(c.text_for_embedding, ''), nullif(c.retrieval_text, ''), c.content)",
    "chunk_embeddings",
)


def helper_fetch_rows(
    target: VectorTarget,
    dsn: str | None,
    limit: int | None,
    offset: int,
    missing_only: bool,
    model_name: str,
    max_text_chars: int | None = None,
) -> list[tuple[str, str]]:
    """按受信任的目标配置读取一页待向量化文本。"""
    content = target.text_expression
    params: list[Any] = []
    if max_text_chars is not None:
        content = f"left({content}, %s)"
        params.append(max_text_chars)
    sql = f"SELECT {target.alias}.{target.id_column}, {content} AS content FROM {target.source_table} {target.alias}"
    if missing_only:
        sql += (
            f" LEFT JOIN {target.embedding_table} e"
            f" ON e.{target.id_column}={target.alias}.{target.id_column} AND e.model=%s"
            f" WHERE e.{target.id_column} IS NULL"
        )
        params.append(model_name)
    sql += f" ORDER BY {target.alias}.{target.id_column} OFFSET %s"
    params.append(offset)
    if limit:
        sql += " LIMIT %s"
        params.append(limit)
    with get_pool(dsn).connection(timeout=10) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def helper_vectorize_rows(
    rows: list[tuple[str, str]],
    dsn: str | None,
    model_name: str,
    batch_size: int,
    target: VectorTarget,
    model: Any | None = None,
) -> dict[str, Any]:
    """把已读入内存的行批量编码并 UPSERT 到指定 embedding 表。

    流程：
    1. 模型按需懒加载（仅首次调用或显式传入 None 时加载），打印 device 便于调试。
    2. 按 batch_size 切片送入 model.encode；批量推理比逐条调用节省大量 CPU/GPU 调度开销。
    3. 编码结果必须与数据库 vector 列的固定维度一致。
    4. 使用 executemany + ON CONFLICT (id_column, model) DO UPDATE：
       - 同一 (id, model) 已存在时刷新 dim/embedding/created_at；
       - 不存在时正常 INSERT；
       - 关键设计：换模型时旧记录自然不被更新，新记录按新 model 写入。
    5. 每批 commit 一次，避免长事务膨胀 WAL 并允许中途崩溃后断点续跑。
    """
    print(f"[vectorize] selected {len(rows)} {target.source_table}", flush=True)
    if model is None:
        print(f"[vectorize] loading model {model_name}...", flush=True)
        model = helper_load_model(model_name)
        print(f"[vectorize] model loaded on {getattr(model, 'device', 'unknown')}", flush=True)
    inserted = 0
    with get_pool(dsn).connection(timeout=10) as conn:
        with conn.cursor() as cur:
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                texts = [row[1] for row in batch]
                print(f"[vectorize] encoding {target.source_table} {start + 1}-{start + len(batch)}...", flush=True)
                embeddings = _encode_with_model(model, texts, model_name=model_name)
                dim = int(embeddings.shape[1])
                if dim != EMBEDDING_DIM:
                    raise ValueError(
                        f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {dim}"
                    )
                payload = [
                    (row[0], model_name, dim, _vector_literal(embedding))
                    for row, embedding in zip(batch, embeddings)
                ]
                cur.executemany(
                    f"""
                    INSERT INTO {target.embedding_table} ({target.id_column}, model, dim, embedding)
                    VALUES (%s, %s, %s, %s::vector)
                    ON CONFLICT ({target.id_column}, model) DO UPDATE SET
                        dim=EXCLUDED.dim,
                        embedding=EXCLUDED.embedding,
                        created_at=now()
                    """,
                    payload,
                )
                conn.commit()
                inserted += len(payload)
                print(f"[vectorize] upserted {target.source_table} {inserted}/{len(rows)}", flush=True)
    return {"kind": target.source_table, "model": model_name, "selected": len(rows), "upserted": inserted}


def helper_vectorize_pages(
    target: VectorTarget,
    dsn: str | None,
    model_name: str,
    batch_size: int,
    limit: int | None,
    offset: int,
    missing_only: bool,
    page_size: int,
    max_text_chars: int | None = None,
) -> dict[str, Any]:
    """分页驱动 helper_vectorize_rows，覆盖大表向量化场景。

    关键设计：
    - 模型只在第一页加载一次（model is None 判断），之后整轮复用同一 SentenceTransformer
      实例，避免每页重复加载权重（加载一次需要几秒到几十秒）。
    - missing_only=True 时：分页使用固定 offset（缺失集合稳定）；
      missing_only=False 时：每页推进 fetch_offset，否则会无限循环处理同一批。
    - 提前退出条件：当一页返回行数 < fetch_limit（已读完）或已达 limit 上限。
    - 参数边界：batch_size / page_size 必须正数；limit 正数才合法。
    """
    if batch_size < 1 or page_size < 1:
        raise ValueError("batch_size and page_size must be positive")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    total = 0
    fetch_offset = offset
    model = None
    while limit is None or total < limit:
        fetch_limit = page_size if limit is None else min(page_size, limit - total)
        # missing_only 模式：每页都从同一 offset 开始查剩余缺失项；
        # 否则：每页推进 OFFSET 顺序扫描全表。
        page_offset = offset if missing_only else fetch_offset
        rows = helper_fetch_rows(
            target, dsn, fetch_limit, page_offset, missing_only, model_name, max_text_chars
        )
        if not rows:
            break
        if model is None:
            print(f"[vectorize] loading model {model_name}...", flush=True)
            model = helper_load_model(model_name)
            print(f"[vectorize] model loaded on {getattr(model, 'device', 'unknown')}", flush=True)
        result = helper_vectorize_rows(rows, dsn, model_name, batch_size, target, model)
        total += int(result["upserted"])
        if not missing_only:
            fetch_offset += len(rows)
        if len(rows) < fetch_limit:
            break
    return {"kind": target.source_table, "model": model_name, "selected": total, "upserted": total}


def vectorize_document_cards(
    dsn: str | None = None,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 16,
    limit: int | None = None,
    offset: int = 0,
    missing_only: bool = True,
    skip_schema: bool = False,
    max_text_chars: int = 12000,
    page_size: int = 2000,
) -> dict[str, Any]:
    """向量化 document_cards 表（默认参数：批次 16，文本上限 12000 字符，分页 2000）。

    跳过 schema 创建：通常用于多表共享一次 ensure_vector_schema 的场景，避免重复 DDL。
    """
    if not skip_schema:
        ensure_vector_schema(dsn)
    return helper_vectorize_pages(
        DOCUMENT_CARDS,
        dsn,
        model_name,
        batch_size,
        limit,
        offset,
        missing_only,
        page_size,
        max_text_chars,
    )


def vectorize_document_views(
    dsn: str | None = None,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 16,
    limit: int | None = None,
    offset: int = 0,
    missing_only: bool = True,
    skip_schema: bool = False,
    max_text_chars: int = 8000,
    page_size: int = 2000,
) -> dict[str, Any]:
    """向量化 document_views 表（视图文本通常比卡片短，默认上限 8000 字符）。"""
    if not skip_schema:
        ensure_vector_schema(dsn)
    return helper_vectorize_pages(
        DOCUMENT_VIEWS,
        dsn,
        model_name,
        batch_size,
        limit,
        offset,
        missing_only,
        page_size,
        max_text_chars,
    )


def vectorize_chunks(
    dsn: str | None = None,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    limit: int | None = None,
    offset: int = 0,
    missing_only: bool = True,
    skip_schema: bool = False,
    page_size: int = 2000,
) -> dict[str, Any]:
    """向量化 chunks 表（默认批次 32，文本由 chunk_normalizer 收敛到 text_for_embedding）。

    批次比 cards/views 大（32 vs 16）：chunk 文本普遍较短，单位 token 计算更密集。
    """
    if not skip_schema:
        ensure_vector_schema(dsn)
    return helper_vectorize_pages(
        CHUNKS,
        dsn,
        model_name,
        batch_size,
        limit,
        offset,
        missing_only,
        page_size,
    )


def main() -> None:
    """CLI 入口：python -m src.storage.vectorize <target> [flags]。

    target 必填，三选一：
      document_cards / document_views / chunks

    常用 flag：
      --include-existing   强制重算所有向量（覆盖已有记录，UPSERT 行为）
      --limit / --offset   分批执行，方便长任务中断续跑
      --batch-size         GPU 批次大小；显存不足时减小
      --page-size          单次从 PG 拉取的源表行数；与 batch_size 解耦
      --skip-schema        跳过 ensure_vector_schema（多表共享场景）
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=["document_cards", "document_views", "chunks"])
    parser.add_argument("--dsn")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--page-size", type=int, default=2000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--include-existing", action="store_true")
    parser.add_argument("--skip-schema", action="store_true")
    parser.add_argument("--max-text-chars", type=int)
    args = parser.parse_args()
    common = {
        "dsn": args.dsn,
        "model_name": args.model,
        "batch_size": args.batch_size,
        "page_size": args.page_size,
        "limit": args.limit,
        "offset": args.offset,
        "missing_only": not args.include_existing,
        "skip_schema": args.skip_schema,
    }
    if args.target == "document_cards":
        if args.max_text_chars is not None:
            common["max_text_chars"] = args.max_text_chars
        payload = vectorize_document_cards(**common)
    elif args.target == "document_views":
        if args.max_text_chars is not None:
            common["max_text_chars"] = args.max_text_chars
        payload = vectorize_document_views(**common)
    else:
        payload = vectorize_chunks(**common)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
