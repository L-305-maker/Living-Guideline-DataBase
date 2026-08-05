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
from typing import Any

from src.storage.postgres_store import connect
from src.storage.query_embedding import (
    DEFAULT_MODEL,
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

    with connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(VECTOR_SCHEMA_SQL)
        conn.commit()


def helper_fetch_document_cards(
    dsn: str | None,
    limit: int | None,
    offset: int,
    missing_only: bool,
    model_name: str,
    max_text_chars: int,
) -> list[tuple[str, str]]:
    """从 document_cards 拉取待向量化的卡片文本。

    关键逻辑：
    - missing_only=True：通过 LEFT JOIN document_card_embeddings e 并过滤 e.doc_id IS NULL，
      实现"对当前 model 还没生成向量的卡片"过滤。注意 JOIN 条件里 e.model=%s 限定当前模型，
      否则会因旧模型已有向量而误判为已完成。
    - text 用 left(card_text, max_text_chars) 截断，避免超长卡片把 GPU 显存撑爆；
      cards 默认 max_text_chars=12000，与下方函数签名保持一致。
    - ORDER BY dc.doc_id OFFSET %s：稳定排序 + 显式 OFFSET，分页重试可幂等。
      如不加稳定排序，OFFSET 可能跳过或重复记录（重要）。
    """
    sql = """
        SELECT dc.doc_id, left(dc.card_text, %s) AS content
        FROM document_cards dc
    """
    params: list[Any] = [max_text_chars]
    if missing_only:
        sql += " LEFT JOIN document_card_embeddings e ON e.doc_id=dc.doc_id AND e.model=%s WHERE e.doc_id IS NULL"
        params.append(model_name)
    sql += " ORDER BY dc.doc_id OFFSET %s"
    params.append(offset)
    if limit:
        sql += " LIMIT %s"
        params.append(limit)
    with connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def helper_fetch_document_views(
    dsn: str | None,
    limit: int | None,
    offset: int,
    missing_only: bool,
    model_name: str,
    max_text_chars: int,
) -> list[tuple[str, str]]:
    """从 document_views 拉取待向量化的视图文本。

    与 helper_fetch_document_cards 几乎相同，差异：
    - 唯一键改用 view_id；
    - 默认 max_text_chars=8000（视图文本通常比卡片短）。
    """
    sql = """
        SELECT v.view_id, left(v.text, %s) AS content
        FROM document_views v
    """
    params: list[Any] = [max_text_chars]
    if missing_only:
        sql += " LEFT JOIN document_view_embeddings e ON e.view_id=v.view_id AND e.model=%s WHERE e.view_id IS NULL"
        params.append(model_name)
    sql += " ORDER BY v.view_id OFFSET %s"
    params.append(offset)
    if limit:
        sql += " LIMIT %s"
        params.append(limit)
    with connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def helper_fetch_chunks(dsn: str | None, limit: int | None, offset: int, missing_only: bool, model_name: str) -> list[tuple[str, str]]:
    """从 chunks 拉取待向量化的分块文本。

    文本优先级（重要）：
      text_for_embedding > retrieval_text > content
    三层 coalesce 是为了兼容不同阶段的 chunk 记录：
    - 新版 chunk 由 chunk_normalizer 写入 text_for_embedding 字段；
    - 早期版本只有 retrieval_text；
    - 极端情况下两者都为空则退回 content（保证至少有内容可编码）。
    """
    sql = """
        SELECT c.chunk_id, coalesce(nullif(c.text_for_embedding, ''), nullif(c.retrieval_text, ''), c.content) AS content
        FROM chunks c
    """
    params: list[Any] = []
    if missing_only:
        sql += " LEFT JOIN chunk_embeddings e ON e.chunk_id=c.chunk_id AND e.model=%s WHERE e.chunk_id IS NULL"
        params.append(model_name)
    sql += " ORDER BY c.chunk_id OFFSET %s"
    params.append(offset)
    if limit:
        sql += " LIMIT %s"
        params.append(limit)
    with connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def helper_vectorize_rows(
    rows: list[tuple[str, str]],
    dsn: str | None,
    model_name: str,
    batch_size: int,
    kind: str,
    table: str,
    id_column: str,
    model: Any | None = None,
) -> dict[str, Any]:
    """把已读入内存的行批量编码并 UPSERT 到指定 embedding 表。

    流程：
    1. 模型按需懒加载（仅首次调用或显式传入 None 时加载），打印 device 便于调试。
    2. 按 batch_size 切片送入 model.encode；批量推理比逐条调用节省大量 CPU/GPU 调度开销。
    3. payload 形如 [(id, model, dim, vector_literal), ...]，dim 从 embeddings.shape[1]
       提取，避免硬编码 1024；换 MRL 维度后无需修改本函数。
    4. 使用 executemany + ON CONFLICT (id_column, model) DO UPDATE：
       - 同一 (id, model) 已存在时刷新 dim/embedding/created_at；
       - 不存在时正常 INSERT；
       - 关键设计：换模型时旧记录自然不被更新，新记录按新 model 写入。
    5. 每批 commit 一次，避免长事务膨胀 WAL 并允许中途崩溃后断点续跑。
    """
    print(f"[vectorize] selected {len(rows)} {kind}", flush=True)
    if model is None:
        print(f"[vectorize] loading model {model_name}...", flush=True)
        model = helper_load_model(model_name)
        print(f"[vectorize] model loaded on {getattr(model, 'device', 'unknown')}", flush=True)
    inserted = 0
    with connect(dsn) as conn:
        with conn.cursor() as cur:
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                texts = [row[1] for row in batch]
                print(f"[vectorize] encoding {kind} {start + 1}-{start + len(batch)}...", flush=True)
                embeddings = _encode_with_model(model, texts, model_name=model_name)
                payload = [
                    (row[0], model_name, int(embeddings.shape[1]), _vector_literal(embedding))
                    for row, embedding in zip(batch, embeddings)
                ]
                cur.executemany(
                    f"""
                    INSERT INTO {table} ({id_column}, model, dim, embedding)
                    VALUES (%s, %s, %s, %s::vector)
                    ON CONFLICT ({id_column}, model) DO UPDATE SET
                        dim=EXCLUDED.dim,
                        embedding=EXCLUDED.embedding,
                        created_at=now()
                    """,
                    payload,
                )
                conn.commit()
                inserted += len(payload)
                print(f"[vectorize] upserted {kind} {inserted}/{len(rows)}", flush=True)
    return {"kind": kind, "model": model_name, "selected": len(rows), "upserted": inserted}


def helper_vectorize_pages(
    fetch_rows: Any,
    fetch_kwargs: dict[str, Any],
    dsn: str | None,
    model_name: str,
    batch_size: int,
    kind: str,
    table: str,
    id_column: str,
    limit: int | None,
    offset: int,
    missing_only: bool,
    page_size: int,
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
        rows = fetch_rows(dsn, fetch_limit, page_offset, missing_only, model_name, **fetch_kwargs)
        if not rows:
            break
        if model is None:
            print(f"[vectorize] loading model {model_name}...", flush=True)
            model = helper_load_model(model_name)
            print(f"[vectorize] model loaded on {getattr(model, 'device', 'unknown')}", flush=True)
        result = helper_vectorize_rows(rows, dsn, model_name, batch_size, kind, table, id_column, model)
        total += int(result["upserted"])
        if not missing_only:
            fetch_offset += len(rows)
        if len(rows) < fetch_limit:
            break
    return {"kind": kind, "model": model_name, "selected": total, "upserted": total}


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
        helper_fetch_document_cards,
        {"max_text_chars": max_text_chars},
        dsn,
        model_name,
        batch_size,
        "document_cards",
        "document_card_embeddings",
        "doc_id",
        limit,
        offset,
        missing_only,
        page_size,
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
        helper_fetch_document_views,
        {"max_text_chars": max_text_chars},
        dsn,
        model_name,
        batch_size,
        "document_views",
        "document_view_embeddings",
        "view_id",
        limit,
        offset,
        missing_only,
        page_size,
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
        helper_fetch_chunks,
        {},
        dsn,
        model_name,
        batch_size,
        "chunks",
        "chunk_embeddings",
        "chunk_id",
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