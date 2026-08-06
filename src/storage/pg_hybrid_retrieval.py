# PostgreSQL 混合检索：全文召回 + 向量召回 + RRF 融合 + Reranker 重排。
#
# 检索通道（4 路）：
#   1. document_cards 全文（BM25-like）
#   2. document_views 全文（BM25-like，多视图合并）
#   3. document_cards 向量（pgvector 余弦距离）
#   4. document_views 向量（pgvector 余弦距离）
# 对分块检索只用 (1)+(3) 两路（无 view 概念）。
#
# 关键设计：
# - RRF（Reciprocal Rank Fusion）融合：score = Σ 1/(60 + rank)，
#   避免不同通道的原始分数（余弦距离 vs ts_rank）量纲不一致的问题。
# - 向量通道可选降级：当 PG_VECTOR_RETRIEVAL_REQUIRED=1 时强制要求向量就绪，
#   否则允许任意通道失败并继续走纯文本通道。
# - consensus 兜底：检索结果不足 topk 时，从 document_kind=consensus 集合补充。
# - 临床科室配额：单科室/多科室按 9:1 比例补充，控制路由倾斜。

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from collections.abc import Callable
from typing import Any

from src.retrieval.reranker import ChunkReranker, DocumentReranker, default_chunk_reranker, default_document_reranker
from src.retrieval.common import (
    fill_consensus_fallback as helper_fill_consensus_fallback,
    source_quote_context as helper_source_quote_context,
)
from src.retrieval.rrf import rrf_fusion
from src.storage.query_embedding import DEFAULT_MODEL, query_vector_literal
from src.storage.postgres_store import (
    helper_time_filter_sql,
    get_pool,
    PooledConn,
    retrieve_chunks_pg,
    search_document_cards_pg,
    search_document_views_pg,
)



LOGGER = logging.getLogger(__name__)


def helper_query_vector(query: str, model_name: str) -> str:
    # 对查询文本做向量化并返回 pgvector 字面量字符串（SQL 直传）。
    # 转发到 query_embedding.query_vector_literal，保证与数据库向量化使用同一模型与归一化设置。

    return query_vector_literal(query, model_name)


def pg_vector_retrieval_required() -> bool:
    # 读取环境变量 PG_VECTOR_RETRIEVAL_REQUIRED；为真时强制向量通道不可缺失。
    # 业务意义：开启后若向量缺失或查询异常，检索直接抛错而不是悄悄降级，
    # 避免"声称混合检索但实际只有 BM25"的隐性回退。
    
    return os.getenv("PG_VECTOR_RETRIEVAL_REQUIRED", "0").strip().lower() in {"1", "true", "yes", "y"}


@lru_cache(maxsize=4)
def helper_assert_pg_vectors_ready(dsn: str | None, model_name: str) -> None:
    # 校验 3 张源表的行数 == 对应 embedding 表按 model 过滤的行数，否则抛错。
    # 用 (dsn, model_name) 作缓存键：同一连接同一模型只校验一次，避免每次请求都跑 SQL。
    # 仅校验文档 cards / views / chunks 三层；不涉及 sections 与主 documents 表。
    
    targets = (
        ("document_cards", "document_card_embeddings"),
        ("document_views", "document_view_embeddings"),
        ("chunks", "chunk_embeddings"),
    )
    with PooledConn(get_pool(dsn)) as conn:
        with conn.cursor() as cur:
            for source_table, embedding_table in targets:
                # 源表条数是当前数据快照应具备的向量基线。
                cur.execute(f"SELECT count(*) FROM {source_table}")
                source_count = int(cur.fetchone()[0])
                # 按模型过滤向量条数：防止把旧模型生成的向量误判为已完成。
                cur.execute(f"SELECT count(*) FROM {embedding_table} WHERE model = %s", (model_name,))
                embedding_count = int(cur.fetchone()[0])
                if source_count != embedding_count:
                    raise RuntimeError(
                        f"pgvector is not ready for {source_table}: expected {source_count} embeddings for "
                        f"model {model_name!r}, found {embedding_count}. Run the matching "
                        "src.storage.bge_m3_vectorize command, then postgres_store index-vectors."
                    )


def vector_search_document_cards_pg(
    query: str,
    dsn: str | None = None,
    source_institution: str | None = None,
    clinical_department: str | None = None,
    time_range: str | dict[str, str] | None = None,
    publication_date: str | None = None,
    topk: int = 50,
    model_name: str = DEFAULT_MODEL,
    document_kind: str | None = None,
) -> list[dict[str, Any]]:
    # 向量通道：document_card_embeddings 上的余弦相似度检索。
    # - 余弦距离 <=> 越小越相似；1 - distance 转为相似度，可与其它通道统一按降序解释；
    # - 必须 WHERE e.model = %s 保证只查当前模型（兼容历史模型记录）；
    # - time_sql 把 publication_date 改成 dc.publication_date，消除 cards JOIN documents 后的列歧义。
    
    vector = helper_query_vector(query, model_name)
    params: list[Any] = [vector, model_name]
    where = ["e.model = %s"]
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
               1 - (e.embedding <=> %s::vector) AS score
        FROM document_card_embeddings e
        JOIN document_cards dc ON dc.doc_id = e.doc_id
        JOIN documents d ON d.doc_id = dc.doc_id
        WHERE {' AND '.join(where)} {time_sql}
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
    """
    params.extend([vector, topk])
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
            "vector_channel": "document_card",
        }
        for row in rows
    ]


def vector_search_document_views_pg(
    query: str,
    dsn: str | None = None,
    source_institution: str | None = None,
    clinical_department: str | None = None,
    time_range: str | dict[str, str] | None = None,
    publication_date: str | None = None,
    topk: int = 50,
    model_name: str = DEFAULT_MODEL,
    document_kind: str | None = None,
) -> list[dict[str, Any]]:
    # 向量通道：document_view_embeddings 上的余弦相似度检索。
    # 与 cards 通道差异：
    # - 多取候选（max(topk*3, topk)），同一文档的多个视图都被召回后 Python 端按 doc_id 去重；
    #   不预先在 SQL 里 DISTINCT 是为了保留优先级和分数用于融合。

    vector = helper_query_vector(query, model_name)
    params: list[Any] = [vector, model_name]
    where = ["e.model = %s"]
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
               v.view_id, v.view_type, 1 - (e.embedding <=> %s::vector) AS score
        FROM document_view_embeddings e
        JOIN document_views v ON v.view_id = e.view_id
        JOIN documents d ON d.doc_id = v.doc_id
        WHERE {' AND '.join(where)} {time_sql}
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
    """
    # 同一文档可命中多个语义视图，先过召回再按 doc_id 去重，避免视图数挤占文档数。
    params.extend([vector, max(topk * 3, topk)])
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
                "vector_channel": "document_view",
            }
        )
        if len(results) >= topk:
            break
    return results


def vector_retrieve_chunks_pg(
    query: str,
    dsn: str | None = None,
    source_institution: str | None = None,
    clinical_department: str | None = None,
    time_range: str | dict[str, str] | None = None,
    publication_date: str | None = None,
    topk: int = 50,
    model_name: str = DEFAULT_MODEL,
    document_kind: str | None = None,
) -> list[dict[str, Any]]:
    # 向量通道：chunk_embeddings 上的余弦相似度分块检索。
    # 关键过滤：
    # - e.model = %s 锁定当前模型；
    # - c.is_reference_section = false 在 SQL 层排除参考文献（与 BM25 通道一致）。
    # 返回保留完整追踪字段（chunk_id / doc_id / section_path / chunk_type 等），
    # 便于后续 Reranker 与 MCP 返回引用。
    
    vector = helper_query_vector(query, model_name)
    params: list[Any] = [vector, model_name]
    where = ["e.model = %s", "c.is_reference_section = false"]
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
        SELECT c.chunk_id, c.doc_id, c.content, c.title, c.publication_date, c.source_institution,
               c.clinical_department, c.section_path, c.chunk_index, c.source_file, c.markdown_clean_path,
               c.chunk_type, c.retrieval_text,
               1 - (e.embedding <=> %s::vector) AS score, c.retrieval_key
        FROM chunk_embeddings e
        JOIN chunks c ON c.chunk_id = e.chunk_id
        JOIN documents d ON d.doc_id = c.doc_id
        WHERE {' AND '.join(where)} {time_sql}
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
    """
    params.extend([vector, topk])
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




def helper_section_path_items(section_path: Any) -> list[Any]:
    # 把 section_path 规整为 list。
    # 兼容三种来源：
    # - str（pgvector 入库前 JSON 字符串）：尝试 json.loads，失败时回退到 [str]；
    # - list：原样返回；
    # - 其它可迭代对象：list() 转换。
    
    if isinstance(section_path, str):
        try:
            value = json.loads(section_path)
        except json.JSONDecodeError:
            return [section_path]
        return value if isinstance(value, list) else [value]
    if isinstance(section_path, list):
        return section_path
    return list(section_path) if section_path else []


def helper_section_path_json(section_path: Any) -> str:
    # 把 section_path 序列化为紧凑 JSON 字符串（用于 PG jsonb 字段）。
    return json.dumps(helper_section_path_items(section_path), ensure_ascii=False, separators=(",", ":"))


def helper_enrich_chunk_context_pg(dsn: str | None, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # 给每个 chunk 补充其所属 section 的 heading 与 char_start/char_end。
    # 行为：
    # - 通过 doc_id + section_path 查找对应 section 行（按 section_index 取第一条）；
    # - 找不到时回退使用 section_path 最后一项作为 heading 占位；
    # - 同时填 source_quote_context（用于 MCP 显示引用上下文）；
    # - char_span_kind 标记 section/document 跨度，便于前端区分引用粒度。
    # 当前为 N+1 查询（每 chunk 一次），数据量大时可改为批量 IN 查询优化。
    
    if not items:
        return []
    enriched: list[dict[str, Any]] = []
    with PooledConn(get_pool(dsn)) as conn:
        with conn.cursor() as cur:
            for item in items:
                heading = None
                section_char_start = None
                section_char_end = None
                section_path_json = helper_section_path_json(item.get("section_path"))
                cur.execute(
                    """
                    SELECT heading, char_start, char_end
                    FROM sections
                    WHERE doc_id=%s AND section_path=%s::jsonb
                    ORDER BY section_index
                    LIMIT 1
                    """,
                    (item["doc_id"], section_path_json),
                )
                section = cur.fetchone()
                if section:
                    heading = section[0]
                    section_char_start = section[1]
                    section_char_end = section[2]

                section_path_items = helper_section_path_items(item.get("section_path"))
                updated = dict(item)
                updated["heading"] = heading or ((section_path_items or [None])[-1])
                updated["prev_chunk_id"] = None
                updated["next_chunk_id"] = None
                updated["char_start"] = section_char_start
                updated["char_end"] = section_char_end
                updated["char_span_kind"] = "section" if section_char_start is not None or section_char_end is not None else None
                updated["source_quote_context"] = helper_source_quote_context("", item.get("content", ""), "")
                enriched.append(updated)
    return enriched


def helper_department_fields(item: dict[str, Any]) -> dict[str, Any]:
    # 把 PG 中的 clinical_department 单字符串（含 '|' 分隔的多科室）拆为 list。
    # 例如：'呼吸内科|感染科' → clinical_departments=['呼吸内科','感染科']，
    # clinical_department='呼吸内科'（取第一个作为主科室），
    # department_scope='compositive'（>1 否则 'single'）。
    # 该字段在 MCP 返回前由 MCP 层统一处理；本函数先在检索阶段补齐，便于排序与配额。
    
    labels = [label for label in str(item.get("clinical_department") or "").split("|") if label] or ["未分类"]
    item = dict(item)
    item.update(
        clinical_department=labels[0],
        clinical_departments=labels,
        department_scope="compositive" if len(labels) > 1 else "single",
    )
    return item


def helper_route_documents_pg(items: list[dict[str, Any]], clinical_department: str | None) -> list[dict[str, Any]]:
    # 当用户传入 clinical_department 过滤时，按单科/多科配额重组候选。
    # 配额：45 个单科室 + 5 个多科室（共 50），剩余位置按原顺序补齐去重。
    # 这样既能保证相关性强的单科室文档优先，又保留少量多科室覆盖。
    # 不传临床科室过滤时直接取前 50，不做配额（避免过度限制无过滤场景）。
    
    if not clinical_department:
        return items[:50]
    single = [item for item in items if "|" not in str(item.get("clinical_department") or "")][:45]
    compositive = [item for item in items if "|" in str(item.get("clinical_department") or "")][:5]
    selected = single + compositive
    seen = {item["doc_id"] for item in selected}
    selected.extend(item for item in items if item["doc_id"] not in seen and len(selected) < 50)
    rank = {item["doc_id"]: index for index, item in enumerate(items)}
    return sorted(selected, key=lambda item: rank[item["doc_id"]])[:50]


def helper_route_chunks_pg(items: list[dict[str, Any]], dsn: str | None, clinical_department: str | None) -> list[dict[str, Any]]:
    # 与 helper_route_documents_pg 类似但作用在 chunk 上。
    # 配额：91 个单科室 + 9 个多科室（共 100）。
    # 单/多科室需要查 documents 表（因为 chunks 表只有单数字段），先批量取一次避免 N+1。
    
    if not clinical_department:
        return items[:100]
    doc_ids = list(dict.fromkeys(item["doc_id"] for item in items))
    scopes: dict[str, str] = {}
    if doc_ids:
        with PooledConn(get_pool(dsn)) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT doc_id, clinical_department FROM documents WHERE doc_id = ANY(%s)", (doc_ids,))
                scopes = {row[0]: ("compositive" if "|" in str(row[1] or "") else "single") for row in cur.fetchall()}
    single = [item for item in items if scopes.get(item["doc_id"], "single") == "single"][:91]
    compositive = [item for item in items if scopes.get(item["doc_id"]) == "compositive"][:9]
    selected = single + compositive
    seen = {item["chunk_id"] for item in selected}
    selected.extend(item for item in items if item["chunk_id"] not in seen and len(selected) < 100)
    rank = {item["chunk_id"]: index for index, item in enumerate(items)}
    return sorted(selected, key=lambda item: rank[item["chunk_id"]])[:100]


def helper_optional_vector_channel(
    channel: str,
    required: bool,
    search: Callable[..., list[dict[str, Any]]],
    *args: Any,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    # 执行可选向量通道；按 required 决定失败时是抛错还是降级为空结果。
    # 设计动机：4 个向量通道分别捕获 cards / views / chunks 的异常，集中处理避免在
    # search_*_hybrid_pg 内重复 try/except。required 为真时（PG_VECTOR_RETRIEVAL_REQUIRED=1）
    # 异常直接上抛，符合"强制向量就绪"的业务诉求；否则记 warning 后返回空列表，
    # 由后续 RRF 融合自然回退为纯文本检索。
    
    try:
        return search(*args, **kwargs)
    except Exception:
        if required:
            raise
        LOGGER.warning("向量检索通道 %s 不可用，已降级为文本检索", channel, exc_info=True)
        return []


def search_documents_hybrid_pg(
    query: str, dsn: str | None = None, source_institution: str | None = None,
    clinical_department: str | None = None, time_range: str | dict[str, str] | None = None,
    publication_date: str | None = None, topk: int = 20, pool_size: int = 50,
    model_name: str = DEFAULT_MODEL, reranker: DocumentReranker | None = None, document_kind: str = "guideline",
) -> list[dict[str, Any]]:
    # 文档级混合检索：BM25 + Dense 双通道 → RRF → Reranker → topk。
    # 流程：
    # 1. 若 PG_VECTOR_RETRIEVAL_REQUIRED=1，先校验 3 张 embedding 表与源表行数一致；
    # 2. 四路召回（card_text / view_text / card_dense / view_dense）各取 pool_size 条；
    # 3. 仅保留非空通道做 RRF（空通道不影响融合分数）；
    # 4. 临床科室配额（helper_route_documents_pg）筛选出至多 50 个候选；
    # 5. 默认 DocumentReranker 重排至 topk（封顶 20）。
    
    require_vector = pg_vector_retrieval_required()
    if require_vector:
        helper_assert_pg_vectors_ready(dsn, model_name)
    if pool_size < 1:
        raise ValueError("pool_size must be positive")
    recall_n = max(pool_size, topk)
    card_text = search_document_cards_pg(query, dsn, source_institution, clinical_department, time_range, publication_date, recall_n, document_kind=document_kind)
    view_text = search_document_views_pg(query, dsn, source_institution, clinical_department, time_range, publication_date, recall_n, document_kind=document_kind)
    card_dense = helper_optional_vector_channel(
        "document_cards",
        require_vector,
        vector_search_document_cards_pg,
        query, dsn, source_institution, clinical_department, time_range, publication_date,
        recall_n, model_name, document_kind=document_kind,
    )
    view_dense = helper_optional_vector_channel(
        "document_views",
        require_vector,
        vector_search_document_views_pg,
        query, dsn, source_institution, clinical_department, time_range, publication_date,
        recall_n, model_name, document_kind=document_kind,
    )
    # RRF 只接收非空通道；空通道不应占用名次权重或改变融合分母。
    channels = [ranked for ranked in (card_text, view_text, card_dense, view_dense) if ranked]
    if not channels:
        return []
    by_id = {item["doc_id"]: item for ranked in channels for item in ranked}
    rank_maps = {
        "card_bm25_rank": {row["doc_id"]: i for i, row in enumerate(card_text, 1)},
        "view_bm25_rank": {row["doc_id"]: i for i, row in enumerate(view_text, 1)},
        "card_dense_rank": {row["doc_id"]: i for i, row in enumerate(card_dense, 1)},
        "view_dense_rank": {row["doc_id"]: i for i, row in enumerate(view_dense, 1)},
    }
    candidates = []
    for doc_id, score in rrf_fusion([[item["doc_id"] for item in ranked] for ranked in channels]):
        item = dict(by_id[doc_id])
        item["score"] = score
        item["retrieval_scores"] = {name: ranks.get(doc_id) for name, ranks in rank_maps.items()}
        candidates.append(item)
    candidates = [{**helper_department_fields(item), "document_kind": document_kind} for item in helper_route_documents_pg(candidates, clinical_department)]
    reranker = reranker or default_document_reranker()
    return reranker.rerank(query, candidates, min(topk, 20))


def retrieve_chunks_hybrid_pg(
    query: str, dsn: str | None = None, source_institution: str | None = None,
    clinical_department: str | None = None, time_range: str | dict[str, str] | None = None,
    publication_date: str | None = None, topk: int = 30, pool_size: int = 100,
    model_name: str = DEFAULT_MODEL, reranker: ChunkReranker | None = None, document_kind: str = "guideline",
) -> list[dict[str, Any]]:
    # 分块级混合检索：BM25 + Dense 双通道 → RRF → 路由 → 上下文增强 → Reranker → topk。
    # 流程：
    # 1. 强制向量就绪校验（可选）；
    # 2. 两路召回（text_ranked / vector_ranked）各取 pool_size；
    # 3. RRF 融合（按 chunk_id 而非 doc_id）；
    # 4. helper_route_chunks_pg 按 91/9 配额重组；
    # 5. helper_enrich_chunk_context_pg 补充 section heading 与 char_span；
    # 6. 默认 ChunkReranker 重排至 topk（封顶 30）。
    
    require_vector = pg_vector_retrieval_required()
    if require_vector:
        helper_assert_pg_vectors_ready(dsn, model_name)
    if pool_size < 1:
        raise ValueError("pool_size must be positive")
    recall_n = max(pool_size, topk)
    text_ranked = retrieve_chunks_pg(query, dsn, source_institution, clinical_department, time_range, publication_date, recall_n, document_kind=document_kind)
    vector_ranked = helper_optional_vector_channel(
        "chunks",
        require_vector,
        vector_retrieve_chunks_pg,
        query, dsn, source_institution, clinical_department, time_range, publication_date,
        recall_n, model_name, document_kind=document_kind,
    )
    # 分块阶段同样融合词法与向量名次，而不是直接比较量纲不同的原始分数。
    channels = [ranked for ranked in (text_ranked, vector_ranked) if ranked]
    if not channels:
        return []
    by_id = {item["chunk_id"]: item for ranked in channels for item in ranked}
    bm25_ranks = {row["chunk_id"]: i for i, row in enumerate(text_ranked, 1)}
    vector_ranks = {row["chunk_id"]: i for i, row in enumerate(vector_ranked, 1)}
    candidates = []
    for chunk_id, score in rrf_fusion([[item["chunk_id"] for item in ranked] for ranked in channels]):
        item = dict(by_id[chunk_id])
        item["score"] = score
        item["retrieval_scores"] = {"bm25_rank": bm25_ranks.get(chunk_id), "vector_rank": vector_ranks.get(chunk_id)}
        candidates.append(item)
    candidates = [{**helper_department_fields(item), "document_kind": document_kind} for item in helper_route_chunks_pg(candidates, dsn, clinical_department)]
    candidates = helper_enrich_chunk_context_pg(dsn, candidates)
    reranker = reranker or default_chunk_reranker()
    return reranker.rerank(query, candidates, min(topk, 30))


def helper_attach_document_views_pg(
    items: list[dict[str, Any]], dsn: str | None = None,
) -> list[dict[str, Any]]:
    # 批量回填每个文档的 view_type → text 映射，供前端展示摘要/目录等。
    # - dict.fromkeys 保序去重：保持 item 中 doc_id 首次出现顺序；
    # - views_by_doc[view_type] 取首次出现的视图文本（同一 type 多个 view 时第一条优先）。
    
    if not items:
        return []
    doc_ids = list(dict.fromkeys(item["doc_id"] for item in items))
    views_by_doc: dict[str, dict[str, str]] = {doc_id: {} for doc_id in doc_ids}
    with PooledConn(get_pool(dsn)) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT doc_id, view_type, text
                FROM document_views
                WHERE doc_id = ANY(%s)
                ORDER BY doc_id, priority DESC, view_type
                """,
                (doc_ids,),
            )
            for doc_id, view_type, text in cur.fetchall():
                views_by_doc[doc_id].setdefault(view_type, text or "")
    return [
        {**item, "document_views": views_by_doc.get(item["doc_id"], {})}
        for item in items
    ]


def search_documents_with_consensus_fallback_pg(
    query: str, dsn: str | None = None, source_institution: str | None = None,
    clinical_department: str | None = None, time_range: str | dict[str, str] | None = None,
    publication_date: str | None = None, topk: int = 20, reranker: DocumentReranker | None = None,
) -> list[dict[str, Any]]:
    # 文档检索 + consensus 兜底：先用 guideline 集合，缺额时按缺口量补充 consensus。
    # wanted = min(topk, 20)：与 search_documents_hybrid_pg 的封顶 20 保持一致，
    # 避免 fallback 调用传入过大的 topk 导致 helper_fuse 截断后再被二次拉满。
    
    wanted = min(topk, 20)
    guidelines = search_documents_hybrid_pg(
        query, dsn, source_institution, clinical_department, time_range, publication_date,
        topk=wanted, reranker=reranker, document_kind="guideline",
    )
    output = helper_fill_consensus_fallback(
        guidelines,
        wanted,
        lambda deficit: search_documents_hybrid_pg(
            query, dsn, source_institution, clinical_department, time_range, publication_date,
            topk=deficit, reranker=reranker, document_kind="consensus",
        ),
    )
    return helper_attach_document_views_pg(output, dsn)


def retrieve_chunks_with_consensus_fallback_pg(
    query: str, dsn: str | None = None, source_institution: str | None = None,
    clinical_department: str | None = None, time_range: str | dict[str, str] | None = None,
    publication_date: str | None = None, topk: int = 30, reranker: ChunkReranker | None = None,
) -> list[dict[str, Any]]:
    # 分块检索 + consensus 兜底：先用 guideline 集合，缺额时按缺口量补充 consensus。
    # wanted = min(topk, 30)：与 retrieve_chunks_hybrid_pg 封顶一致。
    # 注意分块 consensus 兜底可能在补充出的 chunk 上下文信息（如 char_span）不完整，
    # 实际使用中 guideline 集合已能覆盖大部分医学问题。
    
    wanted = min(topk, 30)
    guidelines = retrieve_chunks_hybrid_pg(
        query, dsn, source_institution, clinical_department, time_range, publication_date,
        topk=wanted, reranker=reranker, document_kind="guideline",
    )
    return helper_fill_consensus_fallback(
        guidelines,
        wanted,
        lambda deficit: retrieve_chunks_hybrid_pg(
            query, dsn, source_institution, clinical_department, time_range, publication_date,
            topk=deficit, reranker=reranker, document_kind="consensus",
        ),
    )