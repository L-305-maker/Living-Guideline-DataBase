"""Hybrid PostgreSQL retrieval with full-text rank plus BGE-M3/pgvector rank."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

from src.retrieval.reranker import ChunkReranker, DocumentReranker, default_chunk_reranker, default_document_reranker
from src.retrieval.rrf import rrf_fusion
from src.storage.query_embedding import DEFAULT_MODEL, query_vector_literal
from src.storage.postgres_store import (
    helper_time_filter_sql,
    connect,
    retrieve_chunks_pg,
    search_document_cards_pg,
    search_document_views_pg,
)


def helper_query_vector(query: str, model_name: str) -> str:
    return query_vector_literal(query, model_name)


def pg_vector_retrieval_required() -> bool:
    return os.getenv("PG_VECTOR_RETRIEVAL_REQUIRED", "0").strip().lower() in {"1", "true", "yes", "y"}


@lru_cache(maxsize=4)
def helper_assert_pg_vectors_ready(dsn: str | None, model_name: str) -> None:
    # 强制向量模式要求三个检索层级全部就绪，避免某一层静默退化为纯文本检索。
    targets = (
        ("document_cards", "document_card_embeddings"),
        ("document_views", "document_view_embeddings"),
        ("chunks", "chunk_embeddings"),
    )
    with connect(dsn) as conn:
        with conn.cursor() as cur:
            for source_table, embedding_table in targets:
                # 源表条数是当前数据快照应具备的向量基线。
                cur.execute(f"SELECT count(*) FROM {source_table}")
                source_count = int(cur.fetchone()[0])
                # 按模型过滤向量条数，防止把旧模型生成的向量误判为已完成。
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
    # 时间过滤 SQL 默认使用裸列名，这里改成 card 表别名以消除联表歧义。
    time_sql = helper_time_filter_sql(time_range, params).replace("publication_date", "dc.publication_date")
    sql = f"""
        SELECT dc.doc_id, dc.title, d.abstract, dc.publication_date, dc.source_institution, dc.clinical_department,
               -- 余弦距离越小越相似；转换为相似度后可与其他通道统一按降序解释。
               1 - (e.embedding <=> %s::vector) AS score
        FROM document_card_embeddings e
        JOIN document_cards dc ON dc.doc_id = e.doc_id
        JOIN documents d ON d.doc_id = dc.doc_id
        WHERE {' AND '.join(where)} {time_sql}
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
    """
    params.extend([vector, topk])
    with connect(dsn) as conn:
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
    with connect(dsn) as conn:
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
    with connect(dsn) as conn:
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


def helper_clip_text(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def helper_source_quote_context(prev_content: str, content: str, next_content: str) -> str:
    parts = []
    if prev_content:
        parts.append("[previous] " + helper_clip_text(prev_content[-500:], 500))
    parts.append("[current] " + helper_clip_text(content, 1400))
    if next_content:
        parts.append("[next] " + helper_clip_text(next_content[:500], 500))
    return "\n".join(parts)


def helper_section_path_items(section_path: Any) -> list[Any]:
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
    return json.dumps(helper_section_path_items(section_path), ensure_ascii=False, separators=(",", ":"))


def helper_enrich_chunk_context_pg(dsn: str | None, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not items:
        return []
    enriched: list[dict[str, Any]] = []
    with connect(dsn) as conn:
        with conn.cursor() as cur:
            for item in items:
                chunk_index = item.get("chunk_index")
                prev_chunk_id = None
                next_chunk_id = None
                prev_content = ""
                next_content = ""
                if chunk_index is not None:
                    cur.execute(
                        """
                        SELECT chunk_id, chunk_index, content
                        FROM chunks
                        WHERE doc_id=%s AND chunk_index IN (%s, %s)
                        """,
                        (item["doc_id"], int(chunk_index) - 1, int(chunk_index) + 1),
                    )
                    for row in cur.fetchall():
                        if row[1] == int(chunk_index) - 1:
                            prev_chunk_id = row[0]
                            prev_content = row[2] or ""
                        elif row[1] == int(chunk_index) + 1:
                            next_chunk_id = row[0]
                            next_content = row[2] or ""

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
                updated["prev_chunk_id"] = prev_chunk_id
                updated["next_chunk_id"] = next_chunk_id
                updated["char_start"] = section_char_start
                updated["char_end"] = section_char_end
                updated["char_span_kind"] = "section" if section_char_start is not None or section_char_end is not None else None
                updated["source_quote_context"] = helper_source_quote_context(prev_content, item.get("content", ""), next_content)
                enriched.append(updated)
    return enriched


def helper_department_fields(item: dict[str, Any]) -> dict[str, Any]:
    labels = [label for label in str(item.get("clinical_department") or "").split("|") if label] or ["未分类"]
    item = dict(item)
    item.update(clinical_department=labels[0], clinical_departments=labels, department_scope="compositive" if len(labels) > 1 else "single")
    return item


def helper_route_documents_pg(items: list[dict[str, Any]], clinical_department: str | None) -> list[dict[str, Any]]:
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
    if not clinical_department:
        return items[:100]
    doc_ids = list(dict.fromkeys(item["doc_id"] for item in items))
    scopes: dict[str, str] = {}
    if doc_ids:
        with connect(dsn) as conn:
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


def search_documents_hybrid_pg(
    query: str, dsn: str | None = None, source_institution: str | None = None,
    clinical_department: str | None = None, time_range: str | dict[str, str] | None = None,
    publication_date: str | None = None, topk: int = 20, pool_size: int = 50,
    model_name: str = DEFAULT_MODEL, reranker: DocumentReranker | None = None, document_kind: str = "guideline",
) -> list[dict[str, Any]]:
    """BM25+Dense multichannel document recall top50, then one rerank to top20."""
    require_vector = pg_vector_retrieval_required()
    if require_vector:
        helper_assert_pg_vectors_ready(dsn, model_name)
    recall_n = 500 if clinical_department else 100
    card_text = search_document_cards_pg(query, dsn, source_institution, clinical_department, time_range, publication_date, recall_n, document_kind=document_kind)
    view_text = search_document_views_pg(query, dsn, source_institution, clinical_department, time_range, publication_date, recall_n, document_kind=document_kind)
    try:
        card_dense = vector_search_document_cards_pg(
            query, dsn, source_institution, clinical_department, time_range, publication_date, recall_n, model_name, document_kind=document_kind
        )
    except Exception:
        if require_vector:
            raise
        card_dense = []
    try:
        view_dense = vector_search_document_views_pg(
            query, dsn, source_institution, clinical_department, time_range, publication_date, recall_n, model_name, document_kind=document_kind
        )
    except Exception:
        if require_vector:
            raise
        view_dense = []
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
    """BM25+Dense chunk recall top100, then one rerank to top30."""
    require_vector = pg_vector_retrieval_required()
    if require_vector:
        helper_assert_pg_vectors_ready(dsn, model_name)
    recall_n = 1000 if clinical_department else 100
    text_ranked = retrieve_chunks_pg(query, dsn, source_institution, clinical_department, time_range, publication_date, recall_n, document_kind=document_kind)
    try:
        vector_ranked = vector_retrieve_chunks_pg(
            query, dsn, source_institution, clinical_department, time_range, publication_date, recall_n, model_name, document_kind=document_kind
        )
    except Exception:
        if require_vector:
            raise
        vector_ranked = []
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


def search_documents_with_consensus_fallback_pg(
    query: str, dsn: str | None = None, source_institution: str | None = None,
    clinical_department: str | None = None, time_range: str | dict[str, str] | None = None,
    publication_date: str | None = None, topk: int = 20, reranker: DocumentReranker | None = None,
) -> list[dict[str, Any]]:
    wanted = min(topk, 20)
    guidelines = search_documents_hybrid_pg(
        query, dsn, source_institution, clinical_department, time_range, publication_date,
        topk=wanted, reranker=reranker, document_kind="guideline",
    )
    output = [{**item, "document_kind": "guideline", "is_fallback": False} for item in guidelines]
    deficit = wanted - len(output)
    if deficit <= 0:
        return output
    consensus = search_documents_hybrid_pg(
        query, dsn, source_institution, clinical_department, time_range, publication_date,
        topk=deficit, reranker=reranker, document_kind="consensus",
    )
    output.extend(
        {**item, "document_kind": "consensus", "is_fallback": True,
         "fallback_reason": "insufficient_guideline_results", "fallback_rank": rank}
        for rank, item in enumerate(consensus, 1)
    )
    return output[:wanted]


def retrieve_chunks_with_consensus_fallback_pg(
    query: str, dsn: str | None = None, source_institution: str | None = None,
    clinical_department: str | None = None, time_range: str | dict[str, str] | None = None,
    publication_date: str | None = None, topk: int = 30, reranker: ChunkReranker | None = None,
) -> list[dict[str, Any]]:
    wanted = min(topk, 30)
    guidelines = retrieve_chunks_hybrid_pg(
        query, dsn, source_institution, clinical_department, time_range, publication_date,
        topk=wanted, reranker=reranker, document_kind="guideline",
    )
    output = [{**item, "document_kind": "guideline", "is_fallback": False} for item in guidelines]
    deficit = wanted - len(output)
    if deficit <= 0:
        return output
    consensus = retrieve_chunks_hybrid_pg(
        query, dsn, source_institution, clinical_department, time_range, publication_date,
        topk=deficit, reranker=reranker, document_kind="consensus",
    )
    output.extend(
        {**item, "document_kind": "consensus", "is_fallback": True,
         "fallback_reason": "insufficient_guideline_results", "fallback_rank": rank}
        for rank, item in enumerate(consensus, 1)
    )
    return output[:wanted]
