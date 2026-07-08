"""Hybrid PostgreSQL retrieval with full-text rank plus BGE-M3/pgvector rank."""

from __future__ import annotations

import json
from typing import Any

from src.retrieval.rrf import rrf_fusion
from src.storage.query_embedding import DEFAULT_MODEL, query_vector_literal
from src.storage.postgres_store import (
    _time_filter_sql,
    connect,
    retrieve_chunks_pg,
    search_document_cards_pg,
    search_document_views_pg,
)


def _query_vector(query: str, model_name: str) -> str:
    return query_vector_literal(query, model_name)


def vector_search_document_cards_pg(
    query: str,
    dsn: str | None = None,
    source_institution: str | None = None,
    clinical_department: str | None = None,
    time_range: str | dict[str, str] | None = None,
    publication_date: str | None = None,
    topk: int = 50,
    model_name: str = DEFAULT_MODEL,
) -> list[dict[str, Any]]:
    vector = _query_vector(query, model_name)
    params: list[Any] = [vector, model_name]
    where = ["e.model = %s"]
    if source_institution:
        where.append("dc.source_institution ILIKE %s")
        params.append(f"%{source_institution}%")
    if clinical_department:
        where.append("dc.clinical_department ILIKE %s")
        params.append(f"%{clinical_department}%")
    if publication_date:
        where.append("dc.publication_date = %s")
        params.append(publication_date)
    time_sql = _time_filter_sql(time_range, params).replace("publication_date", "dc.publication_date")
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
) -> list[dict[str, Any]]:
    vector = _query_vector(query, model_name)
    params: list[Any] = [vector, model_name]
    where = ["e.model = %s"]
    if source_institution:
        where.append("v.source_institution ILIKE %s")
        params.append(f"%{source_institution}%")
    if clinical_department:
        where.append("v.clinical_department ILIKE %s")
        params.append(f"%{clinical_department}%")
    if publication_date:
        where.append("v.publication_date = %s")
        params.append(publication_date)
    time_sql = _time_filter_sql(time_range, params).replace("publication_date", "v.publication_date")
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
) -> list[dict[str, Any]]:
    vector = _query_vector(query, model_name)
    params: list[Any] = [vector, model_name]
    where = ["e.model = %s", "c.is_reference_section = false"]
    if source_institution:
        where.append("c.source_institution ILIKE %s")
        params.append(f"%{source_institution}%")
    if clinical_department:
        where.append("c.clinical_department ILIKE %s")
        params.append(f"%{clinical_department}%")
    if publication_date:
        where.append("c.publication_date = %s")
        params.append(publication_date)
    time_sql = _time_filter_sql(time_range, params).replace("publication_date", "c.publication_date")
    sql = f"""
        SELECT c.chunk_id, c.doc_id, c.content, c.title, c.publication_date, c.source_institution,
               c.clinical_department, c.section_path, c.chunk_index, c.source_file, c.markdown_clean_path,
               c.chunk_type, c.retrieval_text,
               1 - (e.embedding <=> %s::vector) AS score, c.retrieval_key
        FROM chunk_embeddings e
        JOIN chunks c ON c.chunk_id = e.chunk_id
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


def _clip_text(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _source_quote_context(prev_content: str, content: str, next_content: str) -> str:
    parts = []
    if prev_content:
        parts.append("[previous] " + _clip_text(prev_content[-500:], 500))
    parts.append("[current] " + _clip_text(content, 1400))
    if next_content:
        parts.append("[next] " + _clip_text(next_content[:500], 500))
    return "\n".join(parts)


def _section_path_items(section_path: Any) -> list[Any]:
    if isinstance(section_path, str):
        try:
            value = json.loads(section_path)
        except json.JSONDecodeError:
            return [section_path]
        return value if isinstance(value, list) else [value]
    if isinstance(section_path, list):
        return section_path
    return list(section_path) if section_path else []


def _section_path_json(section_path: Any) -> str:
    return json.dumps(_section_path_items(section_path), ensure_ascii=False, separators=(",", ":"))


def _enrich_chunk_context_pg(dsn: str | None, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
                section_path_json = _section_path_json(item.get("section_path"))
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

                section_path_items = _section_path_items(item.get("section_path"))
                updated = dict(item)
                updated["heading"] = heading or ((section_path_items or [None])[-1])
                updated["prev_chunk_id"] = prev_chunk_id
                updated["next_chunk_id"] = next_chunk_id
                updated["char_start"] = section_char_start
                updated["char_end"] = section_char_end
                updated["char_span_kind"] = "section" if section_char_start is not None or section_char_end is not None else None
                updated["source_quote_context"] = _source_quote_context(prev_content, item.get("content", ""), next_content)
                enriched.append(updated)
    return enriched


def search_documents_hybrid_pg(
    query: str,
    dsn: str | None = None,
    source_institution: str | None = None,
    clinical_department: str | None = None,
    time_range: str | dict[str, str] | None = None,
    publication_date: str | None = None,
    topk: int = 10,
    pool_size: int = 50,
    model_name: str = DEFAULT_MODEL,
) -> list[dict[str, Any]]:
    card_text_ranked = search_document_cards_pg(
        query, dsn, source_institution, clinical_department, time_range, publication_date, pool_size
    )
    view_text_ranked = search_document_views_pg(
        query, dsn, source_institution, clinical_department, time_range, publication_date, pool_size
    )
    rank_lists: list[list[str]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for ranked in (card_text_ranked, view_text_ranked):
        if ranked:
            rank_lists.append([item["doc_id"] for item in ranked])
            by_id.update({item["doc_id"]: item for item in ranked})
    try:
        card_vector_ranked = vector_search_document_cards_pg(
            query, dsn, source_institution, clinical_department, time_range, publication_date, pool_size, model_name
        )
        if card_vector_ranked:
            rank_lists.append([item["doc_id"] for item in card_vector_ranked])
            by_id.update({item["doc_id"]: item for item in card_vector_ranked})
    except Exception:
        card_vector_ranked = []
    try:
        view_vector_ranked = vector_search_document_views_pg(
            query, dsn, source_institution, clinical_department, time_range, publication_date, pool_size, model_name
        )
        if view_vector_ranked:
            rank_lists.append([item["doc_id"] for item in view_vector_ranked])
            by_id.update({item["doc_id"]: item for item in view_vector_ranked})
    except Exception:
        view_vector_ranked = []
    if not rank_lists:
        return []
    fused = rrf_fusion(rank_lists)
    results = []
    for doc_id, score in fused[:topk]:
        item = dict(by_id[doc_id])
        item["score"] = score
        channels = []
        if any(entry["doc_id"] == doc_id for entry in card_text_ranked):
            channels.append("document_card_text")
        if any(entry["doc_id"] == doc_id for entry in view_text_ranked):
            channels.append("document_view_text")
        if any(entry["doc_id"] == doc_id for entry in card_vector_ranked):
            channels.append("document_card_vector")
        if any(entry["doc_id"] == doc_id for entry in view_vector_ranked):
            channels.append("document_view_vector")
        item["retrieval_channels"] = channels
        results.append(item)
    return results


def retrieve_chunks_hybrid_pg(
    query: str,
    dsn: str | None = None,
    source_institution: str | None = None,
    clinical_department: str | None = None,
    time_range: str | dict[str, str] | None = None,
    publication_date: str | None = None,
    topk: int = 5,
    pool_size: int = 50,
    model_name: str = DEFAULT_MODEL,
) -> list[dict[str, Any]]:
    text_ranked = retrieve_chunks_pg(query, dsn, source_institution, clinical_department, time_range, publication_date, pool_size)
    try:
        vector_ranked = vector_retrieve_chunks_pg(
            query, dsn, source_institution, clinical_department, time_range, publication_date, pool_size, model_name
        )
    except Exception:
        return text_ranked[:topk]
    by_id = {item["chunk_id"]: item for item in text_ranked}
    by_id.update({item["chunk_id"]: item for item in vector_ranked})
    fused = rrf_fusion([[item["chunk_id"] for item in vector_ranked], [item["chunk_id"] for item in text_ranked]])
    results = []
    for chunk_id, score in fused[:topk]:
        item = dict(by_id[chunk_id])
        item["score"] = score
        results.append(item)
    return _enrich_chunk_context_pg(dsn, results)
