"""Hybrid SQLite BM25 plus FAISS retrieval with RRF fusion."""

from __future__ import annotations

from contextlib import closing

import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

from src.retrieval.common import (
    compact_text as helper_compact,
    contains_exact_phrase as helper_contains_exact_phrase,
    enrich_chunk_metadata as helper_enrich_chunk_metadata,
    fill_consensus_fallback as helper_fill_consensus_fallback,
    parse_time_range as helper_time_range,
    query_terms as helper_query_terms,
    source_quote_context as helper_source_quote_context,
)
from src.retrieval.reranker import ChunkReranker, default_chunk_reranker, quality_score_multiplier
from src.retrieval.rrf import rrf_fusion
from src.retrieval.sqlite_store import DEFAULT_DB_PATH, connect, retrieve_chunks_sqlite
from src.retrieval.vector_store import vector_search
from src.utils.io import DATA_DIR
from src.utils.records import publication_year as helper_publication_year


GUIDE_RE = re.compile(r"(指南|共识|guideline|guidelines|consensus|practice parameters?|recommendations?)", re.I)
JOURNAL_HEADER_RE = re.compile(
    r"(中华.{0,16}杂志.{0,90}第\s*\d+\s*卷.{0,50}第\s*\d+\s*期|"
    r"Chin\s+J\s+.{0,90}\bVol\.?\s*\d+.{0,50}\bNo\.?\s*\d+|"
    r"\bVol\.?\s*\d+.{0,50}\bNo\.?\s*\d+)",
    re.I,
)
DOCUMENT_QUALITY_COLUMNS = (
    "cleaning_quality, cleaning_flags, source_pdf_text_quality, source_pdf_needs_ocr, source_pdf_is_scanned, "
    "pdf_text_quality, pdf_needs_ocr, pdf_is_scanned, ocr_engine, ocr_applied, ocr_status, ocr_error"
)




def helper_metadata_sql(
    source_institution: str | None,
    clinical_department: str | None,
    time_range: str | dict[str, str] | None,
    publication_date: str | None,
    exclude_reference_sections: bool = False,
    alias: str | None = None,
) -> tuple[str, list[Any]]:
    prefix = f"{alias}." if alias else ""
    clauses: list[str] = []
    params: list[Any] = []
    if source_institution:
        clauses.append(f"{prefix}source_institution LIKE ?")
        params.append(f"%{source_institution}%")
    if clinical_department:
        clauses.append(f"{prefix}clinical_department LIKE ?")
        params.append(f"%{clinical_department}%")
    start, end = helper_time_range(time_range)
    if start:
        clauses.append(f"{prefix}publication_date >= ?")
        params.append(start)
    if end:
        clauses.append(f"{prefix}publication_date <= ?")
        params.append(end)
    if publication_date:
        clauses.append(f"{prefix}publication_date = ?")
        params.append(publication_date)
    if exclude_reference_sections:
        clauses.append(f"{prefix}is_reference_section = 0")
    return (" AND ".join(clauses), params)


def helper_select_by_ids(
    db_path: str | Path,
    table: str,
    id_field: str,
    ids: list[str],
    columns: str,
    source_institution: str | None,
    clinical_department: str | None,
    time_range: str | dict[str, str] | None,
    publication_date: str | None,
    exclude_reference_sections: bool = False,
    document_kind: str | None = None,
    join_documents: bool = False,
) -> dict[str, sqlite3.Row]:
    # SQL 负责元数据过滤，返回结果再按召回 ID 顺序恢复名次，不能依赖数据库自然顺序。
    if not ids:
        return {}
    alias = "t" if join_documents else None
    prefix = f"{alias}." if alias else ""
    placeholders = ",".join("?" for _ in ids)
    where = [f"{prefix}{id_field} IN ({placeholders})"]
    params: list[Any] = list(ids)
    metadata_where, metadata_params = helper_metadata_sql(
        source_institution,
        clinical_department,
        time_range,
        publication_date,
        exclude_reference_sections,
        alias,
    )
    if metadata_where:
        where.append(metadata_where)
        params.extend(metadata_params)
    if join_documents:
        from_sql = f"{table} t JOIN documents d ON d.doc_id=t.doc_id"
        if document_kind:
            where.append("d.document_kind=?")
            params.append(document_kind)
    else:
        from_sql = table
        if document_kind:
            where.append(f"EXISTS (SELECT 1 FROM documents d_kind WHERE d_kind.doc_id={table}.doc_id AND d_kind.document_kind=?)")
            params.append(document_kind)
    sql = f"SELECT {columns} FROM {from_sql} WHERE " + " AND ".join(where)
    with closing(connect(db_path)) as conn:
        rows = conn.execute(sql, params).fetchall()
    return {row[id_field]: row for row in rows}

def helper_doc_row_to_result(row: sqlite3.Row) -> dict[str, Any]:
    keys = set(row.keys())

    def value(name: str, default: Any = "") -> Any:
        return row[name] if name in keys else default

    def flag(name: str) -> bool:
        raw = value(name, False)
        return raw is True or str(raw).strip().lower() in {"1", "true", "yes", "y"}

    department_value = str(row["clinical_department"] or "")
    departments = [item for item in department_value.split("|") if item] or ["未分类"]
    return {
        "doc_id": row["doc_id"],
        "title": row["title"],
        "abstract": row["abstract"],
        "publication_date": row["publication_date"],
        "source_institution": row["source_institution"],
        "clinical_department": departments[0],
        "clinical_departments": departments,
        "department_scope": "compositive" if len(departments) > 1 else "single",
        "document_kind": value("document_kind", "guideline"),
        "cleaning_quality": value("cleaning_quality"),
        "cleaning_flags": value("cleaning_flags"),
        "source_pdf_text_quality": value("source_pdf_text_quality"),
        "source_pdf_needs_ocr": flag("source_pdf_needs_ocr"),
        "source_pdf_is_scanned": flag("source_pdf_is_scanned"),
        "pdf_text_quality": value("pdf_text_quality"),
        "pdf_needs_ocr": flag("pdf_needs_ocr"),
        "pdf_is_scanned": flag("pdf_is_scanned"),
        "ocr_engine": value("ocr_engine"),
        "ocr_applied": flag("ocr_applied"),
        "ocr_status": value("ocr_status"),
        "ocr_error": value("ocr_error"),
    }


def helper_chunk_row_to_result(row: sqlite3.Row) -> dict[str, Any]:
    try:
        section_path = json.loads(row["section_path"] or "[]")
    except json.JSONDecodeError:
        section_path = []
    department_value = str(row["clinical_department"] or "")
    departments = [item for item in department_value.split("|") if item] or ["未分类"]
    item = {
        "chunk_id": row["chunk_id"],
        "doc_id": row["doc_id"],
        "content": row["content"],
        "title": row["title"],
        "publication_date": row["publication_date"],
        "source_institution": row["source_institution"],
        "clinical_department": departments[0],
        "clinical_departments": departments,
        "department_scope": "compositive" if len(departments) > 1 else "single",
        "section_path": section_path,
        "retrieval_text": row["retrieval_text"] if "retrieval_text" in row.keys() else "",
        "chunk_type": row["chunk_type"] if "chunk_type" in row.keys() else "",
        "token_count": row["token_count"] if "token_count" in row.keys() else 0,
        "retrieval_key": row["retrieval_key"],
        "source_file": row["source_file"],
        "markdown_clean_path": row["markdown_clean_path"],
        "chunk_index": row["chunk_index"],
        "is_background": bool(row["is_background"]) if "is_background" in row.keys() else False,
        "is_reference_section": bool(row["is_reference_section"]) if "is_reference_section" in row.keys() else False,
    }
    return helper_enrich_chunk_metadata(item)




def helper_fuse(
    id_field: str,
    bm25_results: list[dict[str, Any]],
    vector_results: list[dict[str, Any]],
    topk: int,
) -> list[dict[str, Any]]:
    bm25_ids = [item[id_field] for item in bm25_results]
    vector_ids = [item[id_field] for item in vector_results]
    fused = rrf_fusion([bm25_ids, vector_ids])
    by_id: dict[str, dict[str, Any]] = {}
    for item in bm25_results + vector_results:
        by_id.setdefault(item[id_field], item)
    bm25_rank = {item_id: rank for rank, item_id in enumerate(bm25_ids, start=1)}
    vector_rank = {item_id: rank for rank, item_id in enumerate(vector_ids, start=1)}

    output: list[dict[str, Any]] = []
    for item_id, score in fused:
        item = dict(by_id[item_id])
        item["score"] = score
        item["retrieval_scores"] = {
            "bm25_rank": bm25_rank.get(item_id),
            "vector_rank": vector_rank.get(item_id),
            "rrf_score": score,
        }
        output.append(item)
        if len(output) >= topk:
            break
    return output




def helper_field_term_hit(text: str, terms: list[str]) -> bool:
    lowered = (text or "").lower()
    compacted = helper_compact(text)
    for term in terms:
        if term in lowered or helper_compact(term) in compacted:
            return True
    return False


def helper_is_journal_header_title(title: str) -> bool:
    normalized = re.sub(r"\s+", " ", title or "").strip()
    if not normalized:
        return True
    if JOURNAL_HEADER_RE.search(normalized) and not GUIDE_RE.search(normalized):
        return True
    visible = [char for char in normalized if not char.isspace()]
    if len(visible) > 20:
        symbol_count = sum(1 for char in visible if ord(char) < 128 and not char.isalnum())
        if symbol_count / max(1, len(visible)) > 0.45:
            return True
    return False




def helper_recency_boost(publication_date: str | None, enabled: bool) -> float:
    if not enabled:
        return 0.0
    year = helper_publication_year(publication_date)
    if year is None:
        return 0.0
    return max(0.0, min(0.15, (year - 2012) / max(1, 2026 - 2012) * 0.15))


def helper_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def helper_rerank_documents(
    items: list[dict[str, Any]],
    query: str,
    source_institution: str | None,
    recency_boost: bool,
    topk: int,
) -> list[dict[str, Any]]:
    terms = helper_query_terms(query)
    ranked: list[dict[str, Any]] = []
    for item in items:
        title = item.get("title", "")
        abstract = item.get("abstract", "")
        # 期刊页眉常被误识别为标题，直接参与排序会污染高位结果。
        if helper_is_journal_header_title(title):
            continue
        retrieval_scores = item.get("retrieval_scores", {})
        boosts: dict[str, float] = {}
        matched_fields: list[str] = []
        exact_phrase_fields: list[str] = []

        if helper_field_term_hit(title, terms):
            matched_fields.append("title")
            boosts["title"] = 0.35
        if helper_field_term_hit(abstract, terms):
            matched_fields.append("abstract")
            boosts["abstract"] = 0.12
        if helper_contains_exact_phrase(title, query):
            exact_phrase_fields.append("title")
            boosts["exact_phrase_title"] = 0.45
        elif helper_contains_exact_phrase(abstract, query):
            exact_phrase_fields.append("abstract")
            boosts["exact_phrase_abstract"] = 0.2
        if retrieval_scores.get("bm25_rank") is not None:
            matched_fields.append("keyword")
        if retrieval_scores.get("vector_rank") is not None:
            matched_fields.append("vector")
        if source_institution and source_institution.lower() in (item.get("source_institution") or "").lower():
            matched_fields.append("source_institution")
            boosts["source_institution"] = 0.2
        if GUIDE_RE.search(title):
            matched_fields.append("document_type")
            boosts["guideline_or_consensus"] = 0.18
        recency = helper_recency_boost(item.get("publication_date"), recency_boost)
        if recency:
            boosts["publication_date_recency"] = recency
        # 关键词和向量同时命中表示不同召回机制达成一致，给予小幅稳定加成。
        if retrieval_scores.get("bm25_rank") is not None and retrieval_scores.get("vector_rank") is not None:
            boosts["bm25_vector_agreement"] = 0.1

        base_score = float(item.get("score", 0.0))
        boost_total = sum(boosts.values())
        quality_multiplier, quality_penalties = quality_score_multiplier(item)
        item = dict(item)
        # 先叠加匹配加分，再乘质量系数，避免低质量 OCR 文档靠词频占据高位。
        item["score"] = base_score * (1.0 + boost_total) * quality_multiplier
        item["read_key"] = item["doc_id"]
        item["match_reason"] = {
            "matched_fields": sorted(set(matched_fields)),
            "exact_phrase_fields": exact_phrase_fields,
            "boosts": boosts,
            "quality_penalties": quality_penalties,
            "quality_multiplier": quality_multiplier,
            "bm25_rank": retrieval_scores.get("bm25_rank"),
            "vector_rank": retrieval_scores.get("vector_rank"),
            "base_rrf_score": retrieval_scores.get("rrf_score", base_score),
        }
        ranked.append(item)
    ranked.sort(
        key=lambda item: (
            -float(item.get("score", 0.0)),
            item.get("match_reason", {}).get("bm25_rank") is None,
            item.get("match_reason", {}).get("vector_rank") is None,
            -(helper_publication_year(item.get("publication_date")) or 0),
            item.get("doc_id", ""),
        )
    )
    standardized = []
    for item in ranked[:topk]:
        standardized.append(
            {
                "doc_id": item.get("doc_id", ""),
                "title": item.get("title", ""),
                "abstract": item.get("abstract", ""),
                "publication_date": item.get("publication_date", "unknown"),
                "source_institution": item.get("source_institution", "Unknown"),
                "clinical_department": item.get("clinical_department", "未分类"),
                "cleaning_quality": item.get("cleaning_quality", ""),
                "cleaning_flags": item.get("cleaning_flags", ""),
                "source_pdf_text_quality": item.get("source_pdf_text_quality", ""),
                "source_pdf_needs_ocr": item.get("source_pdf_needs_ocr", False),
                "source_pdf_is_scanned": item.get("source_pdf_is_scanned", False),
                "pdf_text_quality": item.get("pdf_text_quality", ""),
                "pdf_needs_ocr": item.get("pdf_needs_ocr", False),
                "pdf_is_scanned": item.get("pdf_is_scanned", False),
                "ocr_engine": item.get("ocr_engine", ""),
                "ocr_applied": item.get("ocr_applied", False),
                "ocr_status": item.get("ocr_status", ""),
                "ocr_error": item.get("ocr_error", ""),
                "score": item.get("score", 0.0),
                "match_reason": item.get("match_reason", {}),
                "read_key": item.get("read_key", item.get("doc_id", "")),
                "retrieval_scores": item.get("retrieval_scores", {}),
            }
        )
    return standardized



def helper_find_chunk_span(markdown_path: str, content: str) -> tuple[int | None, int | None]:
    path = Path(markdown_path or "")
    if not path.exists() or not content:
        return None, None
    try:
        markdown = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, None
    needle = content.strip()
    if not needle:
        return None, None
    position = markdown.find(needle)
    if position < 0:
        # 完整文本可能因清洗产生轻微差异，先用前 300 字做低成本回退定位。
        snippet = needle[:300]
        position = markdown.find(snippet)
        if position < 0:
            # 最后忽略空白进行匹配，同时保存压缩字符到原文偏移的映射。
            compact_markdown_chars: list[str] = []
            compact_markdown_offsets: list[int] = []
            for index, char in enumerate(markdown):
                if not char.isspace():
                    compact_markdown_chars.append(char.lower())
                    compact_markdown_offsets.append(index)
            compact_needle = "".join(char.lower() for char in needle if not char.isspace())
            if len(compact_needle) < 20:
                return None, None
            compact_position = "".join(compact_markdown_chars).find(compact_needle[:300])
            if compact_position < 0:
                return None, None
            start = compact_markdown_offsets[compact_position]
            end_index = min(compact_position + len(compact_needle) - 1, len(compact_markdown_offsets) - 1)
            end = compact_markdown_offsets[end_index] + 1
            return start, end
    return position, position + len(needle)




def helper_enrich_chunk_context(db_path: str | Path, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not items:
        return []
    with closing(connect(db_path)) as conn:
        enriched: list[dict[str, Any]] = []
        for item in items:
            section_path_json = json.dumps(item.get("section_path") or [], ensure_ascii=False, separators=(",", ":"))
            section = conn.execute(
                """
                SELECT heading, char_start, char_end
                FROM sections
                WHERE doc_id=? AND section_path=?
                ORDER BY section_index
                LIMIT 1
                """,
                (item["doc_id"], section_path_json),
            ).fetchone()
            heading = section["heading"] if section else None
            section_char_start = section["char_start"] if section else None
            section_char_end = section["char_end"] if section else None
            char_start, char_end = helper_find_chunk_span(item.get("markdown_clean_path", ""), item.get("content", ""))
            if char_start is None:
                char_start = section_char_start
            if char_end is None:
                char_end = section_char_end

            updated = dict(item)
            display_heading = heading or ((item.get("section_path") or [None])[-1])
            if display_heading and helper_is_journal_header_title(str(display_heading)):
                display_heading = item.get("title")
            updated["heading"] = display_heading
            updated["prev_chunk_id"] = None
            updated["next_chunk_id"] = None
            updated["char_start"] = char_start
            updated["char_end"] = char_end
            updated["source_quote_context"] = helper_source_quote_context("", item.get("content", ""), "")
            enriched.append(updated)
    return enriched


def search_documents_hybrid(
    query: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    index_dir: str | Path = DATA_DIR / "index",
    source_institution: str | None = None,
    clinical_department: str | None = None,
    time_range: str | dict[str, str] | None = None,
    publication_date: str | None = None,
    recency_boost: bool = False,
    topk: int = 20,
    bm25_top_n: int = 200,
    vector_top_n: int = 50,
) -> list[dict[str, Any]]:
    from src.retrieval.document_multiview_search import has_document_representations_sqlite, search_documents_multiview

    if not has_document_representations_sqlite(db_path):
        raise RuntimeError("Document hybrid search requires document_cards/document_views; rebuild rag.sqlite.")
    return search_documents_multiview(
        query,
        db_path,
        index_dir=index_dir,
        source_institution=source_institution,
        clinical_department=clinical_department,
        time_range=time_range,
        publication_date=publication_date,
        recency_boost=recency_boost,
        topk=topk,
        card_bm25_top_n=bm25_top_n,
        view_bm25_top_n=bm25_top_n,
        card_vector_top_n=vector_top_n,
        view_vector_top_n=vector_top_n,
    )


def helper_route_chunk_candidates(
    items: list[dict[str, Any]], db_path: str | Path, clinical_department: str | None, top_n: int,
) -> list[dict[str, Any]]:
    if not clinical_department:
        return items[:top_n]
    doc_ids = list(dict.fromkeys(item["doc_id"] for item in items))
    scopes: dict[str, str] = {}
    if doc_ids:
        placeholders = ",".join("?" for _ in doc_ids)
        conn = connect(db_path)
        try:
            rows = conn.execute(f"SELECT doc_id, clinical_department FROM documents WHERE doc_id IN ({placeholders})", doc_ids).fetchall()
        finally:
            conn.close()
        scopes = {row["doc_id"]: ("compositive" if "|" in str(row["clinical_department"] or "") else "single") for row in rows}
    single_n = round(top_n * 10 / 11)
    compositive_n = top_n - single_n
    single = [item for item in items if scopes.get(item["doc_id"], "single") == "single"][:single_n]
    compositive = [item for item in items if scopes.get(item["doc_id"]) == "compositive"][:compositive_n]
    selected_ids = {item["chunk_id"] for item in single + compositive}
    selected = single + compositive
    selected.extend(item for item in items if item["chunk_id"] not in selected_ids and len(selected) < top_n)
    rank = {item["chunk_id"]: index for index, item in enumerate(items)}
    return sorted(selected, key=lambda item: rank[item["chunk_id"]])[:top_n]


def recall_chunks_hybrid(
    query: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    index_dir: str | Path = DATA_DIR / "index",
    source_institution: str | None = None,
    clinical_department: str | None = None,
    time_range: str | dict[str, str] | None = None,
    publication_date: str | None = None,
    topk: int = 100,
    bm25_top_n: int = 100,
    vector_top_n: int = 100,
    exclude_reference_sections: bool = True,
    document_kind: str | None = None,
) -> list[dict[str, Any]]:
    """Fuse BM25 and Dense channels, then return the pre-rerank candidate pool."""
    # 词法与向量通道独立召回后再做 RRF；任何通道的原始分数都不能直接跨通道比较。
    channel_n = max(bm25_top_n, vector_top_n, topk)
    recall_n = channel_n
    bm25_pool = retrieve_chunks_sqlite(
        query, db_path, source_institution=source_institution, clinical_department=clinical_department,
        time_range=time_range, topk=recall_n, exclude_reference_sections=exclude_reference_sections,
        document_kind=document_kind,
    )
    bm25_rows = helper_select_by_ids(
        db_path, "chunks", "chunk_id", [item["chunk_id"] for item in bm25_pool],
        "t.chunk_id AS chunk_id, t.doc_id AS doc_id, t.content AS content, t.retrieval_text AS retrieval_text, "
        "t.chunk_type AS chunk_type, t.token_count AS token_count, t.title AS title, t.publication_date AS publication_date, "
        "t.source_institution AS source_institution, t.clinical_department AS clinical_department, "
        "t.section_path AS section_path, t.chunk_index AS chunk_index, t.retrieval_key AS retrieval_key, "
        "d.source_file AS source_file, d.markdown_clean_path AS markdown_clean_path, "
        "t.is_background AS is_background, t.is_reference_section AS is_reference_section",
        source_institution, clinical_department, time_range, publication_date, exclude_reference_sections, document_kind,
        join_documents=True,
    )
    bm25_results = [dict(item) for item in bm25_pool if item["chunk_id"] in bm25_rows][:recall_n]

    index_path = Path(index_dir) / "faiss_chunks.index"
    mapping_path = Path(index_dir) / "faiss_chunks_mapping.jsonl"
    vector_ids = vector_search(query, index_path, mapping_path, "chunk_id", top_n=max(recall_n * 4, 200))
    vector_rows = helper_select_by_ids(
        db_path, "chunks", "chunk_id", vector_ids,
        "t.chunk_id AS chunk_id, t.doc_id AS doc_id, t.content AS content, t.retrieval_text AS retrieval_text, "
        "t.chunk_type AS chunk_type, t.token_count AS token_count, t.title AS title, t.publication_date AS publication_date, "
        "t.source_institution AS source_institution, t.clinical_department AS clinical_department, "
        "t.section_path AS section_path, t.chunk_index AS chunk_index, t.retrieval_key AS retrieval_key, "
        "d.source_file AS source_file, d.markdown_clean_path AS markdown_clean_path, "
        "t.is_background AS is_background, t.is_reference_section AS is_reference_section",
        source_institution, clinical_department, time_range, publication_date, exclude_reference_sections, document_kind,
        join_documents=True,
    )
    vector_results = [
        helper_chunk_row_to_result(vector_rows[chunk_id]) for chunk_id in vector_ids if chunk_id in vector_rows
    ][:recall_n]
    fused = helper_fuse("chunk_id", bm25_results, vector_results, max(topk, recall_n * 2))
    for item in fused:
        item["document_kind"] = document_kind or item.get("document_kind") or "guideline"
    return helper_route_chunk_candidates(fused, db_path, clinical_department, topk)


def retrieve_chunks_hybrid(
    query: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    index_dir: str | Path = DATA_DIR / "index",
    source_institution: str | None = None,
    clinical_department: str | None = None,
    time_range: str | dict[str, str] | None = None,
    publication_date: str | None = None,
    topk: int = 30,
    bm25_top_n: int = 100,
    vector_top_n: int = 100,
    exclude_reference_sections: bool = True,
    reranker: ChunkReranker | None = None,
    rerank_pool_n: int | None = None,
    document_kind: str | None = None,
) -> list[dict[str, Any]]:
    """BM25+Dense 召回指定候选池，经路由后执行一次重排。"""

    pool_n = 100 if rerank_pool_n is None else rerank_pool_n
    if pool_n < 1:
        raise ValueError("rerank_pool_n must be positive")
    # 候选池大小只控制重排前召回量，最终返回量仍受 topk 和系统上限约束。
    candidates = recall_chunks_hybrid(
        query, db_path, index_dir=index_dir, source_institution=source_institution,
        clinical_department=clinical_department, time_range=time_range, publication_date=publication_date,
        topk=pool_n, bm25_top_n=bm25_top_n, vector_top_n=vector_top_n,
        exclude_reference_sections=exclude_reference_sections, document_kind=document_kind,
    )
    enriched = helper_enrich_chunk_context(db_path, candidates)
    reranker = reranker or default_chunk_reranker()
    return reranker.rerank(query, enriched, min(topk, 30))


def retrieve_chunks_with_consensus_fallback(
    query: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    index_dir: str | Path = DATA_DIR / "index",
    source_institution: str | None = None,
    clinical_department: str | None = None,
    time_range: str | dict[str, str] | None = None,
    publication_date: str | None = None,
    topk: int = 30,
    exclude_reference_sections: bool = True,
    reranker: ChunkReranker | None = None,
) -> list[dict[str, Any]]:
    """Return guideline chunks first and query consensus only to fill a deficit."""
    wanted = min(topk, 30)
    guidelines = retrieve_chunks_hybrid(
        query, db_path, index_dir=index_dir, source_institution=source_institution,
        clinical_department=clinical_department, time_range=time_range, publication_date=publication_date,
        topk=wanted, exclude_reference_sections=exclude_reference_sections, reranker=reranker,
        document_kind="guideline",
    )
    return helper_fill_consensus_fallback(
        guidelines,
        wanted,
        lambda deficit: retrieve_chunks_hybrid(
            query, db_path, index_dir=index_dir, source_institution=source_institution,
            clinical_department=clinical_department, time_range=time_range, publication_date=publication_date,
            topk=deficit, exclude_reference_sections=exclude_reference_sections, reranker=reranker,
            document_kind="consensus",
        ),
    )
