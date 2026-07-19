"""Multiview document-level search for guideline recall."""

from __future__ import annotations

from contextlib import closing

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.retrieval.chunk_doc_aggregator import aggregate_chunks_to_documents
from src.retrieval.hybrid import helper_doc_row_to_result, helper_select_by_ids, recall_chunks_hybrid
from src.retrieval.reranker import DocumentReranker, default_document_reranker
from src.retrieval.rrf import weighted_rrf_fusion
from src.retrieval.sqlite_store import (
    DEFAULT_DB_PATH,
    connect,
    search_document_cards_sqlite,
    search_document_views_sqlite,
    retrieve_chunks_sqlite,
)
from src.retrieval.vector_store import vector_search
from src.utils.io import DATA_DIR


CHANNEL_WEIGHTS = {
    "card_bm25": 0.45,
    "card_dense": 1.55,
    "view_bm25": 0.55,
    "view_dense": 1.85,
    "chunk_doc": 1.15,
}

QUALITY_SELECT_COLUMNS = (
    "cleaning_quality, cleaning_flags, source_pdf_text_quality, source_pdf_needs_ocr, source_pdf_is_scanned, "
    "pdf_text_quality, pdf_needs_ocr, pdf_is_scanned, ocr_engine, ocr_applied, ocr_status, ocr_error"
)
JOINED_QUALITY_SELECT_COLUMNS = ", ".join(
    f"d.{column} AS {column}" for column in QUALITY_SELECT_COLUMNS.split(", ")
)

@dataclass(frozen=True)
class MultiviewSearchConfig:
    db_path: str | Path = DEFAULT_DB_PATH
    index_dir: str | Path = DATA_DIR / "index"
    source_institution: str | None = None
    clinical_department: str | None = None
    time_range: str | dict[str, str] | None = None
    publication_date: str | None = None
    document_kind: str = "guideline"
    recency_boost: bool = False
    topk: int = 20
    card_bm25_top_n: int = 50
    card_vector_top_n: int = 220
    view_bm25_top_n: int = 70
    view_vector_top_n: int = 280
    chunk_top_n: int = 90

    @property
    def index_path(self) -> Path:
        return Path(self.index_dir)


def has_document_representations_sqlite(db_path: str | Path = DEFAULT_DB_PATH) -> bool:
    path = Path(db_path)
    if not path.exists():
        return False
    try:
        with closing(connect(path)) as conn:
            table_rows = conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type IN ('table', 'virtual table') AND name IN ('document_cards', 'document_views')
                """
            ).fetchall()
            if {row["name"] for row in table_rows} != {"document_cards", "document_views"}:
                return False
            cards = conn.execute("SELECT count(*) FROM document_cards").fetchone()[0]
            views = conn.execute("SELECT count(*) FROM document_views").fetchone()[0]
            return cards > 0 or views > 0
    except sqlite3.Error:
        return False


def helper_publication_date_matches(item: dict[str, Any], publication_date: str | None) -> bool:
    return not publication_date or item.get("publication_date") == publication_date


def helper_load_card_rows(
    db_path: str | Path,
    doc_ids: list[str],
    source_institution: str | None,
    clinical_department: str | None,
    time_range: str | dict[str, str] | None,
    publication_date: str | None,
    document_kind: str | None = None,
) -> dict[str, dict[str, Any]]:
    rows = helper_select_by_ids(
        db_path,
        "document_cards",
        "doc_id",
        doc_ids,
        f"t.doc_id AS doc_id, t.title AS title, t.card_text AS card_text, "
        f"t.publication_date AS publication_date, t.source_institution AS source_institution, "
        f"t.clinical_department AS clinical_department, {JOINED_QUALITY_SELECT_COLUMNS}",
        source_institution,
        clinical_department,
        time_range,
        publication_date,
        document_kind=document_kind,
        join_documents=True,
    )
    cards: dict[str, dict[str, Any]] = {}
    for doc_id, row in rows.items():
        cards[doc_id] = {
            "doc_id": row["doc_id"],
            "title": row["title"],
            "card_text": row["card_text"],
            "fields": {},
            "publication_date": row["publication_date"],
            "source_institution": row["source_institution"],
            "clinical_department": row["clinical_department"],
            "cleaning_quality": row["cleaning_quality"],
            "cleaning_flags": row["cleaning_flags"],
            "source_pdf_text_quality": row["source_pdf_text_quality"],
            "source_pdf_needs_ocr": bool(row["source_pdf_needs_ocr"]),
            "source_pdf_is_scanned": bool(row["source_pdf_is_scanned"]),
            "pdf_text_quality": row["pdf_text_quality"],
            "pdf_needs_ocr": bool(row["pdf_needs_ocr"]),
            "pdf_is_scanned": bool(row["pdf_is_scanned"]),
            "ocr_engine": row["ocr_engine"],
            "ocr_applied": bool(row["ocr_applied"]),
            "ocr_status": row["ocr_status"],
            "ocr_error": row["ocr_error"],
        }
    return cards

def helper_load_view_rows(
    db_path: str | Path,
    view_ids: list[str],
    source_institution: str | None,
    clinical_department: str | None,
    time_range: str | dict[str, str] | None,
    publication_date: str | None,
    document_kind: str | None = None,
) -> list[dict[str, Any]]:
    rows = helper_select_by_ids(
        db_path,
        "document_views",
        "view_id",
        view_ids,
        f"t.view_id AS view_id, t.doc_id AS doc_id, t.view_type AS view_type, t.priority AS priority, "
        f"t.title AS title, t.text AS text, t.publication_date AS publication_date, "
        f"t.source_institution AS source_institution, t.clinical_department AS clinical_department, "
        f"{JOINED_QUALITY_SELECT_COLUMNS}",
        source_institution,
        clinical_department,
        time_range,
        publication_date,
        document_kind=document_kind,
        join_documents=True,
    )
    output = []
    for view_id in view_ids:
        row = rows.get(view_id)
        if not row:
            continue
        output.append(
            {
                "view_id": row["view_id"],
                "doc_id": row["doc_id"],
                "view_type": row["view_type"],
                "priority": float(row["priority"] or 1.0),
                "title": row["title"],
                "text": row["text"],
                "publication_date": row["publication_date"],
                "source_institution": row["source_institution"],
                "clinical_department": row["clinical_department"],
                "cleaning_quality": row["cleaning_quality"],
                "cleaning_flags": row["cleaning_flags"],
                "source_pdf_text_quality": row["source_pdf_text_quality"],
                "source_pdf_needs_ocr": bool(row["source_pdf_needs_ocr"]),
                "source_pdf_is_scanned": bool(row["source_pdf_is_scanned"]),
                "pdf_text_quality": row["pdf_text_quality"],
                "pdf_needs_ocr": bool(row["pdf_needs_ocr"]),
                "pdf_is_scanned": bool(row["pdf_is_scanned"]),
                "ocr_engine": row["ocr_engine"],
                "ocr_applied": bool(row["ocr_applied"]),
                "ocr_status": row["ocr_status"],
                "ocr_error": row["ocr_error"],
            }
        )
    return output

def helper_rank_docs_from_views(views: list[dict[str, Any]], max_views_per_doc: int = 5) -> tuple[list[str], dict[str, list[dict[str, Any]]]]:
    scores: dict[str, float] = {}
    matched: dict[str, list[dict[str, Any]]] = {}
    for rank, view in enumerate(views, start=1):
        doc_id = view.get("doc_id")
        if not doc_id:
            continue
        scores[doc_id] = scores.get(doc_id, 0.0) + float(view.get("priority") or 1.0) / (60 + rank)
        if len(matched.get(doc_id, [])) < max_views_per_doc:
            matched.setdefault(doc_id, []).append({**view, "view_rank": rank})
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [doc_id for doc_id, _score in ranked], matched


def helper_rank_dict(ids: list[str]) -> dict[str, int]:
    ranks = {}
    for rank, item_id in enumerate(ids, start=1):
        ranks.setdefault(item_id, rank)
    return ranks


def helper_merge_matched_views(*view_maps: dict[str, list[dict[str, Any]]], max_views_per_doc: int = 5) -> dict[str, list[dict[str, Any]]]:
    merged: dict[str, list[dict[str, Any]]] = {}
    seen: dict[str, set[str]] = {}
    for view_map in view_maps:
        for doc_id, views in view_map.items():
            for view in views:
                view_id = view.get("view_id")
                if not view_id or view_id in seen.setdefault(doc_id, set()):
                    continue
                seen[doc_id].add(view_id)
                merged.setdefault(doc_id, []).append(view)
                if len(merged[doc_id]) >= max_views_per_doc:
                    break
    return merged


def helper_document_rows(
    db_path: str | Path,
    doc_ids: list[str],
    source_institution: str | None,
    clinical_department: str | None,
    time_range: str | dict[str, str] | None,
    publication_date: str | None,
    document_kind: str | None = None,
) -> dict[str, dict[str, Any]]:
    rows = helper_select_by_ids(
        db_path,
        "documents",
        "doc_id",
        doc_ids,
        f"doc_id, title, abstract, publication_date, source_institution, clinical_department, document_kind, {QUALITY_SELECT_COLUMNS}",
        source_institution,
        clinical_department,
        time_range,
        publication_date,
        document_kind=document_kind,
    )
    return {doc_id: helper_doc_row_to_result(row) for doc_id, row in rows.items()}


def helper_card_recall(query: str, config: MultiviewSearchConfig) -> tuple[list[str], list[str], dict[str, dict[str, Any]]]:
    card_bm25 = [
        item for item in search_document_cards_sqlite(
            query, config.db_path, source_institution=config.source_institution,
            clinical_department=config.clinical_department, time_range=config.time_range,
            topk=config.card_bm25_top_n, document_kind=config.document_kind,
        ) if helper_publication_date_matches(item, config.publication_date)
    ]
    card_bm25_ids = [item["doc_id"] for item in card_bm25]
    # 向量通道先扩大召回池，随后用数据库元数据过滤并截断到目标数量。
    card_dense_ids = vector_search(
        query, config.index_path / "faiss_document_cards.index",
        config.index_path / "faiss_document_cards_mapping.jsonl", "doc_id",
        top_n=max(config.card_vector_top_n * 4, 200),
    )
    card_dense_rows = helper_load_card_rows(
        config.db_path, card_dense_ids, config.source_institution, config.clinical_department,
        config.time_range, config.publication_date, config.document_kind,
    )
    card_dense_ids = [doc_id for doc_id in card_dense_ids if doc_id in card_dense_rows][:config.card_vector_top_n]
    card_records = {item["doc_id"]: item for item in card_bm25}
    card_records.update(card_dense_rows)
    return card_bm25_ids, card_dense_ids, card_records


def helper_view_recall(
    query: str, config: MultiviewSearchConfig,
) -> tuple[list[str], list[str], dict[str, list[dict[str, Any]]]]:
    view_bm25 = [
        item for item in search_document_views_sqlite(
            query, config.db_path, source_institution=config.source_institution,
            clinical_department=config.clinical_department, time_range=config.time_range,
            topk=config.view_bm25_top_n, document_kind=config.document_kind,
        ) if helper_publication_date_matches(item, config.publication_date)
    ]
    view_bm25_doc_ids, view_bm25_matches = helper_rank_docs_from_views(view_bm25)
    view_dense_ids = vector_search(
        query, config.index_path / "faiss_document_views.index",
        config.index_path / "faiss_document_views_mapping.jsonl", "view_id",
        top_n=max(config.view_vector_top_n * 4, 240),
    )
    view_dense = helper_load_view_rows(
        config.db_path, view_dense_ids, config.source_institution, config.clinical_department,
        config.time_range, config.publication_date, config.document_kind,
    )[:config.view_vector_top_n]
    # view_id 召回需要聚合回 doc_id，同一文档的多个命中 view 作为解释信息保留。
    view_dense_doc_ids, view_dense_matches = helper_rank_docs_from_views(view_dense)
    return view_bm25_doc_ids, view_dense_doc_ids, helper_merge_matched_views(view_bm25_matches, view_dense_matches)


def helper_chunk_doc_recall(query: str, config: MultiviewSearchConfig) -> tuple[list[str], dict[str, dict[str, Any]]]:
    chunk_results = recall_chunks_hybrid(
        query, config.db_path, index_dir=config.index_path, source_institution=config.source_institution,
        clinical_department=config.clinical_department, time_range=config.time_range,
        publication_date=config.publication_date, topk=config.chunk_top_n,
        bm25_top_n=config.chunk_top_n, vector_top_n=config.chunk_top_n,
        exclude_reference_sections=True, document_kind=config.document_kind,
    )
    chunk_doc_results = aggregate_chunks_to_documents(chunk_results, topk=max(config.chunk_top_n, 80))
    return [item["doc_id"] for item in chunk_doc_results], {item["doc_id"]: item for item in chunk_doc_results}


def helper_fuse_doc_ids(
    card_bm25_ids: list[str],
    card_dense_ids: list[str],
    view_bm25_doc_ids: list[str],
    view_dense_doc_ids: list[str],
    chunk_doc_ids: list[str],
    topk: int,
) -> list[tuple[str, float]]:
    rank_lists = [
        (card_bm25_ids, CHANNEL_WEIGHTS["card_bm25"]),
        (card_dense_ids, CHANNEL_WEIGHTS["card_dense"]),
        (view_bm25_doc_ids, CHANNEL_WEIGHTS["view_bm25"]),
        (view_dense_doc_ids, CHANNEL_WEIGHTS["view_dense"]),
        (chunk_doc_ids, CHANNEL_WEIGHTS["chunk_doc"]),
    ]
    return weighted_rrf_fusion([(ids, weight) for ids, weight in rank_lists if ids], k=60)[:topk]


def helper_channel_ranks(
    card_bm25_ids: list[str],
    card_dense_ids: list[str],
    view_bm25_doc_ids: list[str],
    view_dense_doc_ids: list[str],
    chunk_doc_ids: list[str],
) -> dict[str, dict[str, int]]:
    return {
        "card_bm25_rank": helper_rank_dict(card_bm25_ids),
        "card_dense_rank": helper_rank_dict(card_dense_ids),
        "view_bm25_rank": helper_rank_dict(view_bm25_doc_ids),
        "view_dense_rank": helper_rank_dict(view_dense_doc_ids),
        "chunk_doc_rank": helper_rank_dict(chunk_doc_ids),
    }


def helper_complete_card_records(
    candidate_ids: list[str],
    card_records: dict[str, dict[str, Any]],
    config: MultiviewSearchConfig,
) -> dict[str, dict[str, Any]]:
    missing_ids = [doc_id for doc_id in candidate_ids if doc_id not in card_records]
    if not missing_ids:
        return card_records
    completed = dict(card_records)
    completed.update(
        helper_load_card_rows(
            config.db_path,
            missing_ids,
            config.source_institution,
            config.clinical_department,
            config.time_range,
            config.publication_date,
            config.document_kind,
        )
    )
    return completed


def helper_build_candidates(
    candidate_ids: list[str],
    fused_scores: dict[str, float],
    card_records: dict[str, dict[str, Any]],
    matched_views: dict[str, list[dict[str, Any]]],
    chunk_by_doc: dict[str, dict[str, Any]],
    ranks: dict[str, dict[str, int]],
    config: MultiviewSearchConfig,
) -> list[dict[str, Any]]:
    docs = helper_document_rows(
        config.db_path,
        candidate_ids,
        config.source_institution,
        config.clinical_department,
        config.time_range,
        config.publication_date,
        config.document_kind,
    )
    candidates = []
    for doc_id in candidate_ids:
        doc = docs.get(doc_id)
        if not doc:
            continue
        # 保存每个渠道的原始名次，后续重排和结果解释都依赖这份证据。
        retrieval_scores = {name: channel[doc_id] for name, channel in ranks.items() if doc_id in channel}
        channels = sorted(name.removesuffix("_rank") for name in retrieval_scores)
        chunk_info = chunk_by_doc.get(doc_id, {})
        views = matched_views.get(doc_id, [])
        view_types = sorted({view.get("view_type") for view in views if view.get("view_type")})
        candidates.append(
            {
                **doc,
                "score": fused_scores.get(doc_id, 0.0),
                "read_key": doc_id,
                "document_card": card_records.get(doc_id, {}),
                "matched_views": views,
                "matched_chunks": chunk_info.get("matched_chunks", []),
                "retrieval_scores": retrieval_scores,
                "match_reason": {
                    "matched_fields": channels,
                    "channels": channels,
                    "view_types": view_types,
                    "matched_chunk_count": chunk_info.get("matched_chunk_count", 0),
                    "base_rrf_score": fused_scores.get(doc_id, 0.0),
                },
            }
        )
    return candidates


def helper_route_document_ids(candidate_ids: list[str], docs: dict[str, dict[str, Any]], clinical_department: str | None) -> list[str]:
    if not clinical_department:
        return candidate_ids[:50]
    single = [doc_id for doc_id in candidate_ids if docs.get(doc_id, {}).get("department_scope") == "single"][:45]
    compositive = [doc_id for doc_id in candidate_ids if docs.get(doc_id, {}).get("department_scope") == "compositive"][:5]
    selected = single + compositive
    seen = set(selected)
    selected.extend(doc_id for doc_id in candidate_ids if doc_id not in seen and len(selected) < 50)
    rank = {doc_id: index for index, doc_id in enumerate(candidate_ids)}
    return sorted(selected, key=rank.get)[:50]


def search_documents_multiview(
    query: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    index_dir: str | Path = DATA_DIR / "index",
    source_institution: str | None = None,
    clinical_department: str | None = None,
    time_range: str | dict[str, str] | None = None,
    publication_date: str | None = None,
    recency_boost: bool = False,
    topk: int = 20,
    card_bm25_top_n: int = 100,
    card_vector_top_n: int = 100,
    view_bm25_top_n: int = 100,
    view_vector_top_n: int = 100,
    chunk_top_n: int = 100,
    reranker: DocumentReranker | None = None,
    document_kind: str = "guideline",
) -> list[dict[str, Any]]:
    """Fuse BM25 and Dense card/view/chunk channels, then rerank documents."""

    config = MultiviewSearchConfig(
        db_path=db_path,
        index_dir=index_dir,
        source_institution=source_institution,
        clinical_department=clinical_department,
        time_range=time_range,
        publication_date=publication_date,
        document_kind=document_kind,
        recency_boost=recency_boost,
        topk=topk,
        card_bm25_top_n=card_bm25_top_n,
        card_vector_top_n=card_vector_top_n,
        view_bm25_top_n=view_bm25_top_n,
        view_vector_top_n=view_vector_top_n,
        chunk_top_n=chunk_top_n,
    )
    card_bm25_ids, card_dense_ids, card_records = helper_card_recall(query, config)
    view_bm25_doc_ids, view_dense_doc_ids, matched_views = helper_view_recall(query, config)
    chunk_doc_ids, chunk_by_doc = helper_chunk_doc_recall(query, config)
    fused = helper_fuse_doc_ids(card_bm25_ids, card_dense_ids, view_bm25_doc_ids, view_dense_doc_ids, chunk_doc_ids, 50)
    all_candidate_ids = [doc_id for doc_id, _score in fused]
    route_docs = helper_document_rows(db_path, all_candidate_ids, source_institution, clinical_department, time_range, publication_date, document_kind)
    # 路由只改变候选配额，不改写各渠道分数，最终排序仍由统一重排器完成。
    candidate_ids = helper_route_document_ids(all_candidate_ids, route_docs, clinical_department)
    fused_scores = dict(fused)
    ranks = helper_channel_ranks(card_bm25_ids, card_dense_ids, view_bm25_doc_ids, view_dense_doc_ids, chunk_doc_ids)
    card_records = helper_complete_card_records(candidate_ids, card_records, config)
    candidates = helper_build_candidates(candidate_ids, fused_scores, card_records, matched_views, chunk_by_doc, ranks, config)
    reranker = reranker or default_document_reranker(recency_boost=config.recency_boost)
    return reranker.rerank(query, candidates, min(topk, 20))


def search_documents_with_consensus_fallback(
    query: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    index_dir: str | Path = DATA_DIR / "index",
    source_institution: str | None = None,
    clinical_department: str | None = None,
    time_range: str | dict[str, str] | None = None,
    publication_date: str | None = None,
    recency_boost: bool = False,
    topk: int = 20,
    reranker: DocumentReranker | None = None,
) -> list[dict[str, Any]]:
    """Return guidelines first and query consensus only to fill a result deficit."""
    wanted = min(topk, 20)
    guidelines = search_documents_multiview(
        query, db_path, index_dir=index_dir, source_institution=source_institution,
        clinical_department=clinical_department, time_range=time_range, publication_date=publication_date,
        recency_boost=recency_boost, topk=wanted, reranker=reranker, document_kind="guideline",
    )
    output = [{**item, "document_kind": "guideline", "is_fallback": False} for item in guidelines]
    deficit = wanted - len(output)
    if deficit <= 0:
        return output
    consensus = search_documents_multiview(
        query, db_path, index_dir=index_dir, source_institution=source_institution,
        clinical_department=clinical_department, time_range=time_range, publication_date=publication_date,
        recency_boost=recency_boost, topk=deficit, reranker=reranker, document_kind="consensus",
    )
    output.extend(
        {**item, "document_kind": "consensus", "is_fallback": True,
         "fallback_reason": "insufficient_guideline_results", "fallback_rank": rank}
        for rank, item in enumerate(consensus, 1)
    )
    return output[:wanted]
