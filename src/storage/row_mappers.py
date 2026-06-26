from __future__ import annotations

from typing import Any, Dict

from src.storage.utils import as_int, json_list, json_obj


JsonDict = Dict[str, Any]


def guideline_row(seed: JsonDict) -> JsonDict:
    return {
        "guideline_id": seed["guideline_id"],
        "title": seed.get("title") or "",
        "disease_area": seed.get("disease_area"),
        "guideline_type": seed.get("guideline_type") or "standard",
        "status": seed.get("status") or "active",
        "source": seed.get("source"),
        "issuer": seed.get("issuer"),
        "publication_url": seed.get("publication_url"),
        "pdf_url": seed.get("pdf_url"),
        "current_version": seed.get("current_version"),
        "update_frequency": seed.get("update_frequency"),
        "published_date": seed.get("published_date"),
        "last_updated_at": seed.get("last_updated_at"),
        "created_at": seed.get("created_at") or "",
        "updated_at": seed.get("updated_at") or "",
    }


def paper_row(seed: JsonDict) -> JsonDict:
    return {
        "paper_id": seed["paper_id"],
        "title": seed.get("title") or "",
        "pmid": seed.get("pmid"),
        "doi": seed.get("doi"),
        "abstract": seed.get("abstract"),
        "authors": json_list(seed.get("authors")),
        "journal": seed.get("journal"),
        "publication_date": seed.get("publication_date"),
        "article_type": seed.get("article_type"),
        "source_database": seed.get("source_database"),
        "url": seed.get("url"),
        "first_seen_at": seed.get("first_seen_at") or "",
        "screening_status": seed.get("screening_status") or "unscreened",
    }


def cleaned_record_row(item: JsonDict) -> JsonDict:
    direct = item.get("direct_extraction") or {}
    return {
        "record_id": item["record_id"],
        "guideline_id": direct.get("guideline_id") or (item.get("guideline_seed") or {}).get("guideline_id"),
        "paper_id": direct.get("paper_id") or (item.get("paper_seed") or {}).get("paper_id"),
        "source": item.get("source"),
        "issuer": item.get("issuer"),
        "title": item.get("title"),
        "url": item.get("url"),
        "published_year": item.get("published_year"),
        "raw_pdf_path": item.get("raw_pdf_path"),
        "url_provenance": item.get("url_provenance"),
        "content": item.get("content") or "",
        "tables": json_list(item.get("tables")),
        "table_count": as_int(item.get("table_count")),
        "table_row_count": as_int(item.get("table_row_count")),
        "table_cell_count": as_int(item.get("table_cell_count")),
        "guideline_seed": json_obj(item.get("guideline_seed")),
        "paper_seed": json_obj(item.get("paper_seed")),
        "direct_extraction": json_obj(direct),
        "raw_record": json_obj(item),
    }
