"""Smoke-test PostgreSQL Search, Read, and Retrieve APIs behind MCP tools."""

from __future__ import annotations

import argparse
import json

from src.mcp.api_pg import read_pg, retrieve_pg, search_pg


def smoke_test(query: str = "diabetes hypertension guideline", topk: int = 3) -> dict[str, object]:
    search_results = search_pg(
        {
            "query": query,
            "time_range": "2012-2026",
            "topk": topk,
            "recency_boost": True,
        }
    )
    if not search_results:
        raise RuntimeError("search_pg returned no results")

    doc = read_pg({"doc_id": search_results[0]["doc_id"], "max_chars": 500})
    chunks = retrieve_pg(
        {
            "query": query,
            "time_range": "2012-2026",
            "topk": topk,
        }
    )
    if not chunks:
        raise RuntimeError("retrieve_pg returned no chunks")

    return {
        "search_count": len(search_results),
        "search_first": {
            key: search_results[0].get(key)
            for key in ("doc_id", "title", "publication_date", "source_institution", "read_key")
        },
        "search_has_match_reason": bool(search_results[0].get("match_reason")),
        "read_doc_id": doc.get("doc_id"),
        "read_chars": len(doc.get("content", "")),
        "retrieve_count": len(chunks),
        "retrieve_first": {
            key: chunks[0].get(key)
            for key in ("chunk_id", "doc_id", "heading", "prev_chunk_id", "next_chunk_id", "char_start", "char_end")
        },
        "retrieve_has_source_context": bool(chunks[0].get("source_quote_context")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="diabetes hypertension guideline")
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(smoke_test(args.query, args.topk), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
