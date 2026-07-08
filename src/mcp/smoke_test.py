"""Smoke-test Search, Read, and Retrieve APIs behind the MCP tools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.mcp.api_read import read
from src.mcp.api_retrieve import retrieve
from src.mcp.api_search import search


def smoke_test(data_dir: str | Path = "data/evidence") -> dict[str, object]:
    search_results = search(
        {
            "query": "糖尿病 胰岛素",
            "time_range": "2012-2026",
            "topk": 3,
            "recency_boost": True,
        },
        data_dir=data_dir,
    )
    if not search_results:
        raise RuntimeError("search returned no results")

    doc = read({"doc_id": search_results[0]["doc_id"], "max_chars": 500}, data_dir=data_dir)
    chunks = retrieve(
        {
            "query": "糖尿病患者胰岛素治疗如何管理",
            "time_range": "2012-2026",
            "topk": 3,
        },
        data_dir=data_dir,
    )
    if not chunks:
        raise RuntimeError("retrieve returned no chunks")

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
    parser.add_argument("--data-dir", default="data/evidence")
    args = parser.parse_args()
    print(json.dumps(smoke_test(args.data_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
