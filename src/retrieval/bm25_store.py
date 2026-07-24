"""Pure-Python BM25 indexes for document cards, document views, and chunks."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from src.retrieval.chunk_normalizer import iter_normalized_chunks
from src.retrieval.common import parse_time_range as helper_parse_time_range
from src.retrieval.document_repr.builder import build_document_card, build_document_views
from src.utils.records import record_text as helper_record_text
from src.utils.text import tokenize_search_text as tokenize
from src.utils.io import DATA_DIR, ensure_dir, iter_markdown_files, read_jsonl






def helper_in_time_range(publication_date: str, time_range: str | dict[str, str] | None) -> bool:
    start, end = helper_parse_time_range(time_range)
    if not start and not end:
        return True
    value = publication_date or "unknown"
    if value == "unknown":
        return False
    if start and value < start:
        return False
    if end and value > end:
        return False
    return True


class BM25Store:
    def __init__(self, records: list[dict[str, Any]], text_field: str, id_field: str) -> None:
        self.records = records
        self.text_field = text_field
        self.id_field = id_field
        self.tokenized = [tokenize(helper_record_text(record, text_field)) for record in records]
        self.avgdl = sum(len(tokens) for tokens in self.tokenized) / max(1, len(self.tokenized))
        self.df: Counter[str] = Counter()
        for tokens in self.tokenized:
            self.df.update(set(tokens))

    def search(
        self,
        query: str,
        top_n: int = 50,
        source_institution: str | None = None,
        clinical_department: str | None = None,
        time_range: str | dict[str, str] | None = None,
        exclude_reference_sections: bool = False,
    ) -> list[tuple[str, float]]:
        # 元数据条件先过滤候选，BM25 只对合格记录计分，并用稳定标识打破同分排序。
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        results: list[tuple[str, float]] = []
        n_docs = max(1, len(self.records))
        for record, tokens in zip(self.records, self.tokenized):
            if source_institution and source_institution.lower() not in (record.get("source_institution") or "").lower():
                continue
            if clinical_department and clinical_department.lower() not in (record.get("clinical_department") or "").lower():
                continue
            if not helper_in_time_range(record.get("publication_date", ""), time_range):
                continue
            if exclude_reference_sections and record.get("is_reference_section"):
                continue
            tf = Counter(tokens)
            dl = len(tokens) or 1
            score = 0.0
            for token in query_tokens:
                if token not in tf:
                    continue
                idf = math.log(1 + (n_docs - self.df[token] + 0.5) / (self.df[token] + 0.5))
                freq = tf[token]
                score += idf * (freq * 2.2) / (freq + 1.2 * (1 - 0.75 + 0.75 * dl / max(self.avgdl, 1)))
            if score > 0:
                results.append((record[self.id_field], score))
        results.sort(key=lambda item: (-item[1], item[0]))
        return results[:top_n]




def helper_card_records(clean_dir: str | Path, cards_path: str | Path | None = None) -> list[dict[str, Any]]:
    path = Path(cards_path) if cards_path else None
    if path and path.exists():
        return list(read_jsonl(path))
    records = []
    for markdown_path in iter_markdown_files(clean_dir):
        markdown = markdown_path.read_text(encoding="utf-8", errors="replace")
        records.append(build_document_card(markdown, markdown_path))
    return records


def helper_view_records(clean_dir: str | Path, views_path: str | Path | None = None) -> list[dict[str, Any]]:
    path = Path(views_path) if views_path else None
    if path and path.exists():
        return list(read_jsonl(path))
    views: list[dict[str, Any]] = []
    for card in helper_card_records(clean_dir):
        views.extend(build_document_views(card))
    return views


def helper_chunk_records(chunks_path: str | Path) -> list[dict[str, Any]]:
    path = Path(chunks_path)
    data_dir = path.parent.parent
    return list(iter_normalized_chunks(data_dir, path))


def build_bm25_indexes(
    clean_dir: str | Path = DATA_DIR / "markdown_clean",
    chunks_path: str | Path = DATA_DIR / "chunks" / "all_chunks.jsonl",
    output_dir: str | Path = DATA_DIR / "index",
) -> dict[str, Any]:
    out = ensure_dir(output_dir)
    data_dir = Path(output_dir).parent
    cards = helper_card_records(clean_dir, data_dir / "document_cards.jsonl")
    views = helper_view_records(clean_dir, data_dir / "document_views.jsonl")
    chunks = helper_chunk_records(chunks_path) if Path(chunks_path).exists() else []
    cards_payload = {"kind": "document_cards", "records": cards}
    views_payload = {"kind": "document_views", "records": views}
    chunks_payload = {"kind": "chunks", "records": chunks}
    (out / "bm25_document_cards.json").write_text(json.dumps(cards_payload, ensure_ascii=False), encoding="utf-8")
    (out / "bm25_document_views.json").write_text(json.dumps(views_payload, ensure_ascii=False), encoding="utf-8")
    (out / "bm25_chunks.json").write_text(json.dumps(chunks_payload, ensure_ascii=False), encoding="utf-8")
    return {"document_cards": len(cards), "document_views": len(views), "chunks": len(chunks), "index_dir": str(out)}


def load_store(path: str | Path) -> BM25Store:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload["records"]
    if payload.get("kind") == "documents":
        raise ValueError("Legacy document BM25 indexes are not supported; use document_cards/document_views.")
    if payload.get("kind") == "document_cards":
        return BM25Store(records, "card_text", "doc_id")
    if payload.get("kind") == "document_views":
        return BM25Store(records, "text", "view_id")
    return BM25Store(records, "retrieval_text", "chunk_id")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", default=str(DATA_DIR / "markdown_clean"))
    parser.add_argument("--chunks", default=str(DATA_DIR / "chunks" / "all_chunks.jsonl"))
    parser.add_argument("--output-dir", default=str(DATA_DIR / "index"))
    args = parser.parse_args()
    print(json.dumps(build_bm25_indexes(args.clean_dir, args.chunks, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
