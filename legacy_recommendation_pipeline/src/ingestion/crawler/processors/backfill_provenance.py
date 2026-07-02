"""爬虫后处理文件：把下载或抓取到的原始内容整理成 origin JSONL、PDF 清单或可追溯来源记录。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

from src.ingestion.crawler.common.jsonl import iter_jsonl, write_jsonl
from src.ingestion.crawler.processors.origin import find_source_dir


logger = logging.getLogger(__name__)

DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
URL_RE = re.compile(r"https?://[^\s<>\]\)\"']+", re.I)
DOI_SOURCES = {"aasm", "esc", "jacc"}
SOURCE_DOMAINS: dict[str, tuple[str, ...]] = {}
URL_BLOCKLIST = (
    "creativecommons.org",
    "elsevier.com/about",
    "adobe.com",
    "agreetrust.org",
    "www.w3.org",
)


def backfill_source(source: str, raw_root: str | Path, origin_root: str | Path, dry_run: bool = False) -> dict[str, int]:
    origin_path = Path(origin_root) / f"{source}_origin.jsonl"
    if not origin_path.exists():
        logger.warning("Skip missing origin file: %s", origin_path)
        return {"rows": 0, "filled_url": 0, "raw_path": 0, "not_found": 0}

    rows = list(iter_jsonl(origin_path))
    pdf_items = list(iter_source_pdf_items(source, raw_root=raw_root))
    enriched: list[dict[str, Any]] = []
    stats = {"rows": len(rows), "filled_url": 0, "raw_path": 0, "not_found": 0}

    for row, item in zip_rows_to_pdf_items(source, rows, pdf_items):
        updated = dict(row)
        pdf_path = str(item.get("pdf_path", ""))
        if pdf_path:
            updated["raw_pdf_path"] = pdf_path
            stats["raw_path"] += 1

        current_url = str(updated.get("url", "")).strip()
        current_provenance = str(updated.get("url_provenance", "")).strip()
        should_recompute = current_provenance in {"doi_from_text", "cdc_stacks_from_text"} and (
            source in DOI_SOURCES or source == "cdc"
        )
        if current_url and not should_recompute:
            updated["url_provenance"] = current_provenance or item.get("url_provenance") or "existing"
        else:
            inferred_url, provenance = infer_source_url(source, updated, item)
            updated["url"] = inferred_url
            updated["url_provenance"] = provenance
            if inferred_url:
                stats["filled_url"] += 1
            else:
                stats["not_found"] += 1
        enriched.append(updated)

    if len(enriched) != len(rows):
        logger.warning("%s: mapped %s/%s origin rows", source, len(enriched), len(rows))
    if not dry_run:
        write_jsonl(origin_path, enriched)
    return stats


def iter_source_pdf_items(source: str, raw_root: str | Path) -> Iterable[dict[str, Any]]:
    source_dir = find_source_dir(source, raw_root=raw_root)
    metadata_path = source_dir / "metadata.jsonl"
    metadata = list(iter_jsonl(metadata_path))
    if metadata:
        for item in metadata:
            path = item.get("pdf_path") or item.get("path") or ""
            yield {
                **item,
                "pdf_path": str(path),
                "url_provenance": "metadata",
            }
        return

    for pdf_path in sorted(source_dir.rglob("*.pdf")):
        yield {
            "pdf_path": str(pdf_path),
            "title": pdf_path.stem.replace("_", " "),
            "url": "",
            "landing_url": "",
        }


def zip_rows_to_pdf_items(
    source: str,
    rows: list[dict[str, Any]],
    pdf_items: list[dict[str, Any]],
) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    if len(rows) == len(pdf_items):
        yield from zip(rows, pdf_items)
        return

    by_url: dict[str, dict[str, Any]] = {}
    by_title: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for item in pdf_items:
        url = str(item.get("url") or item.get("landing_url") or "").strip()
        if url:
            by_url[url] = item
        title_key = normalize_key(str(item.get("title", "")))
        if title_key:
            by_title[title_key].append(item)

    remaining = deque(pdf_items)
    for row in rows:
        item = None
        row_url = str(row.get("url", "")).strip()
        if row_url:
            item = by_url.get(row_url)
        if item is None:
            item = _pop_title_match(by_title, str(row.get("title", "")))
        if item is None:
            item = _pop_existing_path_match(remaining, str(row.get("title", "")))
        if item is None:
            logger.warning("%s: could not map origin row to raw PDF: %s", source, row.get("title", ""))
            item = {}
        yield row, item


def _pop_title_match(by_title: dict[str, deque[dict[str, Any]]], title: str) -> dict[str, Any] | None:
    title_key = normalize_key(title)
    if title_key in by_title and by_title[title_key]:
        return by_title[title_key].popleft()
    for key, values in by_title.items():
        if values and (title_key in key or key in title_key):
            return values.popleft()
    return None


def _pop_existing_path_match(items: deque[dict[str, Any]], title: str) -> dict[str, Any] | None:
    title_key = normalize_key(title)
    for item in list(items):
        path_key = normalize_key(Path(str(item.get("pdf_path", ""))).stem)
        if title_key and (title_key in path_key or path_key in title_key):
            items.remove(item)
            return item
    return items.popleft() if items else None


def infer_source_url(source: str, row: dict[str, Any], item: dict[str, Any]) -> tuple[str, str]:
    item_url = str(item.get("url") or item.get("landing_url") or "").strip()
    if item_url:
        return item_url, "metadata"

    text = " ".join(
        str(value or "")
        for value in (
            row.get("title"),
            item.get("title"),
            row.get("content"),
            Path(str(item.get("pdf_path", ""))).stem,
        )
    )

    if source in DOI_SOURCES:
        doi = first_doi(text)
        if doi:
            return f"https://doi.org/{doi}", "doi_from_text"

    domain_url = first_source_domain_url(source, text)
    if domain_url:
        return domain_url, "source_domain_from_text"

    return "", "not_found"


def first_doi(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text)
    for match in DOI_RE.finditer(normalized):
        doi = match.group(0).rstrip(".,;:")
        if len(doi) > 10:
            return doi
    return ""


def first_source_domain_url(source: str, text: str) -> str:
    domains = SOURCE_DOMAINS.get(source, ())
    if not domains:
        return ""
    normalized = re.sub(r"\s+", " ", text)
    for match in URL_RE.finditer(normalized):
        url = match.group(0).rstrip(".,;:")
        lower = url.lower()
        if any(blocked in lower for blocked in URL_BLOCKLIST):
            continue
        if any(domain in lower for domain in domains):
            if source == "kdigo" and "guideline" not in lower and "kdigo.org" in lower:
                continue
            return url
    return ""


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill raw PDF provenance and source URLs in *_origin.jsonl.")
    parser.add_argument("sources", nargs="+")
    parser.add_argument("--raw-root", default="data/raw_pdf")
    parser.add_argument("--origin-root", default="data/origin")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s %(name)s: %(message)s")
    for source in args.sources:
        stats = backfill_source(source.lower(), args.raw_root, args.origin_root, dry_run=args.dry_run)
        logger.info("%s: %s", source, stats)


if __name__ == "__main__":
    main()

