"""爬虫后处理文件：把下载或抓取到的原始内容整理成 origin JSONL、PDF 清单或可追溯来源记录。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import tempfile
from itertools import islice
from pathlib import Path
from typing import Iterable

from src.ingestion.crawler.common.jsonl import iter_jsonl, write_jsonl
from src.ingestion.crawler.common.pdf_parser import parse_pdf
from src.ingestion.crawler.common.text import extract_year, normalize_space


logger = logging.getLogger(__name__)


def iter_origin_records(
    source: str,
    raw_root: str | Path = "data/raw_pdf",
    parse_timeout: int | None = None,
) -> Iterable[dict[str, object]]:
    for item in iter_source_pdfs(source, raw_root=raw_root):
        pdf_path = Path(str(item.get("pdf_path", "")))
        if not pdf_path.exists():
            logger.warning("Skip missing PDF for %s: %s", source, pdf_path)
            continue
        try:
            parsed = parse_pdf_with_timeout(pdf_path, parse_timeout)
        except Exception as exc:
            logger.exception("Failed to parse PDF %s: %s", pdf_path, exc)
            continue

        title = normalize_space(str(parsed.get("title", ""))) or normalize_space(str(item.get("title", ""))) or pdf_path.stem
        year = (
            normalize_space(str(item.get("published_year", "")))
            or normalize_space(str(parsed.get("published_year", "")))
            or extract_year(title, str(item.get("url", "")))
        )
        yield {
            "content": parsed.get("content", ""),
            "title": title,
            "url": item.get("url", "") or item.get("landing_url", ""),
            "published_year": year,
            "source": source,
            "tables": parsed.get("tables", []),
        }


def iter_source_pdfs(source: str, raw_root: str | Path = "data/raw_pdf") -> Iterable[dict[str, object]]:
    source_dir = find_source_dir(source, raw_root=raw_root)
    metadata_path = source_dir / "metadata.jsonl"
    metadata_items = list(iter_jsonl(metadata_path))
    if metadata_items:
        yield from metadata_items
        return

    # 旧版 NICE/KDIGO/CDC 数据通常位于 data/raw_pdf/<source>_pdf/<guidance_id>/*.pdf，
    # 这些文件可能还没有爬虫生成的 metadata.jsonl。
    for pdf_path in sorted(source_dir.rglob("*.pdf")):
        title = normalize_space(pdf_path.stem.replace("_", " "))
        reference = pdf_path.parent.name if pdf_path.parent != source_dir else ""
        yield {
            "source": source,
            "url": infer_legacy_url(source, reference),
            "title": title,
            "published_year": extract_year(title, str(pdf_path)),
            "pdf_path": str(pdf_path),
            "landing_url": "",
            "reference": reference,
        }


def find_source_dir(source: str, raw_root: str | Path = "data/raw_pdf") -> Path:
    root = Path(raw_root)
    candidates = [
        root / source,
        root / f"{source}_pdf",
        root / f"{source}_pdfs",
        root / source.removesuffix("_pdf"),
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return candidates[0]


def infer_legacy_url(source: str, reference: str) -> str:
    if not reference:
        return ""
    if source == "nice":
        return f"https://www.nice.org.uk/guidance/{reference.lower()}"
    return ""


def process_source(
    source: str,
    raw_root: str | Path = "data/raw_pdf",
    output_root: str | Path = "data/origin",
    limit: int | None = None,
    parse_timeout: int | None = None,
) -> int:
    """把某个来源的 PDF 解析成对应 origin JSONL 文件。"""

    output_path = Path(output_root) / f"{source}_origin.jsonl"
    records = iter_origin_records(source, raw_root=raw_root, parse_timeout=parse_timeout)
    if limit is not None:
        records = islice(records, limit)
    count = write_jsonl(output_path, records)
    logger.info("Wrote %s records to %s", count, output_path)
    return count


def parse_pdf_with_timeout(pdf_path: Path, timeout: int | None) -> dict[str, object]:
    if not timeout:
        return parse_pdf(pdf_path)

    ctx = mp.get_context("spawn")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", prefix="pdf_parse_") as tmp:
        result_path = Path(tmp.name)
    process = ctx.Process(target=_parse_pdf_worker, args=(str(pdf_path), str(result_path)))
    try:
        process.start()
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join(5)
            raise TimeoutError(f"Timed out after {timeout}s")
        if not result_path.exists() or result_path.stat().st_size == 0:
            raise RuntimeError("PDF parser exited without returning a result")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") == "ok":
            return result["payload"]
        raise RuntimeError(str(result.get("payload", "")))
    finally:
        result_path.unlink(missing_ok=True)


def _parse_pdf_worker(pdf_path: str, result_path: str) -> None:
    result_file = Path(result_path)
    try:
        payload = {"status": "ok", "payload": parse_pdf(pdf_path)}
    except Exception as exc:
        payload = {"status": "error", "payload": repr(exc)}
    result_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert downloaded PDFs to *_origin.jsonl using pdfplumber.")
    parser.add_argument("sources", nargs="+", help="Source names, e.g. who uspstf ada sign pmc")
    parser.add_argument("--raw-root", default="data/raw_pdf")
    parser.add_argument("--output-root", default="data/origin")
    parser.add_argument("--limit", type=int, default=None, help="Maximum PDFs to process per source")
    parser.add_argument("--parse-timeout", type=int, default=None, help="Maximum seconds to spend on one PDF")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s %(name)s: %(message)s")
    for source in args.sources:
        process_source(
            source.lower(),
            raw_root=args.raw_root,
            output_root=args.output_root,
            limit=args.limit,
            parse_timeout=args.parse_timeout,
        )


if __name__ == "__main__":
    main()

