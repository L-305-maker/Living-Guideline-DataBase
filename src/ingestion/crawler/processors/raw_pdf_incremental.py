from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable

from src.ingestion.crawler.common.jsonl import append_jsonl, iter_jsonl, write_jsonl
from src.ingestion.crawler.common.text import extract_year, normalize_space
from src.ingestion.crawler.processors.origin import parse_pdf_with_timeout


logger = logging.getLogger(__name__)

PATH_FIELDS = ("raw_pdf_path", "pdf_path")


def path_key(value: Any, cwd: Path | None = None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    cwd = cwd or Path.cwd()
    path = Path(text)
    if not path.is_absolute():
        path = cwd / path
    return str(path.resolve(strict=False)).replace("\\", "/").lower()


def display_path(path: Path, cwd: Path | None = None) -> str:
    cwd = cwd or Path.cwd()
    try:
        return str(path.resolve(strict=False).relative_to(cwd.resolve(strict=False)))
    except ValueError:
        return str(path)


def source_from_pdf_path(pdf_path: Path, raw_root: Path) -> str:
    try:
        first = pdf_path.relative_to(raw_root).parts[0]
    except ValueError:
        first = pdf_path.parent.name
    source = first.lower()
    for suffix in ("_pdfs", "_pdf"):
        if source.endswith(suffix):
            return source[: -len(suffix)]
    return source


def iter_raw_pdfs(raw_root: str | Path) -> Iterable[Path]:
    root = Path(raw_root)
    yield from sorted(root.rglob("*.pdf"))


def record_path_keys(record: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for field in PATH_FIELDS:
        key = path_key(record.get(field))
        if key:
            keys.add(key)
    direct = record.get("direct_extraction")
    if isinstance(direct, dict):
        key = path_key(direct.get("raw_pdf_path"))
        if key:
            keys.add(key)
    return keys


def load_recorded_pdf_paths(paths: Iterable[str | Path]) -> set[str]:
    recorded: set[str] = set()
    for path in paths:
        jsonl_path = Path(path)
        if not jsonl_path.exists():
            continue
        for record in iter_jsonl(jsonl_path):
            recorded.update(record_path_keys(record))
    return recorded


def origin_inputs(origin_root: str | Path, output_path: str | Path | None = None) -> list[Path]:
    root = Path(origin_root)
    output = Path(output_path).resolve(strict=False) if output_path else None
    inputs: list[Path] = []
    for path in sorted(root.glob("*_origin.jsonl")):
        if output and path.resolve(strict=False) == output:
            continue
        inputs.append(path)
    return inputs


def build_origin_record(pdf_path: Path, raw_root: Path, parse_timeout: int | None = None) -> dict[str, Any]:
    parsed = parse_pdf_with_timeout(pdf_path, parse_timeout)
    source = source_from_pdf_path(pdf_path, raw_root)
    title = normalize_space(str(parsed.get("title", ""))) or pdf_path.stem
    raw_pdf_path = display_path(pdf_path)
    year = normalize_space(str(parsed.get("published_year", ""))) or extract_year(title, raw_pdf_path)
    return {
        "content": parsed.get("content", ""),
        "title": title,
        "url": "",
        "published_year": year,
        "source": source,
        "tables": parsed.get("tables", []),
        "raw_pdf_path": raw_pdf_path,
        "pdf_metadata": parsed.get("pdf_metadata", {}),
        "origin_processing": {
            "method": "raw_pdf_incremental",
            "raw_root": display_path(raw_root),
        },
    }


def process_missing_pdfs(
    raw_root: str | Path = "data/raw_pdf",
    origin_root: str | Path = "data/origin",
    output_path: str | Path = "data/origin/raw_pdf_incremental_origin.jsonl",
    failures_output: str | Path = "data/origin/raw_pdf_incremental_failures.jsonl",
    summary_output: str | Path = "data/origin/raw_pdf_incremental_summary.jsonl",
    limit: int | None = None,
    parse_timeout: int | None = None,
    progress_every: int = 25,
    retry_failures: bool = False,
    source: str | None = None,
) -> dict[str, Any]:
    """增量解析尚未出现在 origin JSONL 输出中的原始 PDF。

    参数较多是因为这里直接映射 CLI 选项。后续可把路径设置和运行选项
    收敛成一个小配置对象，降低调用方的参数负担。
    """

    raw_root_path = Path(raw_root)
    output = Path(output_path)
    failures = Path(failures_output)

    existing_inputs = origin_inputs(origin_root)
    recorded_paths = load_recorded_pdf_paths(existing_inputs)
    recorded_paths.update(load_recorded_pdf_paths([output]))
    if not retry_failures:
        recorded_paths.update(load_recorded_pdf_paths([failures]))

    source_filter = str(source or "").strip().lower()
    raw_pdfs = [
        path
        for path in iter_raw_pdfs(raw_root_path)
        if not source_filter or source_from_pdf_path(path, raw_root_path) == source_filter
    ]
    missing = [path for path in raw_pdfs if path_key(path) not in recorded_paths]
    if limit is not None:
        missing = missing[:limit]

    processed = 0
    failed = 0
    for index, pdf_path in enumerate(missing, 1):
        try:
            append_jsonl(output, build_origin_record(pdf_path, raw_root_path, parse_timeout=parse_timeout))
            processed += 1
        except Exception as exc:
            failed += 1
            append_jsonl(
                failures,
                {
                    "raw_pdf_path": display_path(pdf_path),
                    "source": source_from_pdf_path(pdf_path, raw_root_path),
                    "error": repr(exc),
                    "origin_processing": {"method": "raw_pdf_incremental"},
                },
            )
            logger.warning("Failed to parse %s: %s", pdf_path, exc)
        if progress_every > 0 and index % progress_every == 0:
            logger.info("Processed %s/%s missing PDFs; success=%s failed=%s", index, len(missing), processed, failed)

    summary = {
        "raw_pdf_total": len(raw_pdfs),
        "already_recorded_pdf_paths": len(recorded_paths),
        "selected_missing_pdfs": len(missing),
        "processed": processed,
        "failed": failed,
        "output": str(output),
        "failures_output": str(failures),
        "parse_timeout": parse_timeout,
        "limit": limit,
        "source": source_filter,
    }
    write_jsonl(summary_output, [summary])
    return summary


def dedup_key(record: dict[str, Any]) -> str:
    keys = sorted(record_path_keys(record))
    if keys:
        return "path:" + keys[0]
    source = str(record.get("source") or "").strip().lower()
    url = str(record.get("url") or "").strip().lower()
    title = str(record.get("title") or "").strip().lower()
    year = str(record.get("published_year") or "").strip()
    return "meta:" + json.dumps([source, url, title, year], ensure_ascii=False)


def merge_origin_files(
    origin_root: str | Path = "data/origin",
    output_path: str | Path = "data/origin/all_origin_merged.jsonl",
    summary_output: str | Path = "data/origin/all_origin_merged_summary.jsonl",
) -> dict[str, Any]:
    output = Path(output_path)
    inputs = origin_inputs(origin_root, output_path=output)
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    duplicate_count = 0
    source_counts: dict[str, int] = {}
    for input_path in inputs:
        for record in iter_jsonl(input_path):
            key = dedup_key(record)
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            merged.append(record)
            source = str(record.get("source") or "unknown").lower()
            source_counts[source] = source_counts.get(source, 0) + 1

    write_jsonl(output, merged)
    summary = {
        "inputs": [str(path) for path in inputs],
        "merged_records": len(merged),
        "duplicates_skipped": duplicate_count,
        "source_counts": source_counts,
        "output": str(output),
    }
    write_jsonl(summary_output, [summary])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Incrementally parse unrecorded raw PDFs and merge origin JSONL files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process_parser = subparsers.add_parser("process-missing", help="Parse raw PDFs missing from origin outputs.")
    process_parser.add_argument("--raw-root", default="data/raw_pdf")
    process_parser.add_argument("--origin-root", default="data/origin")
    process_parser.add_argument("--output", default="data/origin/raw_pdf_incremental_origin.jsonl")
    process_parser.add_argument("--failures-output", default="data/origin/raw_pdf_incremental_failures.jsonl")
    process_parser.add_argument("--summary-output", default="data/origin/raw_pdf_incremental_summary.jsonl")
    process_parser.add_argument("--limit", type=int, default=None)
    process_parser.add_argument("--parse-timeout", type=int, default=None)
    process_parser.add_argument("--progress-every", type=int, default=25)
    process_parser.add_argument("--retry-failures", action="store_true")
    process_parser.add_argument("--source", default=None, help="Optional normalized source filter, e.g. pmc for data/raw_pdf/pmc_pdf.")

    merge_parser = subparsers.add_parser("merge-origin", help="Merge *_origin.jsonl files into one deduplicated origin file.")
    merge_parser.add_argument("--origin-root", default="data/origin")
    merge_parser.add_argument("--output", default="data/origin/all_origin_merged.jsonl")
    merge_parser.add_argument("--summary-output", default="data/origin/all_origin_merged_summary.jsonl")

    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s %(name)s: %(message)s")
    if args.command == "process-missing":
        summary = process_missing_pdfs(
            raw_root=args.raw_root,
            origin_root=args.origin_root,
            output_path=args.output,
            failures_output=args.failures_output,
            summary_output=args.summary_output,
            limit=args.limit,
            parse_timeout=args.parse_timeout,
            progress_every=args.progress_every,
            retry_failures=args.retry_failures,
            source=args.source,
        )
        print(json.dumps(summary, ensure_ascii=False))
        return
    if args.command == "merge-origin":
        summary = merge_origin_files(origin_root=args.origin_root, output_path=args.output, summary_output=args.summary_output)
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
