"""Rebuild damaged evidence documents into an isolated candidate dataset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.cleaning.chunker import chunk_all
from src.pipeline.cleaning.cleaner import clean_all
from src.pipeline.ocr.baidu_ocr import BaiduQuotaExceeded, get_access_token, ocr_pdf_to_markdown
from src.pipeline.ocr.deepseek_ocr import (
    DeepSeekOCRError,
    helper_api_key as get_deepseek_api_key,
    ocr_pdf_to_markdown as deepseek_ocr_pdf_to_markdown,
)
from src.retrieval.document_repr import build_document_representations
from src.utils.front_matter import parse_front_matter
from src.utils.io import read_jsonl

DEFAULT_MISSING_REPORT = Path("data/reports/missing_full_content_candidates_20260720.csv")
DEFAULT_MOJIBAKE_REPORT = Path("outputs/markdown_clean_mojibake_files_20260719_complete.csv")
DEFAULT_SOURCE_INVENTORY = Path("data/evidence/source_pdf_inventory.jsonl")
DEFAULT_EVIDENCE_DIR = Path("data/evidence")
DEFAULT_OUTPUT_DIR = Path("data/evidence_candidate")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def pdf_index() -> dict[str, list[Path]]:
    by_name: dict[str, list[Path]] = {}
    for root in (PROJECT_ROOT / "data/raw_pdf", PROJECT_ROOT / "data/consensus"):
        if not root.exists():
            continue
        for path in root.rglob("*.pdf"):
            if "quarantine" in {part.casefold() for part in path.parts}:
                continue
            by_name.setdefault(path.name.casefold(), []).append(path.resolve())
    return by_name


def resolve_source(value: str, by_name: dict[str, list[Path]]) -> Path | None:
    path = Path(value or "")
    for candidate in (path, PROJECT_ROOT / path):
        if candidate.is_file():
            return candidate.resolve()
    matches = by_name.get(path.name.casefold(), [])
    return matches[0] if len(matches) == 1 else None


def load_active_candidates(
    missing_report: Path,
    mojibake_report: Path,
    evidence_dir: Path,
    cache_dir: Path,
    ocr_provider: str,
    by_name: dict[str, list[Path]],
) -> list[dict[str, Any]]:
    reasons: dict[str, set[str]] = {}
    for row in read_csv(missing_report):
        reasons.setdefault(row["markdown_filename"], set()).add(f"missing_full_content:{row['criterion']}")
    for row in read_csv(mojibake_report):
        reasons.setdefault(row["name"], set()).add(f"mojibake:{row['classification']}")

    cache_by_suffix: dict[str, list[Path]] = {}
    cache_namespace = "accurate_position" if ocr_provider == "baidu" else "deepseek_ocr_2"
    cache_root = cache_dir / cache_namespace
    if cache_root.exists():
        for path in cache_root.iterdir():
            if path.is_dir():
                cache_by_suffix.setdefault(path.name.rsplit("_", 1)[-1].casefold(), []).append(path)

    rows: list[dict[str, Any]] = []
    for name, row_reasons in sorted(reasons.items()):
        clean_path = evidence_dir / "markdown_clean" / name
        if not clean_path.is_file():
            raise FileNotFoundError(f"Candidate Markdown does not exist: {clean_path}")
        metadata, _body = parse_front_matter(clean_path.read_text(encoding="utf-8", errors="replace"))
        doc_id = str(metadata.get("id") or clean_path.stem)
        pages = int(str(metadata.get("pdf_page_count") or 0))
        source = resolve_source(str(metadata.get("source_file") or ""), by_name)
        cache_path = cache_root / doc_id
        if ocr_provider == "baidu" and not cache_path.is_dir():
            matches = cache_by_suffix.get(doc_id.rsplit("_", 1)[-1].casefold(), [])
            cache_path = matches[0] if len(matches) == 1 else cache_path
        cached_pages = len(list(cache_path.glob("*.json")))
        rows.append(
            {
                "candidate_type": "repair_document",
                "doc_id": doc_id,
                "markdown_filename": name,
                "source_institution": str(metadata.get("source_institution") or "Unknown"),
                "title": str(metadata.get("title") or ""),
                "source_file": str(source or metadata.get("source_file") or ""),
                "pdf_page_count": pages,
                "cached_pages": cached_pages,
                "cache_doc_id": cache_path.name,
                "reasons": sorted(row_reasons),
                "action": (
                    "rebuild_from_ocr_cache"
                    if pages > 0 and cached_pages >= pages
                    else f"pending_{ocr_provider}_ocr"
                ),
                "canonical_doc_id": "",
                "source_resolved": source is not None,
                "metadata": metadata,
            }
        )
    return rows


def load_not_in_markdown_clean(inventory_path: Path, by_name: dict[str, list[Path]]) -> list[dict[str, Any]]:
    actions = {
        "duplicate_pdf_content": "reuse_canonical_skip_reextract",
        "duplicate_clean_body": "reuse_canonical_skip_reextract",
        "excluded_clear_non_guideline": "exclude_non_guideline",
        "review_required": "pending_content_review",
        "review_required_no_prior_review": "pending_initial_review",
    }
    rows: list[dict[str, Any]] = []
    for item in read_jsonl(inventory_path):
        status = str(item.get("status") or "")
        if status == "active":
            continue
        source_text = str(item.get("source_file") or "")
        source = resolve_source(source_text, by_name)
        review_status = str(item.get("review_status") or "")
        doc_ids = item.get("doc_ids") or []
        rows.append(
            {
                "candidate_type": "not_in_markdown_clean",
                "doc_id": str(doc_ids[0]) if doc_ids else "",
                "markdown_filename": "",
                "source_institution": "",
                "title": Path(source_text).stem,
                "source_file": str(source or source_text),
                "pdf_page_count": "",
                "cached_pages": "",
                "reasons": [f"not_in_markdown_clean:{status}", f"review_status:{review_status}"],
                "action": actions.get(status, "skip_unclassified"),
                "canonical_doc_id": str(item.get("canonical_doc_id") or ""),
                "source_resolved": source is not None,
                "review_status": review_status,
                "inventory_status": status,
                "decision_source": str(item.get("decision_source") or ""),
            }
        )
    return sorted(rows, key=lambda row: row["source_file"].casefold())


def write_inventory(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "candidate_type", "doc_id", "markdown_filename", "source_institution", "title", "source_file",
        "pdf_page_count", "cached_pages", "reasons", "action", "canonical_doc_id", "source_resolved",
        "review_status", "inventory_status", "decision_source",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {field: row.get(field, "") for field in fields}
            output["reasons"] = "|".join(row.get("reasons") or [])
            writer.writerow(output)


def rebuild_raw(
    candidates: list[dict[str, Any]],
    output_dir: Path,
    cache_dir: Path,
    ocr_provider: str,
    execute_ocr: bool,
    max_api_pages: int,
    ocr_workers: int,
) -> dict[str, Any]:
    raw_dir = output_dir / "markdown_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    credential: str | None = None
    quota_exhausted = False
    provider_halted = False
    api_page_budget_used = 0
    completed = cached = api_calls = 0
    pending: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    expected_engine = "baidu_accurate_position" if ocr_provider == "baidu" else "infini_deepseek_ocr_2"

    for row in candidates:
        source = Path(row["source_file"])
        output = raw_dir / f"{row['doc_id']}.md"
        pages = int(row["pdf_page_count"])
        missing_pages = max(0, pages - int(row["cached_pages"]))
        if output.is_file():
            output_metadata, _body = parse_front_matter(output.read_text(encoding="utf-8", errors="replace"))
            if output_metadata.get("ocr_engine") == expected_engine:
                completed += 1
                continue
        if not row["source_resolved"]:
            failed.append({"doc_id": row["doc_id"], "error": "source_pdf_not_resolved"})
            continue
        budget_exhausted = (
            api_page_budget_used >= max_api_pages
            if ocr_provider == "deepseek"
            else api_page_budget_used + missing_pages > max_api_pages
        )
        if missing_pages and (not execute_ocr or quota_exhausted or provider_halted or budget_exhausted):
            pending.append({"doc_id": row["doc_id"], "missing_pages": missing_pages})
            continue
        try:
            metadata = dict(row["metadata"])
            metadata.pop("cleaning_quality", None)
            metadata.pop("cleaning_flags", None)
            if ocr_provider == "deepseek":
                if missing_pages and credential is None:
                    credential = get_deepseek_api_key()
                result = deepseek_ocr_pdf_to_markdown(
                    source,
                    output,
                    doc_id=row["doc_id"],
                    metadata=metadata,
                    cache_dir=cache_dir,
                    api_key=credential,
                    max_api_pages=max_api_pages - api_page_budget_used,
                    max_workers=ocr_workers,
                )
                if result["complete"]:
                    completed += 1
                else:
                    pending_row = {"doc_id": row["doc_id"], "missing_pages": result["missing_pages"]}
                    if result.get("errors"):
                        pending_row["error"] = " | ".join(result["errors"])
                    pending.append(pending_row)
            else:
                if missing_pages and credential is None:
                    credential = get_access_token()
                result = ocr_pdf_to_markdown(
                    source,
                    output,
                    doc_id=row["doc_id"],
                    metadata=metadata,
                    cache_dir=cache_dir,
                    cache_doc_id=row["cache_doc_id"],
                    access_token=credential or "cache-only",
                    sleep_seconds=0.5 if missing_pages else 0,
                )
                completed += 1
            cached += int(result["cached_pages"])
            api_calls += int(result["api_calls"])
            api_page_budget_used += int(result["api_calls"])
        except BaiduQuotaExceeded as exc:
            quota_exhausted = True
            pending.append({"doc_id": row["doc_id"], "missing_pages": missing_pages, "error": str(exc)})
        except DeepSeekOCRError as exc:
            provider_halted = True
            failed.append({"doc_id": row["doc_id"], "error": str(exc)[:800]})
        except Exception as exc:  # noqa: BLE001
            failed.append({"doc_id": row["doc_id"], "error": str(exc)[:800]})
    return {
        "completed_raw": completed,
        "cached_pages_read": cached,
        "api_calls": api_calls,
        "pending_documents": len(pending),
        "pending_pages": sum(int(row["missing_pages"]) for row in pending),
        "failed": failed,
        "quota_exhausted": quota_exhausted,
        "provider_halted": provider_halted,
        "pending": pending,
    }


def build_artifacts(output_dir: Path) -> dict[str, Any]:
    raw_dir = output_dir / "markdown_raw"
    clean_dir = output_dir / "markdown_clean"
    stages: dict[str, Any] = {}
    stages["clean"] = clean_all(raw_dir, clean_dir, output_dir / "documents.jsonl", progress_every=100)
    stages["document_representations"] = build_document_representations(clean_dir, output_dir)
    stages["chunks"] = chunk_all(clean_dir, output_dir / "chunks")
    return stages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--missing-report", type=Path, default=DEFAULT_MISSING_REPORT)
    parser.add_argument("--mojibake-report", type=Path, default=DEFAULT_MOJIBAKE_REPORT)
    parser.add_argument("--source-inventory", type=Path, default=DEFAULT_SOURCE_INVENTORY)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--ocr-provider", choices=("baidu", "deepseek"), default="baidu")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--execute-ocr", action="store_true")
    parser.add_argument("--max-api-pages", type=int, default=0)
    parser.add_argument("--ocr-workers", type=int, default=1)
    parser.add_argument("--skip-build-artifacts", action="store_true")
    args = parser.parse_args()
    if args.execute_ocr and not args.apply:
        parser.error("--execute-ocr requires --apply")
    if args.max_api_pages < 0:
        parser.error("--max-api-pages must be >= 0")
    if args.execute_ocr and args.max_api_pages == 0:
        parser.error("--execute-ocr requires --max-api-pages > 0")
    if args.ocr_workers < 1:
        parser.error("--ocr-workers must be >= 1")
    if args.execute_ocr and args.ocr_provider == "deepseek":
        get_deepseek_api_key()

    evidence_dir = resolve_path(args.evidence_dir)
    output_dir = resolve_path(args.output_dir)
    default_cache = evidence_dir / "baidu_ocr_pages" if args.ocr_provider == "baidu" else output_dir / "ocr_cache"
    cache_dir = resolve_path(args.cache_dir or default_cache)
    by_name = pdf_index()
    active = load_active_candidates(
        resolve_path(args.missing_report),
        resolve_path(args.mojibake_report),
        evidence_dir,
        cache_dir,
        args.ocr_provider,
        by_name,
    )
    not_in_clean = load_not_in_markdown_clean(resolve_path(args.source_inventory), by_name)
    inventory = active + not_in_clean
    write_inventory(output_dir / "candidate_inventory.csv", inventory)
    summary: dict[str, Any] = {
        "apply": args.apply,
        "ocr_provider": args.ocr_provider,
        "repair_documents": len(active),
        "not_in_markdown_clean": len(not_in_clean),
        "total_candidates": len(inventory),
        "reason_counts": dict(Counter(reason for row in inventory for reason in row["reasons"])),
        "action_counts": dict(Counter(row["action"] for row in inventory)),
        "not_in_markdown_clean_status_counts": dict(Counter(row["inventory_status"] for row in not_in_clean)),
        "not_in_markdown_clean_review_counts": dict(Counter(row["review_status"] for row in not_in_clean)),
        "ocr_pages_total": sum(int(row["pdf_page_count"]) for row in active),
        "ocr_pages_cached": sum(min(int(row["pdf_page_count"]), int(row["cached_pages"])) for row in active),
    }
    if args.apply:
        summary["reextract"] = rebuild_raw(
            active,
            output_dir,
            cache_dir,
            args.ocr_provider,
            args.execute_ocr,
            args.max_api_pages,
            args.ocr_workers,
        )
        if not args.skip_build_artifacts:
            summary["stages"] = build_artifacts(output_dir)
    (output_dir / "candidate_rebuild_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
