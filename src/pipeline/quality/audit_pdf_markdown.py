"""Audit whether converted Markdown preserves enough source PDF text."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from src.pipeline.ocr.pdf_quality import inspect_pdf_text_layer
from src.utils.front_matter import parse_front_matter
from src.utils.io import DATA_DIR, ensure_parent, read_jsonl


PAGE_MARKER_RE = re.compile(r"<!--\s*page:\s*\d+\s*-->", re.I)


def _pdf_text_stats(path: Path) -> dict[str, Any]:
    report = inspect_pdf_text_layer(path)
    return {
        "pdf_pages": report.pages,
        "pdf_text_chars": report.text_chars,
        "pdf_text_chars_per_page": report.text_chars_per_page,
        "pdf_low_text_pages": report.low_text_pages,
        "pdf_low_text_page_ratio": report.low_text_page_ratio,
        "pdf_image_pages": report.image_pages,
        "pdf_image_page_ratio": report.image_page_ratio,
        "pdf_scanned_pages": report.scanned_pages,
        "pdf_scanned_page_ratio": report.scanned_page_ratio,
        "pdf_is_scanned": report.is_scanned,
        "pdf_needs_ocr": report.needs_ocr,
        "pdf_text_quality": report.quality,
    }


def _markdown_stats(path: Path) -> dict[str, Any]:
    markdown = path.read_text(encoding="utf-8", errors="replace")
    metadata, body = parse_front_matter(markdown)
    text = re.sub(r"<!--.*?-->", " ", body, flags=re.S)
    text = re.sub(r"#+\s*", " ", text)
    visible_chars = len(re.sub(r"\s+", "", text))
    return {
        "markdown_chars": visible_chars,
        "markdown_page_markers": len(PAGE_MARKER_RE.findall(markdown)),
        "metadata": metadata,
    }


def audit_pdf_markdown(
    data_dir: str | Path = DATA_DIR,
    output_path: str | Path = DATA_DIR / "reports" / "pdf_markdown_audit.jsonl",
    limit: int | None = None,
    needs_ocr_chars_per_page: int = 80,
    low_completeness_ratio: float = 0.35,
) -> dict[str, Any]:
    data_path = Path(data_dir)
    out_path = ensure_parent(output_path)
    stats = {
        "documents": 0,
        "missing_pdf": 0,
        "missing_markdown": 0,
        "needs_ocr": 0,
        "ocr_unresolved": 0,
        "low_completeness": 0,
        "page_mismatch": 0,
        "ok": 0,
    }
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, record in enumerate(read_jsonl(data_path / "documents.jsonl")):
            if limit is not None and index >= limit:
                break
            stats["documents"] += 1
            pdf_path = Path(record.get("source_file") or "")
            if not pdf_path.exists():
                pdf_path = Path.cwd() / pdf_path
            md_path = Path(record.get("markdown_clean_path") or "")
            row = {
                "doc_id": record.get("doc_id"),
                "title": record.get("title"),
                "publication_date": record.get("publication_date"),
                "source_institution": record.get("source_institution"),
                "source_file": str(pdf_path),
                "markdown_clean_path": str(md_path),
            }
            if not pdf_path.exists():
                row["status"] = "missing_pdf"
                stats["missing_pdf"] += 1
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                continue
            if not md_path.exists():
                row["status"] = "missing_markdown"
                stats["missing_markdown"] += 1
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                continue
            try:
                row.update(_pdf_text_stats(pdf_path))
                row.update(_markdown_stats(md_path))
                metadata = row.get("metadata") or {}
                row["source_pdf_text_quality"] = metadata.get("source_pdf_text_quality") or record.get("source_pdf_text_quality") or ""
                row["effective_pdf_text_quality"] = metadata.get("pdf_text_quality") or record.get("pdf_text_quality") or ""
                row["ocr_status"] = metadata.get("ocr_status") or record.get("ocr_status") or ""
                row["ocr_applied"] = (metadata.get("ocr_applied") or str(record.get("ocr_applied") or "")).lower() == "true"
                ratio = row["markdown_chars"] / max(1, row["pdf_text_chars"])
                row["markdown_to_pdf_text_ratio"] = round(ratio, 4)
                row["needs_ocr"] = bool(row["pdf_needs_ocr"] or row["pdf_text_chars_per_page"] < needs_ocr_chars_per_page)
                row["low_completeness"] = bool(row["pdf_text_chars"] >= 1000 and ratio < low_completeness_ratio)
                row["page_mismatch"] = bool(
                    row["pdf_pages"] > 0
                    and row["markdown_page_markers"] > 0
                    and abs(row["markdown_page_markers"] - row["pdf_pages"]) > max(2, int(row["pdf_pages"] * 0.1))
                )
                row["ocr_unresolved"] = row["ocr_status"] in {"needed_unavailable", "failed", "applied_needs_review"}
                if row["needs_ocr"]:
                    stats["needs_ocr"] += 1
                if row["ocr_unresolved"]:
                    stats["ocr_unresolved"] += 1
                if row["low_completeness"]:
                    stats["low_completeness"] += 1
                if row["page_mismatch"]:
                    stats["page_mismatch"] += 1
                row["status"] = (
                    "needs_review"
                    if row["needs_ocr"] or row["low_completeness"] or row["page_mismatch"] or row["ocr_unresolved"]
                    else "ok"
                )
                if row["status"] == "ok":
                    stats["ok"] += 1
            except Exception as exc:  # noqa: BLE001
                row["status"] = "audit_error"
                row["error"] = str(exc)
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    stats["report"] = str(output_path)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--output", default=str(DATA_DIR / "reports" / "pdf_markdown_audit.jsonl"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(audit_pdf_markdown(args.data_dir, args.output, args.limit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
