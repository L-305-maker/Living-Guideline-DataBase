from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import median

import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.cleaning.cleaner import assess_cleaned_body
DEFAULT_INVENTORY = ROOT / "data/evidence_candidate/candidate_inventory.csv"
DEFAULT_BATCH = ROOT / "data/evidence_candidate/mineru_inventory_batch_001"
BODY_TYPES = {"text", "list", "table", "image", "equation", "interline_equation"}
BAD_GLYPH_RE = re.compile(r"[\ufffd\u25a0\u25a1\u25aa\ue000-\uf8ff]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
HEADING_RE = re.compile(r"(?m)^#{1,6}\s+")
TAG_RE = re.compile(r"<[^>]+>")
MARKDOWN_SYNTAX_RE = re.compile(r"!\[[^]]*]\([^)]*\)|[`#*_>|~-]")


def visible_count(text: str) -> int:
    return sum(not char.isspace() for char in text)


def content_text(item: dict[str, object]) -> str:
    parts: list[str] = []
    for key in ("text", "table_body", "table_caption", "table_footnote", "image_caption", "image_footnote", "list_items"):
        value = item.get(key)
        if isinstance(value, str):
            parts.append(TAG_RE.sub(" ", value))
        elif isinstance(value, list):
            parts.extend(str(entry) for entry in value if isinstance(entry, str))
    return " ".join(parts)


def duplicate_line_ratio(markdown: str) -> float:
    lines = [re.sub(r"\s+", " ", line).strip().lower() for line in markdown.splitlines()]
    lines = [line for line in lines if visible_count(line) >= 25 and not line.startswith("![](")]
    if not lines:
        return 0.0
    counts = Counter(lines)
    excess = sum(count - 1 for count in counts.values() if count > 1)
    return excess / len(lines)


def pdf_pages(path: Path) -> int:
    document = pdfium.PdfDocument(path)
    try:
        return len(document)
    finally:
        document.close()


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def inspect(row: dict[str, str], batch_dir: Path) -> dict[str, object]:
    doc_id = row["doc_id"]
    ocr_dir = batch_dir / "output" / doc_id / "ocr"
    markdown_path = ocr_dir / f"{doc_id}.md"
    content_path = ocr_dir / f"{doc_id}_content_list.json"
    v2_path = ocr_dir / f"{doc_id}_content_list_v2.json"
    model_path = ocr_dir / f"{doc_id}_model.json"
    required = [markdown_path, content_path, v2_path, model_path]
    missing = [path.name for path in required if not path.is_file()]
    result: dict[str, object] = {
        "doc_id": doc_id,
        "title": row["title"],
        "source_file": row["source_file"],
        "inventory_pages": int(row["pdf_page_count"]),
        "source_pdf_pages": 0,
        "content_pages": 0,
        "v2_pages": 0,
        "model_pages": 0,
        "markdown_chars": 0,
        "visible_chars": 0,
        "visible_chars_per_page": 0.0,
        "median_body_chars_per_page": 0,
        "p10_body_chars_per_page": 0,
        "empty_body_pages": 0,
        "empty_body_page_ratio": 0.0,
        "short_body_pages_lt30": 0,
        "short_body_page_ratio": 0.0,
        "image_or_table_pages": 0,
        "headings": 0,
        "signal_ratio": 0.0,
        "duplicate_line_ratio": 0.0,
        "bad_glyphs": 0,
        "control_chars": 0,
        "cleaner_quality": "missing" if missing else "",
        "cleaner_flags": "",
        "review_status": "reocr_recommended" if missing else "",
        "review_reasons": "missing_artifacts:" + "|".join(missing) if missing else "",
        "lowest_body_page": "",
        "markdown_path": str(markdown_path),
        "inventory_reasons": row["reasons"],
    }
    try:
        result["source_pdf_pages"] = pdf_pages(Path(row["source_file"]))
    except Exception as exc:  # noqa: BLE001
        result["review_status"] = "reocr_recommended"
        result["review_reasons"] = f"source_pdf_unreadable:{type(exc).__name__}"
        return result
    if missing:
        return result

    markdown = markdown_path.read_text(encoding="utf-8", errors="replace")
    content = json.loads(content_path.read_text(encoding="utf-8"))
    v2 = json.loads(v2_path.read_text(encoding="utf-8"))
    model = json.loads(model_path.read_text(encoding="utf-8"))
    pages = int(result["source_pdf_pages"])
    page_body_chars = [0] * pages
    page_has_image_or_table = [False] * pages
    page_indices: list[int] = []
    for item in content:
        page = item.get("page_idx")
        if not isinstance(page, int) or page < 0:
            continue
        page_indices.append(page)
        if page >= pages:
            continue
        item_type = str(item.get("type", ""))
        if item_type in BODY_TYPES:
            page_body_chars[page] += visible_count(content_text(item))
        if item_type in {"image", "table"}:
            page_has_image_or_table[page] = True

    markdown_plain = MARKDOWN_SYNTAX_RE.sub("", TAG_RE.sub(" ", markdown))
    assessment = assess_cleaned_body(markdown)
    empty_pages = sum(chars == 0 for chars in page_body_chars)
    short_pages = sum(chars < 30 for chars in page_body_chars)
    nonempty_pairs = [(index + 1, chars) for index, chars in enumerate(page_body_chars) if chars > 0]
    lowest_page = min(nonempty_pairs, key=lambda pair: pair[1]) if nonempty_pairs else (0, 0)
    result.update(
        {
            "content_pages": max(page_indices, default=-1) + 1,
            "v2_pages": len(v2),
            "model_pages": len(model),
            "markdown_chars": len(markdown),
            "visible_chars": visible_count(markdown_plain),
            "visible_chars_per_page": round(visible_count(markdown_plain) / max(1, pages), 1),
            "median_body_chars_per_page": int(median(page_body_chars)) if page_body_chars else 0,
            "p10_body_chars_per_page": percentile(page_body_chars, 0.1),
            "empty_body_pages": empty_pages,
            "empty_body_page_ratio": round(empty_pages / max(1, pages), 4),
            "short_body_pages_lt30": short_pages,
            "short_body_page_ratio": round(short_pages / max(1, pages), 4),
            "image_or_table_pages": sum(page_has_image_or_table),
            "headings": len(HEADING_RE.findall(markdown)),
            "signal_ratio": assessment["signal_ratio"],
            "duplicate_line_ratio": round(duplicate_line_ratio(markdown), 4),
            "bad_glyphs": len(BAD_GLYPH_RE.findall(markdown)),
            "control_chars": len(CONTROL_RE.findall(markdown)),
            "cleaner_quality": assessment["quality"],
            "cleaner_flags": "|".join(assessment["flags"]),
            "lowest_body_page": lowest_page[0],
        }
    )

    severe: list[str] = []
    review: list[str] = []
    expected = int(result["source_pdf_pages"])
    page_counts = {int(result[key]) for key in ("inventory_pages", "content_pages", "v2_pages", "model_pages")}
    if page_counts != {expected}:
        severe.append("page_count_mismatch")
    if assessment["quality"] == "poor":
        severe.append("cleaner_poor:" + "|".join(assessment["flags"]))
    if int(result["visible_chars"]) < max(300, expected * 10):
        severe.append("very_low_document_text")
    if expected >= 4 and float(result["empty_body_page_ratio"]) >= 0.50:
        severe.append("empty_body_pages_ge_50pct")
    if int(result["bad_glyphs"]) >= 10 or int(result["control_chars"]) > 0:
        severe.append("many_bad_glyphs")

    if float(result["visible_chars_per_page"]) < 80:
        review.append("low_text_density_lt80_per_page")
    if expected >= 5 and float(result["empty_body_page_ratio"]) >= 0.20:
        review.append("empty_body_pages_ge_20pct")
    if expected >= 5 and float(result["short_body_page_ratio"]) >= 0.60:
        review.append("short_body_pages_ge_60pct")
    if float(result["duplicate_line_ratio"]) >= 0.20:
        review.append("duplicate_lines_ge_20pct")
    if 0 < int(result["bad_glyphs"]) < 10:
        review.append("some_bad_glyphs")

    if severe:
        result["review_status"] = "reocr_recommended"
        result["review_reasons"] = ";".join(severe + review)
    elif review:
        result["review_status"] = "manual_review"
        result["review_reasons"] = ";".join(review)
    else:
        result["review_status"] = "pass"
        result["review_reasons"] = ""
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit local MinerU OCR output quality")
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH)
    args = parser.parse_args()
    with args.inventory.open(encoding="utf-8-sig", newline="") as handle:
        inventory = [
            row
            for row in csv.DictReader(handle)
            if row["candidate_type"] == "repair_document"
            and row["action"] == "pending_baidu_ocr"
            and row["source_resolved"].lower() == "true"
        ]
    mineru_ids = {path.stem for path in (args.batch_dir / "output").rglob("*.md")}
    selected = [row for row in inventory if row["doc_id"] in mineru_ids]
    results = [inspect(row, args.batch_dir) for row in selected]
    order = {"reocr_recommended": 0, "manual_review": 1, "pass": 2}
    results.sort(key=lambda row: (order[str(row["review_status"])], -float(row["empty_body_page_ratio"]), str(row["doc_id"])))
    report_path = args.batch_dir / "quality_review_20260721.csv"
    write_csv(report_path, results)
    write_csv(args.batch_dir / "quality_review_reocr.csv", [row for row in results if row["review_status"] == "reocr_recommended"] or results[:0]) if any(row["review_status"] == "reocr_recommended" for row in results) else None
    write_csv(args.batch_dir / "quality_review_manual.csv", [row for row in results if row["review_status"] == "manual_review"] or results[:0]) if any(row["review_status"] == "manual_review" for row in results) else None
    counts = Counter(str(row["review_status"]) for row in results)
    summary = {
        "documents_audited": len(results),
        "status_counts": dict(counts),
        "source_pages": sum(int(row["source_pdf_pages"]) for row in results),
        "page_count_mismatches": sum("page_count_mismatch" in str(row["review_reasons"]) for row in results),
        "documents_with_bad_glyphs": sum(int(row["bad_glyphs"]) > 0 for row in results),
        "documents_with_control_chars": sum(int(row["control_chars"]) > 0 for row in results),
        "criteria": {
            "reocr_recommended": [
                "page count mismatch",
                "existing cleaner reports poor OCR quality",
                "less than max(300, 10 chars/page)",
                "at least 50% body-empty pages",
                "at least 10 bad glyphs or any control character",
            ],
            "manual_review": [
                "less than 80 visible chars/page",
                "at least 20% body-empty pages",
                "at least 60% pages have fewer than 30 body chars",
                "at least 20% duplicate long lines",
                "1-9 bad glyphs",
            ],
        },
    }
    (args.batch_dir / "quality_review_summary_20260721.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(report_path)


if __name__ == "__main__":
    main()
