"""Baidu OCR fallback for scanned PDFs."""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path
from typing import Any

import requests

from src.utils.front_matter import dump_front_matter, parse_front_matter
from src.utils.io import ensure_parent


TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
OCR_URL = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic"


def _credentials() -> tuple[str, str]:
    api_key = os.environ.get("BAIDU_OCR_API_KEY") or os.environ.get("BAIDU_API_KEY")
    secret_key = os.environ.get("BAIDU_OCR_SECRET_KEY") or os.environ.get("BAIDU_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError("Set BAIDU_OCR_API_KEY and BAIDU_OCR_SECRET_KEY before running Baidu OCR.")
    return api_key, secret_key


def get_access_token() -> str:
    api_key, secret_key = _credentials()
    response = requests.post(
        TOKEN_URL,
        params={
            "grant_type": "client_credentials",
            "client_id": api_key,
            "client_secret": secret_key,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    token = payload.get("access_token")
    if not token:
        raise RuntimeError(f"Baidu token response did not include access_token: {payload}")
    return token


def _render_page_png(pdf_path: Path, page_index: int, dpi: int) -> bytes:
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required to render PDF pages for OCR.") from exc
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_index]
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        return pixmap.tobytes("png")
    finally:
        doc.close()


def ocr_image_bytes(image_bytes: bytes, access_token: str) -> str:
    response = requests.post(
        OCR_URL,
        params={"access_token": access_token},
        data={
            "image": base64.b64encode(image_bytes),
            "paragraph": "true",
            "detect_direction": "true",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if "error_code" in payload:
        raise RuntimeError(f"Baidu OCR error: {payload}")
    return "\n".join(item.get("words", "") for item in payload.get("words_result", []) if item.get("words"))


def ocr_pdf_to_markdown(
    pdf_path: str | Path,
    output_path: str | Path,
    max_pages: int | None = None,
    dpi: int = 180,
    sleep_seconds: float = 0.2,
) -> dict[str, Any]:
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required to inspect PDF pages for OCR.") from exc
    pdf = Path(pdf_path)
    token = get_access_token()
    doc = fitz.open(str(pdf))
    try:
        pages = len(doc)
    finally:
        doc.close()
    if max_pages is not None:
        pages = min(pages, max_pages)
    parts = [f"# OCR: {pdf.stem}", ""]
    for page_index in range(pages):
        image = _render_page_png(pdf, page_index, dpi)
        text = ocr_image_bytes(image, token)
        parts.append(f"<!-- page: {page_index + 1} -->")
        parts.append(text)
        parts.append("")
        if sleep_seconds:
            time.sleep(sleep_seconds)
    out = ensure_parent(output_path)
    out.write_text("\n".join(parts).strip() + "\n", encoding="utf-8", newline="\n")
    return {"pdf": str(pdf), "output": str(out), "pages_ocr": pages}


def _resolve_path(value: str, data_dir: Path) -> Path:
    path = Path(value or "")
    if path.exists():
        return path
    candidate = data_dir / path
    if candidate.exists():
        return candidate
    return Path.cwd() / path


def _iter_audit_rows(audit_report: Path, include_low_completeness: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with audit_report.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("needs_ocr") or (include_low_completeness and row.get("low_completeness")):
                rows.append(row)
    return rows


def _replace_markdown_body(markdown_path: Path, ocr_markdown_path: Path) -> None:
    original = markdown_path.read_text(encoding="utf-8", errors="replace") if markdown_path.exists() else ""
    metadata, _body = parse_front_matter(original)
    ocr_text = ocr_markdown_path.read_text(encoding="utf-8", errors="replace")
    _ocr_metadata, ocr_body = parse_front_matter(ocr_text)
    body = ocr_body or ocr_text
    markdown_path.write_text(dump_front_matter(metadata, body), encoding="utf-8", newline="\n")


def ocr_from_audit_report(
    audit_report: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    limit: int | None = None,
    include_low_completeness: bool = False,
    replace_markdown: bool = False,
    dry_run: bool = False,
    dpi: int = 180,
    sleep_seconds: float = 0.2,
) -> dict[str, Any]:
    data_path = Path(data_dir)
    report_path = Path(audit_report)
    out_dir = Path(output_dir)
    rows = _iter_audit_rows(report_path, include_low_completeness)
    if limit is not None:
        rows = rows[:limit]
    manifest_path = ensure_parent(out_dir / "baidu_ocr_manifest.jsonl")
    processed = 0
    failed = 0
    with manifest_path.open("w", encoding="utf-8", newline="\n") as manifest:
        for row in rows:
            doc_id = row.get("doc_id") or Path(row.get("source_file", "")).stem
            pdf_path = _resolve_path(row.get("source_file", ""), data_path)
            markdown_path = _resolve_path(row.get("markdown_clean_path", ""), data_path)
            output_path = out_dir / f"{doc_id}.md"
            item = {
                "doc_id": doc_id,
                "pdf": str(pdf_path),
                "markdown_clean_path": str(markdown_path),
                "output": str(output_path),
                "dry_run": dry_run,
            }
            try:
                if not dry_run:
                    item.update(
                        ocr_pdf_to_markdown(
                            pdf_path,
                            output_path,
                            dpi=dpi,
                            sleep_seconds=sleep_seconds,
                        )
                    )
                    if replace_markdown:
                        _replace_markdown_body(markdown_path, output_path)
                        item["markdown_replaced"] = True
                processed += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                item["error"] = str(exc)
            manifest.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {
        "audit_report": str(report_path),
        "candidates": len(rows),
        "processed": processed,
        "failed": failed,
        "output_dir": str(out_dir),
        "manifest": str(manifest_path),
        "replace_markdown": replace_markdown,
        "dry_run": dry_run,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf")
    parser.add_argument("--output")
    parser.add_argument("--audit-report")
    parser.add_argument("--data-dir", default="data/evidence")
    parser.add_argument("--output-dir", default="data/evidence/ocr_baidu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--include-low-completeness", action="store_true")
    parser.add_argument("--replace-markdown", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    if args.audit_report:
        print(
            json.dumps(
                ocr_from_audit_report(
                    args.audit_report,
                    args.data_dir,
                    args.output_dir,
                    args.limit,
                    args.include_low_completeness,
                    args.replace_markdown,
                    args.dry_run,
                    args.dpi,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if not args.pdf or not args.output:
        parser.error("--pdf and --output are required unless --audit-report is used")
    print(json.dumps(ocr_pdf_to_markdown(args.pdf, args.output, args.max_pages, args.dpi), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
