"""Baidu 高精度 OCR（带坐标），用于 PDF → Markdown 恢复。

关键设计：
- 单 PDF → 逐页渲染为 JPEG → 调百度 accurate_position OCR；
- 返回 words_result 含每行坐标（left/top/width/height），便于按版面恢复单/双栏；
- 单页缓存到 page_cache/accurate_position/<doc_id>/<NNNN>.json，避免重复计费；
- 配额耗尽时通过 BaiduQuotaExceeded 抛错，由批处理转为 deferred_quota 任务；
- 错误码 17/19 → 配额；18 → 服务繁忙，按指数退避重试。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

from src.utils.front_matter import dump_front_matter, parse_front_matter
from src.utils.io import ensure_parent


TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
OCR_URL = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate"
MAX_FORM_BYTES = 10 * 1024 * 1024
OCR_INCOMPLETE = {"needed_unavailable", "needed_but_disabled", "failed", "applied_needs_review"}


class BaiduQuotaExceeded(RuntimeError):
    """百度配额耗尽信号（错误码 17/19），由批处理转为 deferred_quota 状态。"""
    pass


def helper_credentials() -> tuple[str, str]:
    """读取 BAIDU_OCR_API_KEY/SECRET_KEY（兼容旧 BAIDU_API_KEY/SECRET_KEY 别名）。"""
    api_key = os.environ.get("BAIDU_OCR_API_KEY") or os.environ.get("BAIDU_API_KEY")
    secret_key = os.environ.get("BAIDU_OCR_SECRET_KEY") or os.environ.get("BAIDU_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError("Set BAIDU_OCR_API_KEY and BAIDU_OCR_SECRET_KEY before running Baidu OCR.")
    return api_key, secret_key


def get_access_token(session: requests.Session | None = None) -> str:
    """获取 access_token：优先用 BAIDU_OCR_ACCESS_TOKEN，否则按 API key 兑换。"""
    direct_token = os.environ.get("BAIDU_OCR_ACCESS_TOKEN")
    if direct_token:
        return direct_token
    api_key, secret_key = helper_credentials()
    client = session or requests.Session()
    response = client.post(
        TOKEN_URL,
        params={"grant_type": "client_credentials", "client_id": api_key, "client_secret": secret_key},
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(f"Baidu token request failed with HTTP {response.status_code}.")
    payload = response.json()
    token = payload.get("access_token")
    if not token:
        code = payload.get("error") or payload.get("error_code") or "unknown"
        raise RuntimeError(f"Baidu token response did not include access_token (error={code}).")
    return str(token)


def helper_form_size(image_bytes: bytes) -> int:
    """估算 base64 编码后 form 表单字节数（百度 10MB 上限参考值）。"""
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return len(urlencode({"image": encoded}).encode("ascii"))


def helper_render_page_jpeg(pdf_path: Path, page_index: int, dpi: int = 180) -> tuple[bytes, int, int, int, int]:
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required to render PDF pages for OCR.") from exc
    attempts = []
    for render_dpi in dict.fromkeys([dpi, min(dpi, 150), min(dpi, 120)]):
        doc = fitz.open(str(pdf_path))
        try:
            page = doc[page_index]
            matrix = fitz.Matrix(render_dpi / 72, render_dpi / 72)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            for quality in (85, 70, 55):
                image = pixmap.tobytes("jpeg", jpg_quality=quality)
                form_size = helper_form_size(image)
                attempts.append((render_dpi, quality, form_size))
                if form_size <= MAX_FORM_BYTES:
                    return image, render_dpi, quality, pixmap.width, pixmap.height
        finally:
            doc.close()
    raise RuntimeError(f"Rendered page exceeds Baidu's 10 MB request limit: {attempts[-1] if attempts else 'unknown'}")


def helper_safe_ocr_error(payload: dict[str, Any]) -> str:
    """把百度错误响应压成可读字符串（Baidu OCR error {code}: {message}）。"""
    code = payload.get("error_code") or payload.get("error") or "unknown"
    message = payload.get("error_msg") or payload.get("error_description") or "unknown error"
    return f"Baidu OCR error {code}: {message}"


def ocr_image_payload(
    image_bytes: bytes,
    access_token: str,
    *,
    session: requests.Session | None = None,
    retries: int = 3,
) -> dict[str, Any]:
    # 凭据、网络请求和响应校验集中在单页边界，错误必须附带可定位的服务信息。
    client = session or requests.Session()
    data = {
        "image": base64.b64encode(image_bytes),
        "language_type": "CHN_ENG",
        "paragraph": "true",
        "detect_direction": "true",
        "probability": "true",
    }
    for attempt in range(retries):
        try:
            response = client.post(
                OCR_URL,
                params={"access_token": access_token},
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=90,
            )
        except requests.RequestException:
            if attempt + 1 >= retries:
                raise RuntimeError("Baidu OCR network request failed after retries.") from None
            time.sleep(2**attempt)
            continue
        if not response.ok:
            if response.status_code >= 500 and attempt + 1 < retries:
                time.sleep(2**attempt)
                continue
            raise RuntimeError(f"Baidu OCR request failed with HTTP {response.status_code}.")
        payload = response.json()
        if "error_code" not in payload:
            return payload
        error_code = payload.get("error_code")
        if error_code in {17, 19}:
            raise BaiduQuotaExceeded(helper_safe_ocr_error(payload))
        if error_code == 18 and attempt + 1 < retries:
            time.sleep(2**attempt)
            continue
        raise RuntimeError(helper_safe_ocr_error(payload))
    raise RuntimeError("Baidu OCR request failed after retries.")


def helper_payload_layout(payload: dict[str, Any], page_width: int | None = None) -> tuple[str, str]:
    """按坐标把 words_result 重组为单栏或双栏 Markdown，恢复原始版面顺序。"""
    lines = [item for item in payload.get("words_result", []) if str(item.get("words") or "").strip()]
    positioned = [item for item in lines if isinstance(item.get("location"), dict)]
    if len(positioned) != len(lines) or not lines:
        # 坐标缺失时不能可靠恢复版面，只能保持 API 返回顺序。
        return "\n".join(str(item.get("words") or "").strip() for item in lines), "api_order"

    width = float(page_width or max(
        float(item["location"].get("left", 0)) + float(item["location"].get("width", 0))
        for item in positioned
    ))
    # 中线两侧保留少量 gutter，跨越中线的标题和表格归入 spanning。
    midpoint = width / 2
    gutter = width * 0.04
    left: list[dict[str, Any]] = []
    right: list[dict[str, Any]] = []
    spanning: list[dict[str, Any]] = []
    for item in positioned:
        location = item["location"]
        x = float(location.get("left", 0))
        line_width = float(location.get("width", 0))
        center = x + line_width / 2
        if center < midpoint and x + line_width <= midpoint + gutter:
            left.append(item)
        elif center >= midpoint and x >= midpoint - gutter:
            right.append(item)
        else:
            spanning.append(item)

    sort_key = lambda item: (float(item["location"].get("top", 0)), float(item["location"].get("left", 0)))
    minimum_column_lines = max(4, int(len(positioned) * 0.12))
    # 任一列样本过少时视为单栏，避免把缩进段落误判成双栏。
    if len(left) < minimum_column_lines or len(right) < minimum_column_lines:
        ordered = sorted(positioned, key=sort_key)
        return "\n".join(str(item["words"]).strip() for item in ordered), "single_column"

    paired_tops = []
    for left_item in left:
        left_top = float(left_item["location"].get("top", 0))
        left_height = float(left_item["location"].get("height", 0))
        for right_item in right:
            right_top = float(right_item["location"].get("top", 0))
            right_height = float(right_item["location"].get("height", 0))
            if abs(left_top - right_top) <= max(35.0, left_height, right_height):
                paired_tops.append(min(left_top, right_top))
                break
    column_top = min(paired_tops) if paired_tops else min(
        float(item["location"].get("top", 0)) for item in left + right
    )
    headers = [item for item in positioned if float(item["location"].get("top", 0)) < column_top]
    left_body = [item for item in left if item not in headers]
    right_body = [item for item in right if item not in headers]
    spanning_body = [item for item in spanning if item not in headers]

    footer_prefixes = ("DOI:", "DOI?", "????", "????", "????", "????")

    def split_column_footer(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        starts = [
            float(item["location"].get("top", 0))
            for item in items
            if str(item.get("words") or "").strip().startswith(footer_prefixes)
        ]
        if not starts:
            return items, []
        footer_top = min(starts)
        return (
            [item for item in items if float(item["location"].get("top", 0)) < footer_top],
            [item for item in items if float(item["location"].get("top", 0)) >= footer_top],
        )

    left_body, left_footers = split_column_footer(left_body)
    right_body, right_footers = split_column_footer(right_body)
    column_bottom = max(
        (
            float(item["location"].get("top", 0)) + float(item["location"].get("height", 0))
            for item in left_body + right_body
        ),
        default=column_top,
    )
    footers = [item for item in spanning_body if float(item["location"].get("top", 0)) >= column_bottom]
    middle = [item for item in spanning_body if item not in footers]
    ordered = (
        sorted(headers, key=sort_key)
        + sorted(left_body, key=sort_key)
        + sorted(right_body, key=sort_key)
        + sorted(middle, key=sort_key)
        + sorted(left_footers + right_footers, key=sort_key)
        + sorted(footers, key=sort_key)
    )
    return "\n".join(str(item["words"]).strip() for item in ordered), "two_column"


def helper_payload_text(payload: dict[str, Any], page_width: int | None = None) -> str:
    """helper_payload_layout 的文本便捷接口（丢弃 layout_mode 信息）。"""
    return helper_payload_layout(payload, page_width)[0]


def helper_payload_confidence(payload: dict[str, Any]) -> float | None:
    """汇总全页 confidence.average，用于审计平均识别准确率。"""
    values = []
    for item in payload.get("words_result", []):
        probability = item.get("probability") or {}
        if isinstance(probability, dict) and isinstance(probability.get("average"), (int, float)):
            values.append(float(probability["average"]))
    return sum(values) / len(values) if values else None


def helper_write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    # OCR 页结果采用临时文件原子替换，重跑时不会读取到截断 JSON。
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, path)


def helper_page_count(pdf_path: Path) -> int:
    """惰性 import fitz，读 PDF 总页数。"""
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required to inspect PDF pages for OCR.") from exc
    doc = fitz.open(str(pdf_path))
    try:
        return len(doc)
    finally:
        doc.close()


def ocr_pdf_to_markdown(
    pdf_path: str | Path,
    output_path: str | Path,
    max_pages: int | None = None,
    dpi: int = 180,
    sleep_seconds: float = 0.5,
    *,
    doc_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    cache_dir: str | Path | None = None,
    cache_doc_id: str | None = None,
    access_token: str | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    # 页面缓存优先于远程请求；缓存未命中时按页识别并保持原页顺序组装 Markdown。
    pdf = Path(pdf_path)
    if not pdf.is_file():
        raise FileNotFoundError(f"OCR source PDF does not exist: {pdf}")
    identifier = doc_id or Path(output_path).stem or pdf.stem
    pages = helper_page_count(pdf)
    if max_pages is not None:
        pages = min(pages, max_pages)
    client = session or requests.Session()
    token = access_token or get_access_token(client)
    cache_root = (Path(cache_dir) if cache_dir else Path(output_path).parent / "page_cache") / "accurate_position"
    parts = [f"# {str((metadata or {}).get('title') or pdf.stem).strip()}", ""]
    page_texts: list[str] = []
    confidences: list[float] = []
    api_calls = 0
    cached_pages = 0
    for page_index in range(pages):
        page_no = page_index + 1
        cache_path = cache_root / (cache_doc_id or identifier) / f"{page_no:04d}.json"
        if cache_path.is_file():
            page_result = json.loads(cache_path.read_text(encoding="utf-8"))
            if page_result.get("words_result"):
                cached_payload = {
                    "words_result": page_result.get("words_result", []),
                    "paragraphs_result": page_result.get("paragraphs_result", []),
                }
                page_result["text"], page_result["layout_mode"] = helper_payload_layout(
                    cached_payload, page_result.get("image_width")
                )
                helper_write_json_atomic(cache_path, page_result)
            cached_pages += 1
        else:
            image, render_dpi, jpeg_quality, image_width, image_height = helper_render_page_jpeg(pdf, page_index, dpi)
            payload = ocr_image_payload(image, token, session=client)
            page_text, layout_mode = helper_payload_layout(payload, image_width)
            page_result = {
                "page": page_no,
                "text": page_text,
                "layout_mode": layout_mode,
                "confidence": helper_payload_confidence(payload),
                "log_id": payload.get("log_id"),
                "render_dpi": render_dpi,
                "jpeg_quality": jpeg_quality,
                "image_width": image_width,
                "image_height": image_height,
                "words_result": payload.get("words_result", []),
                "paragraphs_result": payload.get("paragraphs_result", []),
            }
            helper_write_json_atomic(cache_path, page_result)
            api_calls += 1
            if sleep_seconds:
                time.sleep(sleep_seconds)
        if isinstance(page_result.get("confidence"), (int, float)):
            confidences.append(float(page_result["confidence"]))
        page_text = str(page_result.get("text") or "").strip()
        page_texts.append(page_text)
        parts.extend([f"<!-- page: {page_no} -->", page_text, ""])
    front_matter = dict(metadata or {})
    front_matter.update(
        {
            "id": identifier,
            "title": front_matter.get("title") or pdf.stem,
            "source_file": str(pdf),
            "ocr_engine": "baidu_accurate_position",
            "ocr_applied": "true",
            "ocr_status": "applied" if page_texts and all(page_texts) else "applied_needs_review",
            "ocr_error": "",
        }
    )
    output = ensure_parent(output_path)
    output.write_text(dump_front_matter(front_matter, "\n".join(parts).strip()), encoding="utf-8", newline="\n")
    return {
        "pdf": str(pdf),
        "output": str(output),
        "pages_ocr": pages,
        "api_calls": api_calls,
        "cached_pages": cached_pages,
        "average_confidence": (sum(confidences) / len(confidences)) if confidences else None,
    }


def helper_pdf_index(search_root: Path) -> dict[str, list[Path]]:
    by_name: dict[str, list[Path]] = defaultdict(list)
    if search_root.exists():
        for path in search_root.rglob("*.pdf"):
            by_name[path.name.casefold()].append(path)
    return by_name


def helper_resolve_path(value: str, data_dir: Path, by_name: dict[str, list[Path]] | None = None) -> Path | None:
    path = Path(value or "")
    candidates = [path, data_dir / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = (by_name or {}).get(path.name.casefold(), [])
    return matches[0] if len(matches) == 1 else None


def helper_iter_audit_rows(audit_report: Path, include_low_completeness: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with audit_report.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            reasons = set(row.get("quarantine_reasons") or [])
            needs_ocr = bool(row.get("needs_ocr")) or "ocr_incomplete" in reasons or row.get("ocr_status") in OCR_INCOMPLETE
            if needs_ocr or (include_low_completeness and row.get("low_completeness")):
                rows.append(row)
    return rows


def helper_replace_markdown_body(markdown_path: Path, ocr_markdown_path: Path) -> None:
    original = markdown_path.read_text(encoding="utf-8", errors="replace") if markdown_path.exists() else ""
    metadata, _body = parse_front_matter(original)
    ocr_text = ocr_markdown_path.read_text(encoding="utf-8", errors="replace")
    _ocr_metadata, ocr_body = parse_front_matter(ocr_text)
    metadata.update({"ocr_engine": "baidu_accurate_position", "ocr_applied": "true", "ocr_status": "applied", "ocr_error": ""})
    markdown_path.write_text(dump_front_matter(metadata, ocr_body or ocr_text), encoding="utf-8", newline="\n")


def ocr_from_audit_report(
    audit_report: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    limit: int | None = None,
    include_low_completeness: bool = False,
    replace_markdown: bool = False,
    dry_run: bool = True,
    dpi: int = 180,
    sleep_seconds: float = 0.5,
    *,
    search_root: str | Path = "data",
    cache_dir: str | Path = "data/evidence/baidu_ocr_pages",
    max_pages: int | None = None,
    max_total_pages: int | None = None,
) -> dict[str, Any]:
    # 批处理同时受文档、单篇页数和总页数预算约束，配额耗尽后只延期而不丢弃任务。
    data_path = Path(data_dir)
    report_path = Path(audit_report)
    out_dir = Path(output_dir)
    rows = helper_iter_audit_rows(report_path, include_low_completeness)
    if limit is not None:
        rows = rows[:limit]
    by_name = helper_pdf_index(Path(search_root))
    run_mode = "dry_run" if dry_run else "execute"
    manifest_path = ensure_parent(out_dir / f"baidu_ocr_manifest_{run_mode}_{time.time_ns()}.jsonl")
    results: list[dict[str, Any]] = []
    processed = failed = missing = deferred_quota = api_calls = pages_planned = 0
    quota_exhausted = False
    client = requests.Session()
    token = None if dry_run else get_access_token(client)
    for row in rows:
        doc_id = row.get("doc_id") or Path(row.get("source_file", "")).stem
        pdf_path = helper_resolve_path(row.get("source_file", ""), data_path, by_name)
        markdown_path = helper_resolve_path(row.get("markdown_clean_path", ""), data_path)
        output_path = out_dir / f"{doc_id}.md"
        item: dict[str, Any] = {
            "doc_id": doc_id,
            "document_kind": row.get("document_kind") or "guideline",
            "source_file": row.get("source_file"),
            "resolved_pdf": str(pdf_path) if pdf_path else "",
            "markdown_clean_path": str(markdown_path) if markdown_path else row.get("markdown_clean_path", ""),
            "output": str(output_path),
            "dry_run": dry_run,
        }
        if pdf_path is None:
            item.update({"status": "missing_source_pdf", "error": "Source PDF could not be resolved by path or unique filename."})
            missing += 1
            results.append(item)
            continue
        try:
            pages = helper_page_count(pdf_path)
            if max_pages is not None:
                pages = min(pages, max_pages)
            if max_total_pages not in {None, 0} and pages_planned + pages > int(max_total_pages):
                item.update({"status": "skipped_page_budget", "pages_planned": pages})
                results.append(item)
                continue
            pages_planned += pages
            item["pages_planned"] = pages
            if quota_exhausted:
                deferred_quota += 1
                item.update(
                    {
                        "status": "deferred_quota",
                        "error": "Baidu OCR quota is exhausted; rerun after quota becomes available.",
                    }
                )
            elif dry_run:
                item["status"] = "planned"
            else:
                result = ocr_pdf_to_markdown(
                    pdf_path,
                    output_path,
                    max_pages=max_pages,
                    dpi=dpi,
                    sleep_seconds=sleep_seconds,
                    doc_id=doc_id,
                    metadata=row,
                    cache_dir=cache_dir,
                    access_token=token,
                    session=client,
                )
                item.update(result)
                api_calls += int(result["api_calls"])
                if replace_markdown:
                    if markdown_path is None:
                        raise FileNotFoundError("Existing markdown_clean_path could not be resolved.")
                    helper_replace_markdown_body(markdown_path, output_path)
                    item["markdown_replaced"] = True
                item["status"] = "completed"
            if item["status"] in {"planned", "completed"}:
                processed += 1
        except BaiduQuotaExceeded as exc:
            quota_exhausted = True
            deferred_quota += 1
            item.update({"status": "deferred_quota", "error": str(exc)[:800]})
        except Exception as exc:  # noqa: BLE001
            failed += 1
            item.update({"status": "failed", "error": str(exc)[:800]})
        results.append(item)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as manifest:
        for item in results:
            manifest.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {
        "audit_report": str(report_path),
        "candidates": len(rows),
        "processed_or_planned": processed,
        "missing_source_pdf": missing,
        "failed": failed,
        "deferred_quota": deferred_quota,
        "pages_planned": pages_planned,
        "api_calls": api_calls,
        "output_dir": str(out_dir),
        "page_cache": str(cache_dir),
        "manifest": str(manifest_path),
        "replace_markdown": replace_markdown,
        "dry_run": dry_run,
    }


def main() -> None:
    # 命令行仅负责参数校验和调度，凭据必须来自环境变量而不是参数回显。
    parser = argparse.ArgumentParser(description="Convert OCR-required PDFs to Markdown with Baidu accurate OCR and positions.")
    parser.add_argument("--pdf")
    parser.add_argument("--output")
    parser.add_argument("--audit-report")
    parser.add_argument("--data-dir", default="data/evidence")
    parser.add_argument("--search-root", default="data")
    parser.add_argument("--output-dir", default="data/evidence/markdown_baidu_ocr")
    parser.add_argument("--cache-dir", default="data/evidence/baidu_ocr_pages")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--include-low-completeness", action="store_true")
    parser.add_argument("--replace-markdown", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Make paid API calls. Without this flag only a dry-run is performed.")
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--max-total-pages", type=int)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    args = parser.parse_args()
    if args.execute and args.max_total_pages is None and args.audit_report:
        parser.error("--max-total-pages is required with --execute; use 0 only after confirming the full API quota/cost.")
    if args.replace_markdown and not args.execute:
        parser.error("--replace-markdown requires --execute.")
    if args.audit_report:
        result = ocr_from_audit_report(
            args.audit_report,
            args.data_dir,
            args.output_dir,
            args.limit,
            args.include_low_completeness,
            args.replace_markdown,
            not args.execute,
            args.dpi,
            args.sleep_seconds,
            search_root=args.search_root,
            cache_dir=args.cache_dir,
            max_pages=args.max_pages,
            max_total_pages=args.max_total_pages,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if not args.pdf or not args.output:
        parser.error("--pdf and --output are required unless --audit-report is used")
    if not args.execute:
        print(json.dumps({"pdf": args.pdf, "output": args.output, "pages_planned": helper_page_count(Path(args.pdf)), "dry_run": True}, ensure_ascii=False, indent=2))
        return
    print(
        json.dumps(
            ocr_pdf_to_markdown(
                args.pdf,
                args.output,
                args.max_pages,
                args.dpi,
                args.sleep_seconds,
                cache_dir=args.cache_dir,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
