"""Infini-AI DeepSeek OCR 2 PDF-to-Markdown recovery."""

from __future__ import annotations

import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

from src.pipeline.ocr.baidu_ocr import helper_page_count, helper_write_json_atomic
from src.utils.front_matter import dump_front_matter
from src.utils.io import ensure_parent


API_URL = "https://cloud.infini-ai.com/maas/v1/chat/completions"
MODEL = "deepseek-ocr-2"
PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
MAX_IMAGE_BYTES = 4 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 120
GROUNDING_TOKEN_RE = re.compile(r"<\|(?:ref|det)\|>.*?<\|/(?:ref|det)\|>")


class DeepSeekOCRError(RuntimeError):
    pass


def helper_api_key() -> str:
    api_key = (
        os.environ.get("INFINI_API_KEY")
        or os.environ.get("GENSTUDIO_API_KEY")
        or os.environ.get("API_KEY")
    )
    if not api_key:
        raise RuntimeError("Set INFINI_API_KEY, GENSTUDIO_API_KEY, or API_KEY before running DeepSeek OCR.")
    return api_key


def helper_render_page_jpeg(pdf_path: Path, page_index: int, dpi: int = 180) -> bytes:
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required to render PDF pages for OCR.") from exc
    attempts: list[tuple[int, int, int]] = []
    for render_dpi in dict.fromkeys([dpi, min(dpi, 150), min(dpi, 120), min(dpi, 96)]):
        document = fitz.open(str(pdf_path))
        try:
            page = document[page_index]
            pixmap = page.get_pixmap(matrix=fitz.Matrix(render_dpi / 72, render_dpi / 72), alpha=False)
            for quality in (85, 70, 55, 40):
                image = pixmap.tobytes("jpeg", jpg_quality=quality)
                attempts.append((render_dpi, quality, len(image)))
                if len(image) <= MAX_IMAGE_BYTES:
                    return image
        finally:
            document.close()
    raise RuntimeError(f"Rendered page exceeds the 4 MB OCR safety limit: {attempts[-1]}")


def helper_response_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise DeepSeekOCRError("DeepSeek OCR response did not include choices.")
    content = (choices[0].get("message") or {}).get("content")
    if isinstance(content, list):
        content = "\n".join(str(item.get("text") or "") for item in content if isinstance(item, dict))
    text = str(content or "").strip()
    if text.startswith("```markdown") and text.endswith("```"):
        text = text[len("```markdown") : -3].strip()
    text = GROUNDING_TOKEN_RE.sub("", text).strip()
    if not text:
        raise DeepSeekOCRError("DeepSeek OCR response content was empty.")
    return text


def ocr_image_markdown(
    image_bytes: bytes,
    api_key: str,
    *,
    session: requests.Session | None = None,
    retries: int = 1,
) -> dict[str, Any]:
    # 单页调用负责请求、响应形状和 Markdown 提取，服务错误不得伪装成空识别结果。
    client = session or requests.Session()
    request_payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii")
                        },
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 4096,
        "stream": False,
    }
    for attempt in range(retries):
        try:
            response = client.post(
                API_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=request_payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
                stream=True,
            )
        except requests.RequestException:
            if attempt + 1 >= retries:
                raise DeepSeekOCRError("DeepSeek OCR network request failed after retries.") from None
            time.sleep(2**attempt)
            continue
        if response.status_code == 429 or response.status_code >= 500:
            if attempt + 1 < retries:
                time.sleep(2**attempt)
                continue
        if not response.ok:
            raise DeepSeekOCRError(f"DeepSeek OCR request failed with HTTP {response.status_code}.")
        if hasattr(response, "iter_content"):
            deadline = time.monotonic() + REQUEST_TIMEOUT_SECONDS
            chunks: list[bytes] = []
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if time.monotonic() > deadline:
                    raise DeepSeekOCRError(
                        f"DeepSeek OCR response exceeded the {REQUEST_TIMEOUT_SECONDS} second total limit."
                    )
                if chunk:
                    chunks.append(chunk)
            payload = json.loads(b"".join(chunks))
        else:
            payload = response.json()
        return {
            "text": helper_response_text(payload),
            "request_id": payload.get("request_id") or payload.get("id") or "",
            "usage": payload.get("usage") or {},
        }
    raise DeepSeekOCRError("DeepSeek OCR request failed after retries.")


def ocr_pdf_to_markdown(
    pdf_path: str | Path,
    output_path: str | Path,
    *,
    doc_id: str,
    metadata: dict[str, Any],
    cache_dir: str | Path,
    api_key: str | None = None,
    max_api_pages: int | None = None,
    max_workers: int = 1,
    dpi: int = 180,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    # 逐页缓存与输出顺序绑定，部分页面失败时必须保留明确失败状态。
    pdf = Path(pdf_path)
    pages = helper_page_count(pdf)
    cache_root = Path(cache_dir) / "deepseek_ocr_2" / doc_id
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_paths = [cache_root / f"{page_no:04d}.json" for page_no in range(1, pages + 1)]
    cached_pages = sum(path.is_file() for path in cache_paths)
    missing_indexes = [index for index, path in enumerate(cache_paths) if not path.is_file()]
    if max_api_pages is not None:
        missing_indexes = missing_indexes[:max_api_pages]
    resolved_api_key = api_key or (helper_api_key() if missing_indexes else "")

    def process_page(page_index: int) -> str:
        page_no = page_index + 1
        try:
            result = ocr_image_markdown(
                helper_render_page_jpeg(pdf, page_index, dpi),
                resolved_api_key,
                session=session,
            )
            helper_write_json_atomic(
                cache_paths[page_index],
                {
                    "provider": "infini-ai",
                    "model": MODEL,
                    "doc_id": doc_id,
                    "page": page_no,
                    **result,
                },
            )
            return ""
        except Exception as exc:  # noqa: BLE001
            return f"page {page_no}: {str(exc)[:400]}"

    page_errors: list[str] = []
    workers = 1 if session is not None else max(1, max_workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_page, page_index) for page_index in missing_indexes]
        for future in as_completed(futures):
            error = future.result()
            if error:
                page_errors.append(error)

    api_calls = len(missing_indexes)
    page_texts = []
    for cache_path in cache_paths:
        cached = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else {}
        page_texts.append(GROUNDING_TOKEN_RE.sub("", str(cached.get("text") or "")).strip())
    missing_pages = sum(not text for text in page_texts)
    complete = missing_pages == 0
    if complete:
        front_matter = dict(metadata)
        front_matter.pop("cleaning_quality", None)
        front_matter.pop("cleaning_flags", None)
        front_matter.update(
            {
                "id": doc_id,
                "source_file": str(pdf),
                "ocr_engine": "infini_deepseek_ocr_2",
                "ocr_applied": "true",
                "ocr_status": "applied",
                "ocr_error": "",
            }
        )
        body = "\n\n".join(
            f"<!-- page: {page_no} -->\n{text}" for page_no, text in enumerate(page_texts, 1)
        )
        ensure_parent(output_path).write_text(
            dump_front_matter(front_matter, body), encoding="utf-8", newline="\n"
        )
    return {
        "pdf": str(pdf),
        "output": str(output_path),
        "pages_ocr": pages,
        "api_calls": api_calls,
        "cached_pages": cached_pages,
        "missing_pages": missing_pages,
        "complete": complete,
        "errors": page_errors,
    }
