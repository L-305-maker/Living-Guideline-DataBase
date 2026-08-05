# Infini-AI DeepSeek OCR 2 的 PDF → Markdown 恢复（云端 API）。
#
# 关键设计：
# - 逐页调用，页面级缓存 cache/deepseek_ocr_2/<doc_id>/<NNNN>.json；
# - 多页可并发（ThreadPoolExecutor）；带连接复用通过 session；
# - 单页失败不中断整体，只记录 page_errors；max_api_pages 控制预算；
# - 输出 Markdown 时按页加 <!-- page: N --> 标记，与 PyMuPDF 路径统一。
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


# Infini-AI 兼容 OpenAI chat completions 接口的端点。
API_URL = "https://cloud.infini-ai.com/maas/v1/chat/completions"
MODEL = "deepseek-ocr-2"
# grounding 模式：让模型输出带 ref/det 标签，便于后处理清洗。
PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
# 单张图片 4 MB 上限：避免 API 拒绝超大图；超限自动降 DPI / 降 JPEG 质量。
MAX_IMAGE_BYTES = 4 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 120
# DeepSeek 输出的 ref/det 标签：可视化坐标信息，下游清洗时移除。
GROUNDING_TOKEN_RE = re.compile(r"<\|(?:ref|det)\|>.*?<\|/(?:ref|det)\|>")


class DeepSeekOCRError(RuntimeError):
    """DeepSeek OCR 错误信号，统一对外抛错类型。"""


def helper_api_key() -> str:
    """按优先级读取 API key：INFINI_API_KEY → GENSTUDIO_API_KEY → API_KEY。

    全部缺失时给出明确错误，提示用户在 .env 中设置 INFINI_API_KEY。
    """
    api_key = (
        os.environ.get("INFINI_API_KEY")
        or os.environ.get("GENSTUDIO_API_KEY")
        or os.environ.get("API_KEY")
    )
    if not api_key:
        raise RuntimeError("Set INFINI_API_KEY, GENSTUDIO_API_KEY, or API_KEY before running DeepSeek OCR.")
    return api_key


def helper_render_page_jpeg(pdf_path: Path, page_index: int, dpi: int = 180) -> bytes:
    """把 PDF 单页渲染为 JPEG 字节。

    多次降级策略（保证 ≤4 MB 上限）：
    1. DPI 候选：[180, 150, 120, 96]，按需递减；
    2. 同一 DPI 内，JPEG 质量候选：[85, 70, 55, 40]；
    3. 仍超限则抛 RuntimeError。
    """
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
    """从 OpenAI 风格响应中提取 Markdown 文本。

    处理：
    - content 可能是 list（多段），用 \\n 拼接；
    - 自动剥掉 ```markdown 围栏；
    - 删除 GROUNDING_TOKEN（ref/det 标签）；
    - 空内容抛 DeepSeekOCRError（不让空响应伪装成成功）。
    """
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
    """单页 OCR 调用：构造请求 → POST → 解析响应 → 返回 Markdown。

    重试策略：
    - 网络异常 / HTTP 429 / 5xx：按 2^attempt 指数退避重试，最多 retries 次；
    - 4xx 业务错误（除 429）：立即抛错，不重试；
    - 流式响应：超时限制 REQUEST_TIMEOUT_SECONDS 整体（包含所有 chunk 接收）。
    """
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
    """整本 PDF OCR：逐页渲染 → API 调用 → 缓存 → 拼装 Markdown。

    关键行为：
    - 缓存目录 cache/deepseek_ocr_2/<doc_id>/<NNNN>.json，已存在的页跳过；
    - max_api_pages 限制本次新增 API 调用数（保留预算给后续任务）；
    - 单页失败记入 page_errors，但不影响其它页；
    - 全部成功才写 output（带 front-matter + <!-- page: N --> 标记）；
    - 返回字段：pages_ocr / api_calls / cached_pages / missing_pages / complete / errors。
    """
    pdf = Path(pdf_path)
    pages = helper_page_count(pdf)
    cache_root = Path(cache_dir) / "deepseek_ocr_2" / doc_id
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_paths = [cache_root / f"{page_no:04d}.json" for page_no in range(1, pages + 1)]
    cached_pages = sum(path.is_file() for path in cache_paths)
    missing_indexes = [index for index, path in enumerate(cache_paths) if not path.is_file()]
    if max_api_pages is not None:
        missing_indexes = missing_indexes[:max_api_pages]
    # 无缺失页时跳过 api_key 读取，避免无意义的环境变量错误。
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
    # 当调用方传入自定义 session 时强制单线程：避免共享 session 的竞态。
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