# PDF 文本层评估与 OCR 路由信号。
#
# 流程：inspect_pdf_text_layer → assess_pdf_text_layer → 输出 PdfTextLayerReport。
# PdfTextLayerReport 决定是否走 OCR：
# - is_scanned=True：超过 60% 页面是扫描件 → 必须 OCR
# - needs_ocr=True：文本层字符数过少 → 建议 OCR
# - quality ∈ {ok, warning, poor}：给下游 front-matter 提供提示
"""PDF text-layer inspection and OCR routing signals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class PdfPageTextStats:
    """单页文本层统计：page_number / text_chars / image_area_ratio。

    image_area_ratio 是图片 bbox 面积与页面面积的比值，
    与 text_chars 共同作为 OCR 路由决策的依据。
    """
    page_number: int
    text_chars: int
    image_area_ratio: float


@dataclass(frozen=True)
class PdfTextLayerReport:
    """整篇 PDF 的文本层评估报告。

    字段：
    - pages / text_chars / text_chars_per_page：基本统计
    - low_text_pages / image_pages / scanned_pages：异常页计数
    - low_text_page_ratio / image_page_ratio / scanned_page_ratio：异常页占比
    - is_scanned：是否整篇都是扫描件
    - needs_ocr：是否建议走 OCR
    - quality ∈ {ok, warning, poor}：用于 front-matter 的 cleaning_quality
    """
    pages: int
    text_chars: int
    text_chars_per_page: float
    low_text_pages: int
    low_text_page_ratio: float
    image_pages: int
    image_page_ratio: float
    scanned_pages: int
    scanned_page_ratio: float
    is_scanned: bool
    needs_ocr: bool
    quality: str

    def as_metadata(self, prefix: str = "pdf") -> dict[str, str]:
        """序列化为 front-matter 字段（带 prefix 区分 source_pdf / pdf）。"""
        return {
            f"{prefix}_page_count": str(self.pages),
            f"{prefix}_text_chars": str(self.text_chars),
            f"{prefix}_text_chars_per_page": f"{self.text_chars_per_page:.2f}",
            f"{prefix}_low_text_page_ratio": f"{self.low_text_page_ratio:.4f}",
            f"{prefix}_image_page_ratio": f"{self.image_page_ratio:.4f}",
            f"{prefix}_scanned_page_ratio": f"{self.scanned_page_ratio:.4f}",
            f"{prefix}_is_scanned": str(self.is_scanned).lower(),
            f"{prefix}_needs_ocr": str(self.needs_ocr).lower(),
            f"{prefix}_text_quality": self.quality,
        }


def assess_pdf_text_layer(
    pages: Iterable[PdfPageTextStats],
    *,
    low_text_chars: int = 80,
    image_page_area_ratio: float = 0.45,
    scanned_page_ratio_threshold: float = 0.60,
) -> PdfTextLayerReport:
    """汇总单页统计 → 整篇 PDF 的文本层报告。

    决策表：
    - is_scanned：扫描页占比 ≥60% 或（极低字符密度 + 高图片占比）
    - needs_ocr：扫描件 / 平均字符过少 / 低文本页 ≥50%
    - quality：扫描件或字符密度 <40 → poor；其余 needs_ocr 或低文本页 ≥25% → warning；其它 ok
    """
    page_stats = list(pages)
    page_count = len(page_stats)
    text_chars = sum(page.text_chars for page in page_stats)
    text_chars_per_page = text_chars / max(1, page_count)
    low_text_pages = sum(1 for page in page_stats if page.text_chars < low_text_chars)
    image_pages = sum(1 for page in page_stats if page.image_area_ratio >= image_page_area_ratio)
    scanned_pages = sum(
        1 for page in page_stats if page.text_chars < low_text_chars and page.image_area_ratio >= image_page_area_ratio
    )
    low_text_page_ratio = low_text_pages / max(1, page_count)
    image_page_ratio = image_pages / max(1, page_count)
    scanned_page_ratio = scanned_pages / max(1, page_count)

    # 扫描件判断综合页级字符密度和可读文本比例，单页异常不能直接代表整篇文档。
    is_scanned = bool(
        page_count > 0
        and (
            scanned_page_ratio >= scanned_page_ratio_threshold
            or (text_chars_per_page < 40 and image_page_ratio >= 0.30)
        )
    )
    needs_ocr = bool(is_scanned or text_chars_per_page < low_text_chars or low_text_page_ratio >= 0.50)
    if is_scanned or text_chars_per_page < 40:
        quality = "poor"
    elif needs_ocr or low_text_page_ratio >= 0.25:
        quality = "warning"
    else:
        quality = "ok"

    return PdfTextLayerReport(
        pages=page_count,
        text_chars=text_chars,
        text_chars_per_page=round(text_chars_per_page, 2),
        low_text_pages=low_text_pages,
        low_text_page_ratio=round(low_text_page_ratio, 4),
        image_pages=image_pages,
        image_page_ratio=round(image_page_ratio, 4),
        scanned_pages=scanned_pages,
        scanned_page_ratio=round(scanned_page_ratio, 4),
        is_scanned=is_scanned,
        needs_ocr=needs_ocr,
        quality=quality,
    )


def helper_page_image_area_ratio(page: Any) -> float:
    """计算单页所有图片块面积与页面面积之比。

    容错处理：get_text 抛异常返回 0.0，避免单页错误中断整篇评估。
    """
    page_area = max(1.0, float(page.rect.width * page.rect.height))
    image_area = 0.0
    try:
        blocks = page.get_text("dict").get("blocks", [])
    except Exception:  # noqa: BLE001
        return 0.0
    for block in blocks:
        if block.get("type") != 1:
            continue
        bbox = block.get("bbox") or []
        if len(bbox) != 4:
            continue
        width = max(0.0, float(bbox[2]) - float(bbox[0]))
        height = max(0.0, float(bbox[3]) - float(bbox[1]))
        image_area += width * height
    return round(min(1.0, image_area / page_area), 4)


def inspect_pdf_text_layer(pdf_path: str | Path) -> PdfTextLayerReport:
    """打开 PDF，逐页统计文本与图片面积，汇总为 PdfTextLayerReport。"""
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required to inspect PDF text layers.") from exc

    doc = fitz.open(str(pdf_path))
    try:
        pages = []
        for index, page in enumerate(doc, start=1):
            text = page.get_text("text", sort=True) or ""
            pages.append(
                PdfPageTextStats(
                    page_number=index,
                    text_chars=len("".join(text.split())),
                    image_area_ratio=helper_page_image_area_ratio(page),
                )
            )
        return assess_pdf_text_layer(pages)
    finally:
        doc.close()