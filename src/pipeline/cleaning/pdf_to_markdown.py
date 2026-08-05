# PDF → Markdown 的 evidence 流水线入口。
#
# 真实实现在 src.pipeline.cleaning.pdf_to_md.convert_all / convert_pdf，
# 这里只做 alias，便于 src.pipeline.cleaning 对外暴露稳定接口。
"""PDF to Markdown entry points for the evidence-building pipeline.

This module keeps the src pipeline aligned with the implementation that is
already exercised under src.pipeline.cleaning.pdf_to_md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.pipeline.cleaning.pdf_to_md import convert_all as _convert_all
from src.pipeline.cleaning.pdf_to_md import convert_pdf


def convert_pdfs(
    raw_pdf_dir: str | Path,
    markdown_raw_dir: str | Path,
    manifest_path: str | Path,
    *,
    ocr_mode: str = "auto",
    ocr_output_dir: str | Path | None = None,
    ocr_languages: str = "chi_sim+eng",
    document_kind: str = "guideline",
    append: bool = False,
) -> dict[str, Any]:
    """把 raw_pdf_dir 中的 PDF 转成 front-matter Markdown，并写 manifest。

    参数：
    - ocr_mode="auto"：仅对扫描版 PDF 触发 OCR；"never" 跳过；"force" 全部强制 OCR
    - ocr_languages：OCRmyPDF 的语言列表（默认 chi_sim+eng）
    - document_kind：写入 front-matter 的 document_kind 字段
    - append=True：保留既有 markdown_raw，仅追加新文档（用于增量重建）
    """
    return _convert_all(
        raw_pdf_dir,
        markdown_raw_dir,
        manifest_path,
        ocr_mode=ocr_mode,
        ocr_output_dir=ocr_output_dir,
        ocr_languages=ocr_languages,
        document_kind=document_kind,
        append=append,
    )


# 旧名兼容：convert_all 等价于 convert_pdfs。
convert_all = convert_pdfs