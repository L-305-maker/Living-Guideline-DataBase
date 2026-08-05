# 可选的 OCRmyPDF 集成：扫描版 PDF 的本地 OCR 入口。
#
# 触发条件：src.pipeline.ocr.pdf_quality.assess_pdf_text_layer 判定 needs_ocr=True 时。
# 优点：本地运行、零配额限制；缺点：扫描版 PDF 走 Tesseract，准确性低于云端 OCR / MinerU。
"""Optional OCRmyPDF integration for scanned PDF ingestion."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from shutil import which


@dataclass(frozen=True)
class OcrResult:
    """单次 OCR 调用结果（结构化、便于 audit 与失败重试）。

    engine：调用的引擎名（ocrmypdf / baidu / mineru / deepseek）
    applied：是否实际跑了 OCR（False 时通常因命令缺失或前置检查失败）
    input_pdf / output_pdf：输入输出 PDF 路径
    error：失败原因摘要；成功时为空字符串
    """
    engine: str
    applied: bool
    input_pdf: str
    output_pdf: str
    error: str = ""


def run_ocrmypdf(
    input_pdf: str | Path,
    output_pdf: str | Path,
    *,
    languages: str = "chi_sim+eng",
    redo_ocr: bool = False,
    force_ocr: bool = False,
    optimize: int = 1,
    timeout_seconds: int = 900,
) -> OcrResult:
    """调用本地 ocrmypdf 处理扫描版 PDF。

    默认参数偏向"已含文本层则跳过 OCR（--skip-text）"：
    - redo_ocr=True：删除已有文本层并重跑（删 --skip-text，加 --redo-ocr）
    - force_ocr=True：强制全页 OCR（删 --skip-text，加 --force-ocr）
    - optimize=N：图像压缩级别（0-3，1 是平衡）
    - timeout_seconds：单次超时（默认 15 分钟）

    返回：OcrResult 而非抛异常，便于上层聚合多个 PDF 的 OCR 状态。
    """
    executable = which("ocrmypdf")
    source = Path(input_pdf)
    target = Path(output_pdf)
    if executable is None:
        return OcrResult(
            engine="ocrmypdf",
            applied=False,
            input_pdf=str(source),
            output_pdf=str(target),
            error="ocrmypdf command was not found",
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        "--skip-text",
        "--rotate-pages",
        "--deskew",
        "--clean",
        "--oversample",
        "300",
        "--optimize",
        str(optimize),
        "-l",
        languages,
    ]
    if redo_ocr:
        command = [item for item in command if item != "--skip-text"]
        command.append("--redo-ocr")
    if force_ocr:
        command = [item for item in command if item != "--skip-text"]
        command.append("--force-ocr")
    command.extend([str(source), str(target)])
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds, check=False)
    except Exception as exc:  # noqa: BLE001
        return OcrResult("ocrmypdf", False, str(source), str(target), str(exc))
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "").strip()
        return OcrResult("ocrmypdf", False, str(source), str(target), message[:800])
    return OcrResult("ocrmypdf", True, str(source), str(target))