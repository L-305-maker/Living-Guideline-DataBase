"""Optional OCRmyPDF integration for scanned PDF ingestion."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from shutil import which


@dataclass(frozen=True)
class OcrResult:
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
    """Run OCRmyPDF if it is installed and return a structured result."""
    # 外部命令调用集中在此边界，返回值同时记录是否执行、输出路径和可诊断错误。

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
