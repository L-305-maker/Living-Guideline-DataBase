# 跑 evidence-library 完整数据生产流水线。
#
# 流水线阶段（每阶段都把中间产物写到 data/evidence 子目录，方便分阶段复跑）：
#   raw PDF → markdown_raw → markdown_clean → sections → chunks → JSONL
#
# 设计原则（与 README 一致）：
# - retrieval-first：不抽取 Recommendation / PICOQuestion / GRADE 等结构化实体，
#   这些由下游 Agent 推理层负责。
# - PostgreSQL 入库与向量化不属于本流水线，由 src.storage 单独命令执行。
"""Run the evidence-library build pipeline.

Current pipeline:

raw PDF -> Markdown -> clean Markdown -> complete blocks -> chunks -> PostgreSQL-ready JSONL artifacts.

It deliberately does not extract Recommendation, PICOQuestion, GRADE, or
EvidenceItem entities. PostgreSQL ingestion and vectorization are run by
``src.storage`` commands after the artifacts are built.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.pipeline.cleaning.evidence_pipeline import run_evidence_pipeline
from src.utils.io import DATA_DIR


def run_pipeline(
    *,
    data_dir: str | Path = DATA_DIR,
    raw_pdf_dir: str | Path | None = None,
    consensus_pdf_dir: str | Path | None = None,
    skip_pdf_to_markdown: bool = False,
    ocr_mode: str = "auto",
    ocr_languages: str = "chi_sim+eng",
) -> dict[str, Any]:
    """执行 evidence 流水线全流程的入口。

    参数说明：
    - data_dir：产物根目录，默认 data/evidence
    - raw_pdf_dir / consensus_pdf_dir：PDF 输入目录；缺省时按 data_dir/raw_pdf 与 ../consensus 解析
    - skip_pdf_to_markdown=True：复用已有 markdown_raw，跳过 PDF→MD 阶段
    - ocr_mode："never" 跳过 OCR / "auto" 仅对扫描版触发 / "force" 强制全部 OCR
    - ocr_languages：OCRmyPDF 语言列表，常见 "chi_sim+eng"

    返回：evidence_pipeline.run_evidence_pipeline 的 manifest 字典。
    """
    return run_evidence_pipeline(
        data_dir=data_dir,
        raw_pdf_dir=raw_pdf_dir,
        consensus_pdf_dir=consensus_pdf_dir,
        skip_pdf_to_markdown=skip_pdf_to_markdown,
        ocr_mode=ocr_mode,
        ocr_languages=ocr_languages,
    )


def parse_args() -> argparse.Namespace:
    """解析命令行参数，对外仅暴露流水线级别的开关。

    细粒度参数（如 OCR 引擎选择、并发数等）交给各阶段内部环境变量，避免 CLI 表面膨胀。
    """
    parser = argparse.ArgumentParser(description="Build PostgreSQL-ready guideline evidence artifacts from raw PDFs.")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Evidence artifact root.")
    parser.add_argument("--raw-pdf-dir", default=None, help="Raw PDF directory. Defaults to DATA_DIR/raw_pdf.")
    parser.add_argument("--consensus-pdf-dir", default=None, help="Expert consensus PDF directory. Defaults to DATA_DIR/../consensus.")
    parser.add_argument("--skip-pdf-to-markdown", action="store_true", help="Reuse existing markdown_raw files.")
    parser.add_argument("--ocr-mode", choices=["never", "auto", "force"], default="auto", help="OCRmyPDF mode for scanned PDFs.")
    parser.add_argument("--ocr-languages", default="chi_sim+eng", help="OCR language list passed to OCRmyPDF.")
    return parser.parse_args()


def main() -> None:
    """CLI 入口：python -m src.pipeline.run_pipeline [flags]。"""
    args = parse_args()
    result = run_pipeline(
        data_dir=args.data_dir,
        raw_pdf_dir=args.raw_pdf_dir,
        consensus_pdf_dir=args.consensus_pdf_dir,
        skip_pdf_to_markdown=args.skip_pdf_to_markdown,
        ocr_mode=args.ocr_mode,
        ocr_languages=args.ocr_languages,
    )
    # ensure_ascii=False 让中文字段在 stdout 可读；indent=2 便于人眼 review。
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()