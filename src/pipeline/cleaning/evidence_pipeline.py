# evidence-library 清洗流水线的核心编排。
#
# 流水线（retrieval-first）：
#   raw PDF → raw Markdown → clean Markdown → complete blocks → small chunks → JSONL
#
# 设计原则：
# - 不抽取 Recommendation / PICOQuestion / GRADE 等结构化实体（Agent 推理时的事）。
# - PostgreSQL 入库与向量化由 src.storage 命令在 JSONL 产物就绪后再跑。
"""Evidence-library cleaning pipeline.

The pipeline is intentionally retrieval-first:

raw PDF -> raw Markdown -> clean Markdown -> complete blocks -> small chunks.

It does not extract recommendations, PICO questions, GRADE candidates, or
evidence items. Those concepts are downstream reasoning concerns for the agent,
not entities produced during cleaning. PostgreSQL ingestion and vectorization
are handled by ``src.storage`` commands after these JSONL artifacts exist.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.pipeline.cleaning.block_chunker import chunk_blocks
from src.pipeline.cleaning.block_encoder import encode_blocks
from src.pipeline.cleaning.markdown_cleaner import clean_markdown_dir
from src.pipeline.cleaning.pdf_to_markdown import convert_pdfs
from src.retrieval.document_repr import build_document_representations
from src.utils.io import DATA_DIR


@dataclass(frozen=True)
class EvidencePipelinePaths:
    """一次 evidence-library 构建所用的全部产物路径。

    frozen=True：路径集合在一次运行中是只读的，避免外部修改导致 IO 状态不一致。
    字段：
    - data_dir：产物根目录（默认 data/evidence）
    - raw_pdf_dir / consensus_pdf_dir：两类输入 PDF 目录
    - markdown_raw_dir / markdown_clean_dir：转换与清洗产物
    - blocks_dir（磁盘上是 sections/）：完整源块
    - chunks_dir：切分后的小 chunk（含 all_chunks.jsonl 聚合）
    - raw_manifest：PDF → Markdown 阶段的 manifest
    - document_manifest：清洗阶段的 documents.jsonl
    - document_cards / document_views：多视图表征
    - run_manifest：本流水线最终的机器可读 manifest
    """

    data_dir: Path
    raw_pdf_dir: Path
    consensus_pdf_dir: Path
    markdown_raw_dir: Path
    markdown_clean_dir: Path
    blocks_dir: Path
    chunks_dir: Path
    raw_manifest: Path
    document_manifest: Path
    document_cards: Path
    document_views: Path
    run_manifest: Path

    @classmethod
    def from_data_dir(
        cls,
        data_dir: str | Path,
        raw_pdf_dir: str | Path | None = None,
        consensus_pdf_dir: str | Path | None = None,
    ) -> "EvidencePipelinePaths":
        """根据 data_dir 推导默认路径。

        关键默认：
        - consensus_pdf_dir 默认在 data_dir 的父目录下取 consensus/，因为共识文档通常与
          指南 PDF 放在仓库根的并列目录，便于跨项目共享。
        - blocks_dir 命名固定为 sections/，因为 src.storage 入库与 PG 表名 sections 已耦合。
        """
        root = Path(data_dir)
        return cls(
            data_dir=root,
            raw_pdf_dir=Path(raw_pdf_dir) if raw_pdf_dir else root / "raw_pdf",
            consensus_pdf_dir=Path(consensus_pdf_dir) if consensus_pdf_dir else root.parent / "consensus",
            markdown_raw_dir=root / "markdown_raw",
            markdown_clean_dir=root / "markdown_clean",
            blocks_dir=root / "sections",
            chunks_dir=root / "chunks",
            raw_manifest=root / "documents_raw.jsonl",
            document_manifest=root / "documents.jsonl",
            document_cards=root / "document_cards.jsonl",
            document_views=root / "document_views.jsonl",
            run_manifest=root / "evidence_pipeline_manifest.json",
        )

    def as_dict(self) -> dict[str, str]:
        """序列化为全字符串字典，方便直接写入 JSON manifest。"""
        return {key: str(value) for key, value in self.__dict__.items()}


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """原子写 JSON manifest：先确保父目录存在，ensure_ascii=False 保留中文。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_evidence_pipeline(
    *,
    data_dir: str | Path = DATA_DIR,
    raw_pdf_dir: str | Path | None = None,
    consensus_pdf_dir: str | Path | None = None,
    skip_pdf_to_markdown: bool = False,
    ocr_mode: str = "auto",
    ocr_languages: str = "chi_sim+eng",
) -> dict[str, Any]:
    """按四阶段跑 evidence 流水线（PDF→MD、清洗、编码、chunk），返回机器可读 manifest。

    阶段约定：
    1. pdf_to_markdown（可跳过）：把 raw_pdf 与 consensus_pdf 转成 markdown_raw；
       consensus 用 append=True 追加，避免覆盖 guideline 已写入的 records。
    2. clean_markdown：去噪 / 修复 front-matter，写 documents.jsonl。
    3. encode_blocks：按 heading 切 sections/*.jsonl。
    4. document_representations：生成 document_cards / document_views。
    5. chunk_blocks：把每个 section 切成小 chunk，写 chunks/*.jsonl + 聚合文件。

    最终返回的 manifest 包含 pipeline_version / goal / extracts_* 标记，便于外部脚本
    判断产物是否包含结构化实体（当前一律为 False）。
    """
    paths = EvidencePipelinePaths.from_data_dir(data_dir, raw_pdf_dir, consensus_pdf_dir)
    for directory in [
        paths.markdown_raw_dir,
        paths.markdown_clean_dir,
        paths.blocks_dir,
        paths.chunks_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    stages: dict[str, Any] = {}
    if skip_pdf_to_markdown:
        # 跳过 PDF→MD 时显式记录原因，便于 audit 时回溯。
        stages["pdf_to_markdown"] = {
            "skipped": True,
            "reason": "skip_pdf_to_markdown was set",
            "markdown_raw_dir": str(paths.markdown_raw_dir),
        }
    else:
        stages["pdf_to_markdown"] = convert_pdfs(
            paths.raw_pdf_dir,
            paths.markdown_raw_dir,
            paths.raw_manifest,
            ocr_mode=ocr_mode,
            ocr_output_dir=paths.data_dir / "ocr_pdf",
            ocr_languages=ocr_languages,
            document_kind="guideline",
        )
        if paths.consensus_pdf_dir.exists():
            # consensus 与 guideline 走同一个 markdown_raw 目录；
            # append=True 让 convert_pdfs 保留已存在的 guideline 记录。
            stages["consensus_pdf_to_markdown"] = convert_pdfs(
                paths.consensus_pdf_dir,
                paths.markdown_raw_dir,
                paths.raw_manifest,
                ocr_mode=ocr_mode,
                ocr_output_dir=paths.data_dir / "ocr_pdf",
                ocr_languages=ocr_languages,
                document_kind="consensus",
                append=True,
            )

    stages["clean_markdown"] = clean_markdown_dir(paths.markdown_raw_dir, paths.markdown_clean_dir, paths.document_manifest)
    stages["encode_blocks"] = encode_blocks(paths.markdown_clean_dir, paths.blocks_dir)
    stages["document_representations"] = build_document_representations(paths.markdown_clean_dir, paths.data_dir)
    stages["chunk_blocks"] = chunk_blocks(paths.markdown_clean_dir, paths.chunks_dir)

    manifest = {
        # pipeline_version 标识 manifest 模式：与 src.storage.verify 的兼容性检查配合。
        "pipeline_version": "evidence_cleaning_pg_v1",
        "goal": "clean and split guideline evidence for PostgreSQL ingestion",
        # 显式标记不抽取结构化实体，避免下游误判存在 Recommendation / PICO。
        "extracts_recommendations": False,
        "extracts_pico_questions": False,
        "artifacts": paths.as_dict(),
        "stages": stages,
    }
    write_json(paths.run_manifest, manifest)
    return manifest