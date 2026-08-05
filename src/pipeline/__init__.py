# 指南证据数据生产流水线的入口包。
#
# 主调用入口：
#   python -m src.pipeline.run_pipeline   # 完整跑一遍 PDF → JSONL
# 或：
#   from src.pipeline import run_pipeline
# 各子模块（cleaning / ocr / quality）按职责拆分；本包只负责对外暴露统一接口。
"""Pipeline entry points for guideline evidence ingestion."""

from __future__ import annotations

# 转发到 orchestration 子模块的实际实现，避免循环 import；
# 这里是惰性 re-export，外部仍按 `from src.pipeline import run_pipeline` 使用。
from src.pipeline.orchestration import run_pipeline

__all__ = ["run_pipeline"]