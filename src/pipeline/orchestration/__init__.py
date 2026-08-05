# evidence-library 流水线的统一入口。
#
# 该包负责：
# - CLI 参数解析（parse_args）
# - 转发到 evidence_pipeline.run_evidence_pipeline 执行各阶段
# PostgreSQL 入库与向量化由 src.storage 命令负责，本包不涉及。
"""Evidence-library pipeline orchestration entry points."""

from __future__ import annotations

from typing import Any

# 对外只暴露 run_pipeline；内部实现位于本目录的 run_pipeline.py。
__all__ = ["run_pipeline"]


def run_pipeline(*args: Any, **kwargs: Any) -> dict[str, Any]:
    # 延迟 import：避免 src.pipeline 顶层就被 evidence_pipeline 拉起全部依赖，
    # 单元测试或独立调用 run_pipeline 时只付出一份 import 成本。
    from src.pipeline.orchestration.run_pipeline import run_pipeline as _run_pipeline

    return _run_pipeline(*args, **kwargs)