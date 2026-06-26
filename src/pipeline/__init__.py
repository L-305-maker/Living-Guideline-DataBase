"""Pipeline 顶层入口。

这里只导出候选生成流水线的公开入口；正式发布流程在 `src.pipeline.publish` 中维护。
"""

from src.pipeline.orchestration import PipelineRunResult, run_pipeline

__all__ = ["PipelineRunResult", "run_pipeline"]
