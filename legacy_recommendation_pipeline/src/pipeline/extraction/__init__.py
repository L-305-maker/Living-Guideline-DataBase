"""抽取流水线阶段的稳定对外入口。

这里通过延迟导入暴露路由、规则抽取和版本构建函数，避免上层脚本直接依赖
具体子模块路径，也减少 CLI 启动时的无关导入成本。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "VersionBuildInput": ("src.pipeline.extraction.versioning.recommendation_version_builder", "VersionBuildInput"),
    "build_version": ("src.pipeline.extraction.versioning.recommendation_version_builder", "build_version"),
    "build_versions": ("src.pipeline.extraction.versioning.recommendation_version_builder", "build_versions"),
    "build_versions_file": ("src.pipeline.extraction.versioning.recommendation_version_builder", "build_versions_file"),
    "extract_evidence_items_file": ("src.pipeline.extraction.evidence.item_extractor", "extract_file"),
    "extract_grade_candidates_file": ("src.pipeline.extraction.grade.candidate_extractor", "extract_file"),
    "extract_pico_questions_file": ("src.pipeline.extraction.pico.question_extractor", "extract_file"),
    "extract_recommendation_candidates_file": ("src.pipeline.extraction.recommendation.candidate_extractor", "extract_file"),
    "route_blocks": ("src.pipeline.extraction.routing.candidate_router", "route_blocks"),
    "route_blocks_file": ("src.pipeline.extraction.routing.candidate_router", "route_file"),
    "routed_block": ("src.pipeline.extraction.routing.candidate_router", "routed_block"),
    "run_source_span_first_pipeline": ("src.pipeline.extraction.source_span.first_pass", "run_source_span_first_pipeline"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
