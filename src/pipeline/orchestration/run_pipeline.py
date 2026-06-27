"""候选生成流水线的总调度器。

这个文件只负责编排“从 origin JSONL 到候选产物”的流程：
清洗 -> 质量门 -> 结构解析 -> 路由 -> 四类抽取 -> 版本构建 -> 审核队列 -> 质量报告。
它不会把候选结果发布成正式 Recommendation，也不会直接写 PostgreSQL。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from src.common.process_jsonl import iter_jsonl, write_jsonl
from src.pipeline.cleaning.quality_gate import QualityGatePaths, route_file as route_cleaned_file
from src.pipeline.cleaning.source_cleaner import clean_file
from src.pipeline.extraction.evidence.item_extractor import extract_file as extract_evidence_file
from src.pipeline.extraction.grade.candidate_extractor import extract_file as extract_grade_file
from src.pipeline.extraction.pico.question_extractor import extract_file as extract_pico_file
from src.pipeline.extraction.recommendation.candidate_extractor import extract_file as extract_recommendation_file
from src.pipeline.extraction.routing.candidate_router import route_file as route_blocks_file
from src.pipeline.extraction.versioning.recommendation_version_builder import build_versions_file
from src.pipeline.llm_review.queue.builder import build_queue_file
from src.pipeline.parsing.structure_parser import parse_file
from src.pipeline.update.version_diff import build_update_logs, build_update_logs_file
from src.pipeline.quality.reports import block_quality_report
from src.pipeline.quality.reports import candidate_overall_quality_report
from src.pipeline.quality.reports import candidate_quality_report
from src.pipeline.quality.reports import evidence_quality_report
from src.pipeline.quality.reports import pico_quality_report


JsonDict = Dict[str, Any]
PIPELINE_VERSION = "guideline_review_pipeline_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def count_jsonl(path: str | Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    return sum(1 for _ in iter_jsonl(path))


def write_json(path: str | Path, payload: JsonDict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass(frozen=True)
class PipelinePaths:
    """一次候选生成流水线运行会稳定输出的所有 JSONL 和报告路径。"""

    run_dir: Path
    cleaned: Path
    ready_cleaned: Path
    layout_repair: Path
    parse_failed: Path
    skipped: Path
    cleaning_report: Path
    blocks: Path
    routes: Path
    recommendation_blocks: Path
    grade_blocks: Path
    pico_blocks: Path
    evidence_blocks: Path
    recommendation_candidates: Path
    recommendation_traces: Path
    grade_candidates: Path
    grade_traces: Path
    pico_questions: Path
    pico_traces: Path
    evidence_items: Path
    evidence_traces: Path
    recommendation_versions: Path
    recommendation_versions_report: Path
    update_logs: Path
    update_logs_report: Path
    llm_queue: Path
    llm_queue_summary: Path
    quality_reports: Path
    block_quality_summary: Path
    block_quality_samples: Path
    recommendation_quality_summary: Path
    recommendation_quality_samples: Path
    candidate_overall_quality_summary: Path
    candidate_overall_quality_samples: Path
    pico_quality_summary: Path
    pico_quality_samples: Path
    evidence_quality_summary: Path
    evidence_quality_samples: Path
    manifest: Path

    @classmethod
    def for_run(cls, run_dir: str | Path, *, route_prefix: str = "flow") -> "PipelinePaths":
        # PipelinePaths 是本流水线的“文件路径合同”：
        # 上一步写到哪里，下一步就从对应路径读取，避免路径散落在各阶段逻辑里。
        root = Path(run_dir)
        routes = root / "routes"
        return cls(
            run_dir=root,
            cleaned=root / "cleaned.jsonl",
            ready_cleaned=root / "cleaned.ready.jsonl",
            layout_repair=root / "cleaned.needs_layout_repair.jsonl",
            parse_failed=root / "cleaned.parse_failed.jsonl",
            skipped=root / "cleaned.skipped.jsonl",
            cleaning_report=root / "cleaned.quality_report.jsonl",
            blocks=root / "blocks.jsonl",
            routes=routes,
            recommendation_blocks=routes / f"{route_prefix}_recommendation_blocks.jsonl",
            grade_blocks=routes / f"{route_prefix}_grade_blocks.jsonl",
            pico_blocks=routes / f"{route_prefix}_pico_blocks.jsonl",
            evidence_blocks=routes / f"{route_prefix}_evidence_blocks.jsonl",
            recommendation_candidates=root / "recommendation_candidates.jsonl",
            recommendation_traces=root / "recommendation_traces.jsonl",
            grade_candidates=root / "grade_candidates.jsonl",
            grade_traces=root / "grade_traces.jsonl",
            pico_questions=root / "pico_questions.jsonl",
            pico_traces=root / "pico_traces.jsonl",
            evidence_items=root / "evidence_items.jsonl",
            evidence_traces=root / "evidence_traces.jsonl",
            recommendation_versions=root / "recommendation_versions.jsonl",
            recommendation_versions_report=root / "recommendation_versions_report.jsonl",
            update_logs=root / "update_logs.jsonl",
            update_logs_report=root / "update_logs_report.jsonl",
            llm_queue=root / "llm_review_queue.jsonl",
            llm_queue_summary=root / "llm_review_queue_summary.jsonl",
            quality_reports=root / "quality_reports",
            block_quality_summary=root / "quality_reports" / "block_quality_summary.jsonl",
            block_quality_samples=root / "quality_reports" / "block_quality_samples.jsonl",
            recommendation_quality_summary=root / "quality_reports" / "recommendation_quality_summary.jsonl",
            recommendation_quality_samples=root / "quality_reports" / "recommendation_quality_samples.jsonl",
            candidate_overall_quality_summary=root / "quality_reports" / "candidate_overall_quality_summary.jsonl",
            candidate_overall_quality_samples=root / "quality_reports" / "candidate_overall_quality_samples.jsonl",
            pico_quality_summary=root / "quality_reports" / "pico_quality_summary.jsonl",
            pico_quality_samples=root / "quality_reports" / "pico_quality_samples.jsonl",
            evidence_quality_summary=root / "quality_reports" / "evidence_quality_summary.jsonl",
            evidence_quality_samples=root / "quality_reports" / "evidence_quality_samples.jsonl",
            manifest=root / "run_manifest.json",
        )

    def as_dict(self) -> JsonDict:
        return {field_name: str(getattr(self, field_name)) for field_name in self.__dataclass_fields__}


@dataclass
class PipelineRunResult:
    """候选生成流水线完成后的机器可读摘要，用于 run_manifest 和后续审计。"""

    run_id: str
    input_path: str
    run_dir: str
    started_at: str
    finished_at: str = ""
    stage_summaries: JsonDict = field(default_factory=dict)
    artifact_counts: JsonDict = field(default_factory=dict)
    artifacts: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "pipeline_version": PIPELINE_VERSION,
            "run_id": self.run_id,
            "input_path": self.input_path,
            "run_dir": self.run_dir,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "stage_summaries": self.stage_summaries,
            "artifact_counts": self.artifact_counts,
            "artifacts": self.artifacts,
        }


def artifact_counts(paths: PipelinePaths) -> JsonDict:
    """统计每个关键产物的行数，用于 run_manifest 和命令行摘要。"""

    return {
        "cleaned": count_jsonl(paths.cleaned),
        "ready_cleaned": count_jsonl(paths.ready_cleaned),
        "layout_repair": count_jsonl(paths.layout_repair),
        "parse_failed": count_jsonl(paths.parse_failed),
        "skipped": count_jsonl(paths.skipped),
        "blocks": count_jsonl(paths.blocks),
        "recommendation_blocks": count_jsonl(paths.recommendation_blocks),
        "grade_blocks": count_jsonl(paths.grade_blocks),
        "pico_blocks": count_jsonl(paths.pico_blocks),
        "evidence_blocks": count_jsonl(paths.evidence_blocks),
        "recommendation_candidates": count_jsonl(paths.recommendation_candidates),
        "recommendation_traces": count_jsonl(paths.recommendation_traces),
        "grade_candidates": count_jsonl(paths.grade_candidates),
        "grade_traces": count_jsonl(paths.grade_traces),
        "pico_questions": count_jsonl(paths.pico_questions),
        "pico_traces": count_jsonl(paths.pico_traces),
        "evidence_items": count_jsonl(paths.evidence_items),
        "evidence_traces": count_jsonl(paths.evidence_traces),
        "recommendation_versions": count_jsonl(paths.recommendation_versions),
        "recommendation_versions_report": count_jsonl(paths.recommendation_versions_report),
        "update_logs": count_jsonl(paths.update_logs),
        "update_logs_report": count_jsonl(paths.update_logs_report),
        "llm_queue": count_jsonl(paths.llm_queue),
        "llm_queue_summary": count_jsonl(paths.llm_queue_summary),
        "quality_report_files": len(list(paths.quality_reports.glob("*.jsonl"))) if paths.quality_reports.exists() else 0,
    }


def write_quality_reports(paths: PipelinePaths) -> JsonDict:
    """为主流水线产物写出标准质量报告，帮助定位抽取和路由质量问题。"""

    paths.quality_reports.mkdir(parents=True, exist_ok=True)
    block_report = block_quality_report.build_report(paths.blocks)
    block_quality_report.write_report(block_report, paths.block_quality_summary, paths.block_quality_samples)

    recommendation_report = candidate_quality_report.build_report(paths.recommendation_candidates)
    candidate_quality_report.write_report(
        recommendation_report,
        paths.recommendation_quality_summary,
        paths.recommendation_quality_samples,
    )

    overall_report = candidate_overall_quality_report.build_report(
        paths.recommendation_candidates,
        paths.grade_candidates,
        [paths.recommendation_traces, paths.grade_traces],
    )
    candidate_overall_quality_report.write_report(
        overall_report,
        paths.candidate_overall_quality_summary,
        paths.candidate_overall_quality_samples,
    )

    pico_report = pico_quality_report.build_report(paths.pico_questions)
    pico_quality_report.write_report(pico_report, paths.pico_quality_summary, paths.pico_quality_samples)

    evidence_report = evidence_quality_report.build_report(paths.evidence_items, paths.evidence_traces)
    evidence_quality_report.write_report(evidence_report, paths.evidence_quality_summary, paths.evidence_quality_samples)

    return {
        "block_quality": block_report["summary"],
        "recommendation_quality": recommendation_report["summary"],
        "candidate_overall_quality": overall_report["summary"],
        "pico_quality": pico_report["summary"],
        "evidence_quality": evidence_report["summary"],
    }


def run_pipeline(
    input_path: str | Path,
    output_root: str | Path,
    *,
    run_id: str | None = None,
    route_prefix: str = "flow",
    previous_versions_input: str | Path | None = None,
) -> PipelineRunResult:
    """对一个 origin JSONL 运行本地候选生成流水线。

    注意：这个入口只负责清洗、解析、抽取、候选版本构建和审核队列生成；
    它不会把 RecommendationVersion 发布为正式 Recommendation 当前态。
    正式发布必须走 `pipeline.publish.recommendation_publisher`。
    """

    started_at = utc_now()
    run_id = run_id or default_run_id()
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = PipelinePaths.for_run(run_dir, route_prefix=route_prefix)

    # result 是这次运行的“审计账本”：记录输入、输出目录、阶段摘要和产物计数。
    # 最终会写入 run_manifest.json，方便之后定位数据在哪个阶段丢失或被分流。
    result = PipelineRunResult(
        run_id=run_id,
        input_path=str(input_path),
        run_dir=str(run_dir),
        started_at=started_at,
        artifacts=paths.as_dict(),
    )

    # 下面是主流水线调用链。每一步基本都是“读一个 JSONL，写一个 JSONL/报告”。
    # 注意：这里产出的是候选数据，不是正式发布数据。
    result.stage_summaries["cleaning"] = {"cleaned_records": clean_file(input_path, paths.cleaned)}
    result.stage_summaries["cleaning_gate"] = route_cleaned_file(
        QualityGatePaths(
            paths.cleaned,
            paths.ready_cleaned,
            paths.layout_repair,
            paths.parse_failed,
            paths.skipped,
            paths.cleaning_report,
        )
    )
    result.stage_summaries["parsing"] = {"blocks": parse_file(paths.ready_cleaned, paths.blocks)}
    result.stage_summaries["routing"] = route_blocks_file(paths.blocks, paths.routes, prefix=route_prefix)
    result.stage_summaries["recommendation_extraction"] = extract_recommendation_file(
        paths.recommendation_blocks,
        paths.recommendation_candidates,
        paths.recommendation_traces,
    )
    result.stage_summaries["grade_extraction"] = extract_grade_file(
        paths.grade_blocks,
        paths.recommendation_candidates,
        paths.grade_candidates,
        paths.grade_traces,
    )
    result.stage_summaries["pico_extraction"] = extract_pico_file(
        paths.pico_blocks,
        paths.pico_questions,
        paths.pico_traces,
    )
    result.stage_summaries["evidence_extraction"] = extract_evidence_file(
        paths.evidence_blocks,
        paths.evidence_items,
        paths.evidence_traces,
        picos_input=paths.pico_questions,
        recommendations_input=paths.recommendation_candidates,
    )
    result.stage_summaries["recommendation_versioning"] = build_versions_file(
        recommendations_input=paths.recommendation_candidates,
        grades_input=paths.grade_candidates,
        picos_input=paths.pico_questions,
        evidence_input=paths.evidence_items,
        versions_output=paths.recommendation_versions,
        report_output=paths.recommendation_versions_report,
    )
    if previous_versions_input:
        result.stage_summaries["update_logs"] = build_update_logs_file(
            previous_versions_input,
            paths.recommendation_versions,
            paths.update_logs,
            paths.update_logs_report,
        )
    else:
        update_logs, update_report = build_update_logs([], list(iter_jsonl(paths.recommendation_versions)))
        write_jsonl(paths.update_logs, update_logs)
        write_jsonl(paths.update_logs_report, [update_report])
        result.stage_summaries["update_logs"] = update_report
    result.stage_summaries["llm_queue"] = build_queue_file(
        paths.recommendation_candidates,
        paths.grade_candidates,
        paths.llm_queue,
        paths.llm_queue_summary,
    )
    result.stage_summaries["quality_reports"] = write_quality_reports(paths)
    result.finished_at = utc_now()
    result.artifact_counts = artifact_counts(paths)
    write_json(paths.manifest, result.to_dict())
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the guideline review pipeline over an origin JSONL file.")
    parser.add_argument("--input", required=True, help="Input origin JSONL path.")
    parser.add_argument("--output-root", required=True, help="Directory where run folders are created.")
    parser.add_argument("--run-id", default=None, help="Optional stable run id.")
    parser.add_argument("--previous-versions-input", default=None, help="Optional previous RecommendationVersion JSONL path for update-log diffing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pipeline(args.input, args.output_root, run_id=args.run_id, previous_versions_input=args.previous_versions_input)
    print(
        "run_id={run_id} ready={ready} blocks={blocks} recommendations={recommendations} picos={picos} evidence={evidence} versions={versions} updates={updates} queue={queue} manifest={manifest}".format(
            run_id=result.run_id,
            ready=result.artifact_counts.get("ready_cleaned", 0),
            blocks=result.artifact_counts.get("blocks", 0),
            recommendations=result.artifact_counts.get("recommendation_candidates", 0),
            picos=result.artifact_counts.get("pico_questions", 0),
            evidence=result.artifact_counts.get("evidence_items", 0),
            versions=result.artifact_counts.get("recommendation_versions", 0),
            updates=result.artifact_counts.get("update_logs", 0),
            queue=result.artifact_counts.get("llm_queue", 0),
            manifest=result.artifacts.get("manifest", ""),
        )
    )


if __name__ == "__main__":
    main()
