"""按流水线运行目录生成可交付给 reviewer 的人工审核包。

本模块不重新定义审核规则，只负责把一次 run 目录中的 candidates、PICO、evidence 和
LLM 复核队列收拢成稳定文件。这样可以在 RecommendationVersion 尚未生成时，先把抽取
结果交给人工审核，审核通过后再回流到版本构建和入库流程。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from src.common.process_jsonl import iter_jsonl, write_jsonl
from src.pipeline.review.association_review import AssociationReviewOptions, build_association_review_package
from src.pipeline.review.manual_review_gate import ReviewQueueOptions, build_queue
from src.pipeline.review.rule_assisted_backfill import BackfillInputs


JsonDict = Dict[str, Any]

PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2}


@dataclass(frozen=True)
class EntityQueueConfig:
    """描述一个实体队列的输入文件、实体类型和输出文件。"""

    entity_type: str
    input_name: str
    output_name: str


ENTITY_QUEUE_CONFIGS = (
    EntityQueueConfig("recommendation_candidate", "recommendation_candidates.jsonl", "recommendation_review_queue.jsonl"),
    EntityQueueConfig("grade_candidate", "grade_candidates.jsonl", "grade_review_queue.jsonl"),
    EntityQueueConfig("pico_question", "pico_questions.jsonl", "pico_review_queue.jsonl"),
    EntityQueueConfig("evidence_item", "evidence_items.jsonl", "evidence_review_queue.jsonl"),
)


def optional_jsonl(path: Path) -> Iterable[JsonDict]:
    """读取可选 JSONL；文件不存在时返回空迭代，便于部分流程也能生成审核包。"""

    if not path.exists():
        return []
    return iter_jsonl(path)


def priority_sort_key(row: JsonDict) -> tuple[int, int, str, str]:
    """按优先级、原文顺序和实体标识排序，让 reviewer 看到的队列更稳定。"""

    current_state = row.get("current_state") if isinstance(row.get("current_state"), dict) else {}
    try:
        source_order = int(current_state.get("source_order") or 0)
    except (TypeError, ValueError):
        source_order = 0
    priority = str(row.get("priority") or "P2")
    return (
        PRIORITY_ORDER.get(priority, 99),
        source_order,
        str(row.get("entity_type") or ""),
        str(row.get("entity_id") or ""),
    )


def sorted_queue(rows: Iterable[JsonDict]) -> List[JsonDict]:
    """对人工审核队列做确定性排序，避免同一批数据多次导出顺序漂移。"""

    return sorted(rows, key=priority_sort_key)


def tag_queue_items(rows: Iterable[JsonDict], queue_name: str) -> List[JsonDict]:
    """给合并队列追加来源文件名，方便 reviewer 回溯到单实体队列。"""

    tagged: List[JsonDict] = []
    for row in rows:
        item = dict(row)
        item["source_queue"] = queue_name
        tagged.append(item)
    return tagged


def limit_rows(rows: List[JsonDict], limit: int | None) -> List[JsonDict]:
    """按已排序顺序截取首批审核项；limit 小于等于 0 时表示不限制。"""

    if limit is None or limit <= 0:
        return rows
    return rows[:limit]


def review_group_key(row: JsonDict) -> str:
    """首批抽样时的分组键，优先按指南，其次按来源记录或论文分散队列。"""

    current_state = row.get("current_state") if isinstance(row.get("current_state"), dict) else {}
    for key in ["guideline_id", "record_id", "paper_id"]:
        value = str(current_state.get(key) or "")
        if value:
            return value
    return str(row.get("entity_id") or "")


def stratified_limit_rows(rows: List[JsonDict], limit: int | None) -> List[JsonDict]:
    """按指南/记录轮转截取首批审核项，避免某个来源独占第一批。"""

    if limit is None or limit <= 0 or len(rows) <= limit:
        return rows
    buckets: Dict[str, List[JsonDict]] = {}
    for row in rows:
        buckets.setdefault(review_group_key(row), []).append(row)
    keys = sorted(buckets, key=lambda key: priority_sort_key(buckets[key][0]))
    selected: List[JsonDict] = []
    while keys and len(selected) < limit:
        next_keys: List[str] = []
        for key in keys:
            bucket = buckets[key]
            selected.append(bucket.pop(0))
            if bucket:
                next_keys.append(key)
            if len(selected) >= limit:
                break
        keys = next_keys
    return selected


def build_entity_review_queue(
    run_dir: Path,
    output_dir: Path,
    config: EntityQueueConfig,
    options: ReviewQueueOptions,
    input_path: Path | None = None,
) -> tuple[List[JsonDict], JsonDict]:
    """为单类实体生成审核队列并写入审核包目录。"""

    source_path = input_path or run_dir / config.input_name
    queue, summary = build_queue(optional_jsonl(source_path), config.entity_type, options)
    queue = sorted_queue(queue)
    write_jsonl(output_dir / config.output_name, queue)
    summary["input_file"] = str(source_path)
    summary["output_file"] = config.output_name
    return queue, summary


def filter_llm_priority_queue(rows: Iterable[JsonDict], priorities: Sequence[str]) -> List[JsonDict]:
    """保留指定优先级的 LLM 复核项，作为第一批人工重点处理对象。"""

    allowed = set(priorities)
    filtered = [row for row in rows if str(row.get("priority") or "") in allowed]
    return sorted(filtered, key=lambda row: (PRIORITY_ORDER.get(str(row.get("priority") or "P2"), 99), str(row.get("queue_id") or "")))


def build_review_batch(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    confidence_threshold: float = 0.65,
    llm_priorities: Sequence[str] = ("P0", "P1"),
    max_first_pass_per_entity: int | None = 500,
    max_llm_priority: int | None = 1000,
    input_overrides: Dict[str, str | Path] | None = None,
    include_association_review: bool = True,
    association_batch_size: int = 100,
    max_association_recommendations: int | None = None,
    max_evidence_pico_reviews: int | None = None,
) -> JsonDict:
    """从一次 pipeline run 目录生成完整人工审核包，并返回汇总信息。"""

    run_path = Path(run_dir)
    batch_dir = Path(output_dir) if output_dir else run_path / "review_batch"
    batch_dir.mkdir(parents=True, exist_ok=True)

    options = ReviewQueueOptions(
        include_pending=True,
        include_unclear_fields=True,
        confidence_threshold=confidence_threshold,
    )
    entity_summaries: List[JsonDict] = []
    first_pass_items: List[JsonDict] = []
    overrides = input_overrides or {}

    for config in ENTITY_QUEUE_CONFIGS:
        override_path = overrides.get(config.entity_type)
        queue, summary = build_entity_review_queue(
            run_path,
            batch_dir,
            config,
            options,
            input_path=Path(override_path) if override_path else None,
        )
        entity_summaries.append(summary)
        entity_first_pass = [item for item in queue if item.get("priority") in {"P0", "P1"}]
        summary["first_pass_available_records"] = len(entity_first_pass)
        entity_first_pass = stratified_limit_rows(entity_first_pass, max_first_pass_per_entity)
        summary["first_pass_selected_records"] = len(entity_first_pass)
        first_pass_items.extend(tag_queue_items(entity_first_pass, config.output_name))

    llm_source = Path(overrides.get("llm_review_queue") or run_path / "llm_review_queue.jsonl")
    llm_priority_queue = filter_llm_priority_queue(optional_jsonl(llm_source), llm_priorities)
    llm_priority_total = len(llm_priority_queue)
    llm_priority_queue = limit_rows(llm_priority_queue, max_llm_priority)
    write_jsonl(batch_dir / "llm_priority_queue.jsonl", llm_priority_queue)

    first_pass_items = sorted_queue(first_pass_items)
    write_jsonl(batch_dir / "first_pass_review_queue.jsonl", first_pass_items)

    association_summary: JsonDict | None = None
    if include_association_review:
        association_inputs = BackfillInputs(
            recommendations=list(optional_jsonl(Path(overrides.get("recommendation_candidate") or run_path / "recommendation_candidates.jsonl"))),
            grades=list(optional_jsonl(Path(overrides.get("grade_candidate") or run_path / "grade_candidates.jsonl"))),
            picos=list(optional_jsonl(Path(overrides.get("pico_question") or run_path / "pico_questions.jsonl"))),
            evidence=list(optional_jsonl(Path(overrides.get("evidence_item") or run_path / "evidence_items.jsonl"))),
        )
        association_summary = build_association_review_package(
            association_inputs,
            batch_dir,
            AssociationReviewOptions(
                batch_size=association_batch_size,
                max_recommendation_records=max_association_recommendations,
                max_evidence_pico_records=max_evidence_pico_reviews,
            ),
        )

    summary = {
        "run_dir": str(run_path),
        "output_dir": str(batch_dir),
        "confidence_threshold": confidence_threshold,
        "llm_priorities": list(llm_priorities),
        "max_first_pass_per_entity": max_first_pass_per_entity,
        "max_llm_priority": max_llm_priority,
        "include_association_review": include_association_review,
        "association_batch_size": association_batch_size,
        "max_association_recommendations": max_association_recommendations,
        "max_evidence_pico_reviews": max_evidence_pico_reviews,
        "input_overrides": {key: str(value) for key, value in overrides.items()},
        "entity_queues": entity_summaries,
        "first_pass_review_records": len(first_pass_items),
        "llm_priority_total_records": llm_priority_total,
        "llm_priority_records": len(llm_priority_queue),
        "association_review": association_summary,
    }
    write_jsonl(batch_dir / "review_batch_summary.jsonl", [summary])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 pipeline run 目录生成人工审核包。")
    parser.add_argument("--run-dir", required=True, help="包含 candidates/PICO/evidence/LLM 队列的运行目录。")
    parser.add_argument("--output-dir", default=None, help="审核包输出目录；默认写到 run-dir/review_batch。")
    parser.add_argument("--recommendation-candidates-input", default=None, help="覆盖默认 recommendation_candidates.jsonl。")
    parser.add_argument("--grade-candidates-input", default=None, help="覆盖默认 grade_candidates.jsonl。")
    parser.add_argument("--pico-questions-input", default=None, help="覆盖默认 pico_questions.jsonl。")
    parser.add_argument("--evidence-items-input", default=None, help="覆盖默认 evidence_items.jsonl。")
    parser.add_argument("--llm-review-queue-input", default=None, help="覆盖默认 llm_review_queue.jsonl。")
    parser.add_argument("--confidence-threshold", type=float, default=0.65, help="低于该置信度的候选进入人工审核。")
    parser.add_argument(
        "--max-first-pass-per-entity",
        type=int,
        default=500,
        help="每类实体进入 first_pass_review_queue 的最大条数；小于等于 0 表示不限制。",
    )
    parser.add_argument(
        "--max-llm-priority",
        type=int,
        default=1000,
        help="进入 llm_priority_queue 的最大条数；小于等于 0 表示不限制。",
    )
    parser.add_argument("--skip-association-review", action="store_true", help="不生成关联复核队列。")
    parser.add_argument("--association-batch-size", type=int, default=100, help="关联复核队列的批次大小。")
    parser.add_argument("--max-association-recommendations", type=int, default=None, help="限制 Recommendation 关联复核条数。")
    parser.add_argument("--max-evidence-pico-reviews", type=int, default=None, help="限制 Evidence-PICO 复核条数。")
    parser.add_argument(
        "--llm-priority",
        action="append",
        dest="llm_priorities",
        choices=sorted(PRIORITY_ORDER),
        help="导出到 llm_priority_queue 的优先级，可重复传入；默认 P0/P1。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_overrides = {
        key: value
        for key, value in {
            "recommendation_candidate": args.recommendation_candidates_input,
            "grade_candidate": args.grade_candidates_input,
            "pico_question": args.pico_questions_input,
            "evidence_item": args.evidence_items_input,
            "llm_review_queue": args.llm_review_queue_input,
        }.items()
        if value
    }
    summary = build_review_batch(
        args.run_dir,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence_threshold,
        llm_priorities=tuple(args.llm_priorities or ("P0", "P1")),
        max_first_pass_per_entity=args.max_first_pass_per_entity,
        max_llm_priority=args.max_llm_priority,
        input_overrides=input_overrides,
        include_association_review=not args.skip_association_review,
        association_batch_size=args.association_batch_size,
        max_association_recommendations=args.max_association_recommendations,
        max_evidence_pico_reviews=args.max_evidence_pico_reviews,
    )
    print(f"Wrote review batch to {summary['output_dir']}")


if __name__ == "__main__":
    main()
