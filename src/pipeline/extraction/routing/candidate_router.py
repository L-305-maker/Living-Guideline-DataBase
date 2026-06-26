from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

from src.common.process_jsonl import iter_jsonl, write_jsonl
from src.common.record_quality import block_record_quality_context
from src.pipeline.extraction.recommendation.patterns import has_action_pattern

# 模块职责:
# 把解析阶段给出的宽松候选 hint 转成具体抽取队列。解析器偏召回，
# 本路由器负责提高精度: 抑制噪声、保留可修复推荐文本，并阻止背景块进入抽取器。


JsonDict = Dict[str, object]

# 这些噪声如果同时带有推荐动作词，不能直接丢弃；先送修复队列保住召回。
REPAIR_REASONS = {
    "layout_glued_text",
    "table_or_figure_text",
    "url_heavy_or_identifier_text",
    "administrative_or_reference_text",
}

HARD_SKIP_RE = re.compile(
    r"\b("
    r"references|acknowledg|copyright|supplementary|registry|trial\s+registration|"
    r"corresponding\s+author|contact|affiliations?|conflicts?\s+of\s+interest|"
    r"disclaimer|fda|author\s+contributions?"
    r")\b",
    re.I,
)
LOW_VALUE_SECTION_RE = re.compile(r"\b(?:references|appendix|supplement|acknowledg|methods?|evidence|rationale|discussion)\b", re.I)
RECOMMENDATION_SECTION_RE = re.compile(r"\b(?:recommendations?|guideline statements?|key recommendations?)\b", re.I)
URL_RE = re.compile(r"(?:https?://|www\.|doi\.org|doi:\s*10\.)", re.I)
TABLE_TEXT_RE = re.compile(r"\b(?:figure|fig\.|table)\s*[0-9A-Z]?\b|class\s*of\s*recommendation|level\s*of\s*evidence", re.I)
LAYOUT_GLUE_RE = re.compile(r"(?:\.\.){2,}|[A-Za-z]{18,}|[a-z][A-Z][a-z]")

ROUTE_OUTPUTS = {
    "recommendation": "recommendation_blocks",
    "layout_repair": "layout_repair_candidates",
    "grade": "grade_blocks",
    "pico": "pico_blocks",
    "evidence": "evidence_blocks",
    "background": "background_blocks",
    "skipped": "skipped_blocks",
}


def block_hints(block: JsonDict) -> List[str]:
    """读取 structure parser 写入的候选抽取 hint。"""

    hints = block.get("candidate_hints")
    return [str(hint) for hint in hints] if isinstance(hints, list) else []


def block_quality(block: JsonDict) -> JsonDict:
    """读取块级质量元数据；字段不存在时返回空对象。"""

    quality = block.get("quality")
    return quality if isinstance(quality, dict) else {}


def record_context(block: JsonDict) -> JsonDict:
    """读取随 block 传递下来的记录级质量上下文。"""

    context = block_record_quality_context(block)
    return {
        "cleaning_status": str(context.get("record_cleaning_status") or ""),
        "record_type": str(context.get("record_type") or ""),
        "quality_flags": set(context.get("record_quality_flags") or []),
    }


def _block_text(block: JsonDict) -> str:
    return str(block.get("text") or "")


def _section_text(block: JsonDict) -> str:
    path = block.get("section_path")
    if isinstance(path, list):
        return " ".join(str(item) for item in path)
    return str(path or block.get("heading") or "")


# 汇总块级噪声原因，用于决定 recommendation 块是抽取、跳过还是进入 layout repair 池。
def block_noise_reasons(block: JsonDict) -> List[str]:
    """识别会让推荐抽取不安全的文本污染原因。"""

    text = _block_text(block)
    section = _section_text(block)
    reasons: List[str] = []
    if HARD_SKIP_RE.search(text) or HARD_SKIP_RE.search(section):
        reasons.append("administrative_or_reference_text")
    if URL_RE.search(text):
        reasons.append("url_heavy_or_identifier_text")
    if TABLE_TEXT_RE.search(text) or str(block.get("block_type") or "").lower() == "table":
        reasons.append("table_or_figure_text")
    if LAYOUT_GLUE_RE.search(text) and len(text) > 80:
        reasons.append("layout_glued_text")
    if len(text) > 700 and not has_action_pattern(text):
        reasons.append("long_without_clear_action")
    return reasons


def _route_score(action: str) -> float:
    if action == "skip":
        return 0.1
    if action == "needs_layout_repair":
        return 0.35
    if action == "sentence_only":
        return 0.65
    return 0.85


def _recommendation_action(text: str, section: str, reasons: List[str]) -> str:
    """决定带 recommendation hint 的块应该抽取、修复、复核还是跳过。"""

    has_action = has_action_pattern(text)
    if has_action and REPAIR_REASONS.intersection(reasons):
        return "needs_layout_repair"
    if "administrative_or_reference_text" in reasons or "url_heavy_or_identifier_text" in reasons:
        return "skip"
    if "layout_glued_text" in reasons or "table_or_figure_text" in reasons:
        return "skip"
    if len(text) > 700 and has_action:
        if LOW_VALUE_SECTION_RE.search(section):
            reasons.append("low_value_section_with_action")
        # 长块仍有明确动作词时只抽句子，避免整段背景或证据说明污染推荐文本。
        return "sentence_only"
    if has_action:
        return "normal_extract"
    if LOW_VALUE_SECTION_RE.search(section):
        reasons.append("low_value_section_without_clear_action")
        return "needs_quality_review"
    reasons.append("no_clear_action_recommendation")
    return "skip"


# 对 recommendation hint 做二次质量门：含动作词但结构污染的块不丢弃，转入 layout repair。
def route_block_for_recommendation(block: JsonDict) -> JsonDict:
    """在分配队列前，对 recommendation hint 执行专门质量门。"""

    text = _block_text(block)
    section = _section_text(block)
    reasons = block_noise_reasons(block)
    context = record_context(block)

    if context["cleaning_status"] and context["cleaning_status"] != "ready":
        reasons.append(f"record_cleaning_status_{context['cleaning_status']}")
        return {
            "route_action": "skip",
            "noise_reasons": reasons,
            "quality_score": 0.05,
        }

    if context["record_type"] == "paper" and not RECOMMENDATION_SECTION_RE.search(section):
        reasons.append("paper_record_recommendation_suppressed")
        return {
            "route_action": "skip",
            "noise_reasons": reasons,
            "quality_score": 0.1,
        }

    action = _recommendation_action(text, section, reasons)
    return {
        "route_action": action,
        "noise_reasons": reasons,
        "quality_score": _route_score(action),
    }


def _skip_route(reason: str, queues: List[str] | None = None, **extra: object) -> JsonDict:
    return {
        "task_types": [],
        "primary_task": "skip",
        "method_suggestion": "none",
        "priority": "low",
        "reason": reason,
        "skip": True,
        "queues": queues or ["skipped"],
        **extra,
    }


def _layout_repair_route(queues: List[str] | None = None, **extra: object) -> JsonDict:
    return {
        "task_types": [],
        "primary_task": "layout_repair",
        "method_suggestion": "layout_repair_later",
        "priority": "medium",
        "reason": "recommendation_needs_layout_repair",
        "skip": False,
        "queues": queues or ["layout_repair"],
        **extra,
    }


def _background_route() -> JsonDict:
    return {
        "task_types": [],
        "primary_task": "background",
        "method_suggestion": "none",
        "priority": "low",
        "reason": "no_candidate_hint",
        "skip": False,
        "queues": ["background"],
    }


def _add_recommendation_queue(rec_route: JsonDict, task_types: List[str], queues: List[str]) -> None:
    """根据推荐路由动作加入推荐抽取队列或版面修复队列。"""

    action = rec_route.get("route_action")
    if action == "needs_layout_repair":
        queues.append("layout_repair")
    elif action != "skip":
        task_types.append("recommendation_extraction")
        queues.append("recommendation")


def _add_hint_queues(hints: List[str], rec_route: JsonDict) -> tuple[List[str], List[str]]:
    """把 parser hint 翻译成抽取任务类型和输出队列名。"""

    task_types: List[str] = []
    queues: List[str] = []
    if "recommendation" in hints:
        _add_recommendation_queue(rec_route, task_types, queues)
    for hint, task, queue in [
        ("grade", "grade_extraction", "grade"),
        ("pico", "pico_extraction", "pico"),
        ("evidence", "evidence_extraction", "evidence"),
    ]:
        if hint in hints:
            task_types.append(task)
            queues.append(queue)
    return task_types, queues


def _route_without_queue(rec_route: JsonDict) -> JsonDict:
    # rec_route 为空说明没有任何候选 hint；这是正常背景块，不是错误。
    if rec_route.get("route_action") == "skip":
        return _skip_route("recommendation_quality_gate", **rec_route)
    if rec_route.get("route_action") == "needs_layout_repair":
        return _layout_repair_route(**rec_route)
    return _background_route()


def _primary_task(task_types: List[str]) -> str:
    if "recommendation_extraction" in task_types:
        return "recommendation_extraction"
    if "grade_extraction" in task_types:
        return "grade_extraction"
    return task_types[0]


def _method_for_task(primary_task: str) -> str:
    if primary_task in {"recommendation_extraction", "grade_extraction", "pico_extraction"}:
        return "llm"
    return "medcpt_or_biobert_later"


# 将单个 SourceBlock 分配到 recommendation/grade/pico/evidence/background/repair 队列。
def route_for_block(block: JsonDict) -> JsonDict:
    """返回单个 SourceBlock 的路由决策。"""

    hints = block_hints(block)
    quality = block_quality(block)
    rec_route = route_block_for_recommendation(block) if "recommendation" in hints else {}

    if quality.get("skip_candidate_extraction"):
        return _skip_route(str(quality.get("skip_reason") or "quality_skip"))

    task_types, queues = _add_hint_queues(hints, rec_route)
    if not queues:
        return _route_without_queue(rec_route)

    # 只有 repair 队列、没有抽取任务时，不能继续按 task_types[0] 推断主任务。
    if not task_types and rec_route.get("route_action") == "needs_layout_repair":
        return _layout_repair_route(queues=queues, **rec_route)

    primary_task = _primary_task(task_types)
    priority = "high" if primary_task in {"recommendation_extraction", "grade_extraction"} else "medium"
    return {
        "task_types": task_types,
        "primary_task": primary_task,
        "method_suggestion": _method_for_task(primary_task),
        "priority": priority,
        "reason": "_and_".join(hints) + "_hint",
        "skip": False,
        "queues": queues,
        **rec_route,
    }


def routed_block(block: JsonDict) -> JsonDict:
    """在不修改输入对象的前提下，为 block 附加 route 元数据。"""

    out = dict(block)
    out["route"] = route_for_block(block)
    return out


# 批量路由并按输出队列聚合，同一个块可以进入多个非 recommendation 队列。
def route_blocks(blocks: Iterable[JsonDict]) -> Dict[str, List[JsonDict]]:
    """批量路由 SourceBlock，并按输出队列分组。"""

    routed: Dict[str, List[JsonDict]] = {name: [] for name in ROUTE_OUTPUTS}
    for block in blocks:
        item = routed_block(block)
        route = item.get("route")
        queues = route.get("queues") if isinstance(route, dict) else []
        for queue in queues if isinstance(queues, list) else ["background"]:
            queue_name = str(queue)
            if queue_name in routed:
                routed[queue_name].append(item)
    return routed


# 生成路由统计，按唯一 block 计算主任务、优先级和方法分布，避免多队列重复计数。
def summarize_routes(routed: Dict[str, List[JsonDict]], input_path: str | Path) -> JsonDict:
    """生成路由统计，并保证同一 block 只计数一次。"""

    priority_counts: Counter[str] = Counter()
    method_counts: Counter[str] = Counter()
    primary_task_counts: Counter[str] = Counter()
    unique_blocks: Dict[str, JsonDict] = {}

    for rows in routed.values():
        for row in rows:
            block_id = str(row.get("block_id") or "")
            if block_id and block_id not in unique_blocks:
                unique_blocks[block_id] = row

    for row in unique_blocks.values():
        route = row.get("route")
        if not isinstance(route, dict):
            continue
        priority_counts[str(route.get("priority") or "unknown")] += 1
        method_counts[str(route.get("method_suggestion") or "unknown")] += 1
        primary_task_counts[str(route.get("primary_task") or "unknown")] += 1

    return {
        "input_path": str(input_path),
        "total_unique_blocks": len(unique_blocks),
        "queue_counts": {ROUTE_OUTPUTS[name]: len(rows) for name, rows in routed.items()},
        "priority_counts": dict(priority_counts),
        "method_counts": dict(method_counts),
        "primary_task_counts": dict(primary_task_counts),
    }


def route_file(input_path: str | Path, output_dir: str | Path, prefix: str = "") -> JsonDict:
    """路由 SourceBlock JSONL，并写出各队列对应的 JSONL 文件。"""

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_prefix = prefix or input_path.stem.replace("_blocks", "")

    routed = route_blocks(iter_jsonl(input_path))
    for queue, rows in routed.items():
        output_name = ROUTE_OUTPUTS[queue]
        write_jsonl(output_dir / f"{file_prefix}_{output_name}.jsonl", rows)

    summary = summarize_routes(routed, input_path)
    write_jsonl(output_dir / f"{file_prefix}_route_summary.jsonl", [summary])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route SourceBlock JSONL into candidate extraction task queues.")
    parser.add_argument("--input", required=True, help="Input SourceBlock JSONL path.")
    parser.add_argument("--output-dir", required=True, help="Directory for routed queue JSONL files.")
    parser.add_argument("--prefix", default="", help="Optional output file prefix.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = route_file(args.input, args.output_dir, prefix=args.prefix)
    print(
        "blocks={total_unique_blocks} queues={queue_counts} methods={method_counts}".format(
            **summary
        )
    )


if __name__ == "__main__":
    main()
