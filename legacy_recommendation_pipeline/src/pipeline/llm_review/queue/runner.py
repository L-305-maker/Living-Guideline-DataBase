"""LLM 复核文件：构建复核队列、prompt、响应解析和自动质检，让候选结果进入人工/模型辅助复核流程。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from src.pipeline.llm_review.clients.llm_client import DEFAULT_ENV_FILE, LLMRequest, load_env_file, post_llm_request, resolve_api_key
from src.pipeline.llm_review.parsing.output_records import (
    LLMOutputInput,
    ModelTraceInput,
    build_llm_output_record,
    build_model_trace,
    input_entity_id_for_item,
    trace_input_entity_type,
    trace_task_type,
)
from src.pipeline.llm_review.prompts.builder import (
    CERTAINTY_VALUES,
    DIRECTION_VALUES,
    GRADE_DOMAIN_VALUES,
    GRADE_OUTPUT_TEMPLATE,
    GRADE_RESULT_SCHEMA,
    GRADE_SYSTEM_VALUES,
    PUBLICATION_BIAS_VALUES,
    RECOMMENDATION_OUTPUT_TEMPLATE,
    RECOMMENDATION_RESULT_SCHEMA,
    STRENGTH_VALUES,
    build_grade_prompt,
    build_prompt,
    build_prompt_record,
    build_recommendation_prompt,
    compact_candidate_state,
    compact_grade_state,
    compact_json,
    compact_recommendation_state,
)
from src.pipeline.llm_review.parsing.response_parser import extract_json_object, extract_response_text
from src.pipeline.llm_review.parsing.validators import (
    validate_grade_result,
    validate_llm_result,
    validate_llm_result_for_item,
    validate_recommendation_result,
)
from src.common.extraction_common import JsonDict, utc_now
from src.common.process_jsonl import iter_jsonl, write_jsonl, write_jsonl_obj

# 模块职责:
# 执行或 dry-run LLM 复核队列项。runner 支持可恢复、流式写入，
# 因此长批次 LLM 复核可以安全中断，不会丢失已经验证通过的队列项。


@dataclass(frozen=True)
class LLMRunConfig:
    """试运行 prompt 和真实调用模式共用的 LLM 请求配置。"""

    prompt_version: str = "llm_review_prompt_v1"
    model: str = "gpt-4.1-mini"
    api_url: str = "https://api.openai.com/v1/responses"
    api_key_env: str = "OPENAI_API_KEY"
    env_file: str | Path = DEFAULT_ENV_FILE
    api_format: str = "responses"
    temperature: float = 0.0
    timeout_seconds: int = 60
    max_retries: int = 3
    retry_base_seconds: float = 1.0
    max_response_bytes: int = 2_000_000


@dataclass(frozen=True)
class LLMRunPaths:
    """一次队列运行使用的输入/输出文件路径。"""

    queue_input: str | Path
    outputs_output: str | Path
    traces_output: str | Path
    summary_output: str | Path


@dataclass(frozen=True)
class LLMQueueSelection:
    """用于选择部分复核队列的可选过滤条件。"""

    priorities: Optional[Sequence[str]] = None
    task_types: Optional[Sequence[str]] = None
    limit: Optional[int] = None


@dataclass(frozen=True)
class LLMStreamOptions:
    """控制 LLM 调用时的可恢复流式写入行为。"""

    resume: bool = False
    append_output: bool = False
    progress_every: int = 50
    flush_every: int = 1


@dataclass
class LLMStreamState:
    """队列运行过程中持续更新的可变计数状态。"""

    selected_count: int
    skipped_existing_success: int
    attempted_items: int = 0
    status_counts: Counter[str] = field(default_factory=Counter)
    task_counts: Counter[str] = field(default_factory=Counter)
    last_queue_id: str = ""
    started_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class LLMStreamRun:
    """传给流式执行辅助函数的不可变运行参数包。"""

    paths: LLMRunPaths
    config: LLMRunConfig
    stream: LLMStreamOptions


@dataclass(frozen=True)
class LLMCallResult:
    """一次 LLM 调用的原始响应、解析结果、校验错误和耗时。"""

    raw_response: Any
    response_text: str
    parsed_result: JsonDict | None
    validation_errors: list[str]
    runtime_ms: int
    error_message: str | None


@dataclass(frozen=True)
class PromptFileRequest:
    """试运行生成 prompt 文件时使用的请求对象。"""

    queue_input: str | Path
    prompts_output: str | Path
    summary_output: str | Path
    selection: LLMQueueSelection = LLMQueueSelection()
    prompt_version: str = "llm_review_prompt_v1"


@dataclass(frozen=True)
class LLMOutputSummaryInput:
    """汇总已完成 LLM 输出文件所需的输入集合。"""

    outputs: list[JsonDict]
    traces: list[JsonDict]
    paths: LLMRunPaths
    model: str


def select_queue_items(
    rows: Iterable[JsonDict],
    priorities: Optional[Sequence[str]] = None,
    task_types: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> list[JsonDict]:
    """按优先级、任务类型和可选数量上限选择队列行。"""

    if limit is not None and limit < 0:
        raise ValueError("limit must be greater than or equal to 0")
    priority_set = set(priorities or [])
    task_type_set = set(task_types or [])
    selected: list[JsonDict] = []
    if limit == 0:
        return selected
    for row in rows:
        if priority_set and row.get("priority") not in priority_set:
            continue
        if task_type_set and row.get("task_type") not in task_type_set:
            continue
        selected.append(row)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def execute_llm_call(item: JsonDict, config: LLMRunConfig, prompt_record: JsonDict) -> LLMCallResult:
    """调用配置好的 LLM API，并校验解析后的结果。"""

    started = time.perf_counter()
    raw_response: Any = None
    response_text = ""
    parsed_result: JsonDict | None = None
    validation_errors: list[str] = []
    error_message: str | None = None
    try:
        raw_response = post_llm_request(
            LLMRequest(
                prompt=prompt_record["prompt"],
                model=config.model,
                api_url=config.api_url,
                api_key_env=config.api_key_env,
                env_file=config.env_file,
                api_format=config.api_format,
                temperature=config.temperature,
                timeout_seconds=config.timeout_seconds,
                max_retries=config.max_retries,
                retry_base_seconds=config.retry_base_seconds,
                max_response_bytes=config.max_response_bytes,
            )
        )
        response_text = extract_response_text(raw_response)
        parsed_result = extract_json_object(response_text)
        validation_errors = validate_llm_result_for_item(str(item.get("task_type") or ""), parsed_result, item)
    except Exception as exc:
        error_message = str(exc)

    runtime_ms = int((time.perf_counter() - started) * 1000)
    return LLMCallResult(raw_response, response_text, parsed_result, validation_errors, runtime_ms, error_message)


def call_one_queue_item(item: JsonDict, config: LLMRunConfig) -> tuple[JsonDict, JsonDict]:
    """构建 prompt、执行 LLM 调用，并返回输出记录和 ModelTrace。"""

    prompt_record = build_prompt_record(item, config.prompt_version)
    result = execute_llm_call(item, config, prompt_record)
    trace = build_model_trace(
        ModelTraceInput(
            prompt_record=prompt_record,
            item=item,
            model=config.model,
            api_url=config.api_url,
            api_format=config.api_format,
            temperature=config.temperature,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            retry_base_seconds=config.retry_base_seconds,
            max_response_bytes=config.max_response_bytes,
            raw_response=result.raw_response,
            parsed_result=result.parsed_result,
            validation_errors=result.validation_errors,
            runtime_ms=result.runtime_ms,
            error_message=result.error_message,
        )
    )
    output = build_llm_output_record(
        LLMOutputInput(
            prompt_record=prompt_record,
            item=item,
            raw_response=result.raw_response,
            response_text=result.response_text,
            parsed_result=result.parsed_result,
            validation_errors=result.validation_errors,
            model_trace_id=trace["model_trace_id"],
            runtime_ms=result.runtime_ms,
            error_message=result.error_message,
        )
    )
    return output, trace


def call_queue_items_with_config(
    paths: LLMRunPaths,
    config: LLMRunConfig,
    selection: LLMQueueSelection = LLMQueueSelection(),
    stream: LLMStreamOptions = LLMStreamOptions(),
) -> JsonDict:
    """使用显式配置和流式选项调用选中的队列项。"""

    selected = select_queue_items(
        iter_jsonl(paths.queue_input),
        priorities=selection.priorities,
        task_types=selection.task_types,
        limit=selection.limit,
    )
    completed_queue_ids = successful_queue_ids(paths.outputs_output) if stream.resume else set()
    return call_selected_items_streaming(
        selected=selected,
        run=LLMStreamRun(paths, config, stream=LLMStreamOptions(
            resume=stream.resume,
            append_output=stream.append_output or stream.resume,
            progress_every=stream.progress_every,
            flush_every=stream.flush_every,
        )),
        completed_queue_ids=completed_queue_ids,
    )


def call_queue_items(
    paths: LLMRunPaths,
    config: LLMRunConfig = LLMRunConfig(),
    selection: LLMQueueSelection = LLMQueueSelection(),
    stream: LLMStreamOptions = LLMStreamOptions(),
) -> JsonDict:
    """调用队列项的公共便捷入口。"""

    return call_queue_items_with_config(
        paths=paths,
        config=config,
        selection=selection,
        stream=stream,
    )


def queue_id(row: JsonDict) -> str:
    """从队列项或输出行中读取稳定 queue_id。"""

    return str(row.get("queue_id") or "")


def successful_queue_ids(outputs_output: str | Path) -> set[str]:
    """读取已验证成功的 queue_id，供 resume 模式跳过。"""

    output_path = Path(outputs_output)
    if not output_path.exists():
        return set()
    ids: set[str] = set()
    for row in iter_jsonl(output_path):
        if row.get("status") == "validated":
            qid = queue_id(row)
            if qid:
                ids.add(qid)
    return ids


def empty_run_summary(
    paths: LLMRunPaths,
    model: str,
    state: LLMStreamState,
) -> JsonDict:
    """为所有选中项都被跳过的运行构建摘要。"""

    return {
        "queue_input": str(paths.queue_input),
        "outputs_output": str(paths.outputs_output),
        "traces_output": str(paths.traces_output),
        "model": model,
        "selected_items": state.selected_count,
        "skipped_existing_success": state.skipped_existing_success,
        "attempted_items": 0,
        "output_items": 0,
        "trace_items": 0,
        "status_counts": {},
        "task_counts": {},
        "last_queue_id": "",
        "started_at": state.started_at,
        "updated_at": utc_now(),
        "completed_at": utc_now(),
        "status": "completed_no_items_to_call",
    }


def build_stream_summary(
    paths: LLMRunPaths,
    model: str,
    state: LLMStreamState,
    completed: bool = False,
) -> JsonDict:
    """构建流式 LLM 调用的进度摘要或完成摘要。"""

    summary: JsonDict = {
        "queue_input": str(paths.queue_input),
        "outputs_output": str(paths.outputs_output),
        "traces_output": str(paths.traces_output),
        "model": model,
        "selected_items": state.selected_count,
        "skipped_existing_success": state.skipped_existing_success,
        "attempted_items": state.attempted_items,
        "output_items": state.attempted_items,
        "trace_items": state.attempted_items,
        "status_counts": dict(state.status_counts),
        "task_counts": dict(state.task_counts),
        "last_queue_id": state.last_queue_id,
        "started_at": state.started_at,
        "updated_at": utc_now(),
        "status": "completed" if completed else "running",
    }
    if completed:
        summary["completed_at"] = utc_now()
    return summary


def write_summary(path: str | Path, summary: JsonDict) -> None:
    """写出单行 JSONL 摘要文件。"""

    write_jsonl(path, [summary])


def validate_stream_options(stream: LLMStreamOptions) -> None:
    """写文件前拒绝非法的流式写入选项。"""

    if stream.progress_every < 0:
        raise ValueError("progress_every must be greater than or equal to 0")
    if stream.flush_every <= 0:
        raise ValueError("flush_every must be greater than 0")


def prepare_stream_paths(paths: LLMRunPaths) -> tuple[Path, Path]:
    """创建输出目录，并返回具体 output/trace 路径。"""

    outputs_path = Path(paths.outputs_output)
    traces_path = Path(paths.traces_output)
    outputs_path.parent.mkdir(parents=True, exist_ok=True)
    traces_path.parent.mkdir(parents=True, exist_ok=True)
    return outputs_path, traces_path


def write_stream_outputs(
    to_call: list[JsonDict],
    run: LLMStreamRun,
    state: LLMStreamState,
) -> None:
    """逐条调用队列项，并把 output/trace 流式写入磁盘。"""

    mode = "a" if run.stream.append_output else "w"
    output_path, trace_path = prepare_stream_paths(run.paths)
    with output_path.open(mode, encoding="utf-8") as output_handle, trace_path.open(mode, encoding="utf-8") as trace_handle:
        for item in to_call:
            output, trace = call_one_queue_item(item, run.config)
            write_jsonl_obj(output_handle, output)
            write_jsonl_obj(trace_handle, trace)

            state.attempted_items += 1
            state.status_counts[str(output.get("status") or "unknown")] += 1
            state.task_counts[str(output.get("task_type") or "unknown")] += 1
            state.last_queue_id = queue_id(output) or queue_id(item)

            if state.attempted_items % run.stream.flush_every == 0:
                output_handle.flush()
                trace_handle.flush()
            if run.stream.progress_every and state.attempted_items % run.stream.progress_every == 0:
                write_summary(run.paths.summary_output, build_stream_summary(run.paths, run.config.model, state))

        output_handle.flush()
        trace_handle.flush()


def call_selected_items_streaming(
    selected: list[JsonDict],
    run: LLMStreamRun,
    completed_queue_ids: set[str],
) -> JsonDict:
    """以支持 resume 的流式语义执行选中的队列项。"""

    validate_stream_options(run.stream)
    selected_count = len(selected)
    to_call = [item for item in selected if queue_id(item) not in completed_queue_ids]
    skipped_existing_success = selected_count - len(to_call)
    state = LLMStreamState(selected_count=selected_count, skipped_existing_success=skipped_existing_success)

    if not to_call:
        summary = empty_run_summary(run.paths, run.config.model, state)
        write_summary(run.paths.summary_output, summary)
        return summary

    write_stream_outputs(to_call, run, state)
    summary = build_stream_summary(run.paths, run.config.model, state, completed=True)
    write_summary(run.paths.summary_output, summary)
    return summary


def summarize_llm_outputs(summary_input: LLMOutputSummaryInput) -> JsonDict:
    """汇总已完成的 LLM 输出行和 trace 行。"""

    status_counts: dict[str, int] = defaultdict(int)
    task_counts: dict[str, int] = defaultdict(int)
    for row in summary_input.outputs:
        status_counts[str(row.get("status") or "unknown")] += 1
        task_counts[str(row.get("task_type") or "unknown")] += 1
    return {
        "queue_input": str(summary_input.paths.queue_input),
        "outputs_output": str(summary_input.paths.outputs_output),
        "traces_output": str(summary_input.paths.traces_output),
        "model": summary_input.model,
        "output_items": len(summary_input.outputs),
        "trace_items": len(summary_input.traces),
        "status_counts": dict(status_counts),
        "task_counts": dict(task_counts),
        "created_at": utc_now(),
    }


def summarize_prompts(records: list[JsonDict], queue_input: str | Path) -> JsonDict:
    """在不调用 LLM API 的情况下汇总试运行 prompt 记录。"""

    priority_counts: dict[str, int] = defaultdict(int)
    task_counts: dict[str, int] = defaultdict(int)
    for record in records:
        priority_counts[str(record.get("priority") or "unknown")] += 1
        task_counts[str(record.get("task_type") or "unknown")] += 1
    return {
        "queue_input": str(queue_input),
        "prompt_items": len(records),
        "priority_counts": dict(priority_counts),
        "task_counts": dict(task_counts),
        "status": "dry_run_prompts_created",
        "created_at": utc_now(),
    }


def create_prompt_file(request: PromptFileRequest) -> JsonDict:
    """创建试运行/离线检查用的 prompt JSONL 记录。"""

    selected = select_queue_items(
        iter_jsonl(request.queue_input),
        priorities=request.selection.priorities,
        task_types=request.selection.task_types,
        limit=request.selection.limit,
    )
    prompt_records = [build_prompt_record(item, request.prompt_version) for item in selected]
    summary = summarize_prompts(prompt_records, request.queue_input)
    summary["prompt_version"] = request.prompt_version
    summary["prompts_output"] = str(request.prompts_output)
    write_jsonl(request.prompts_output, prompt_records)
    write_jsonl(request.summary_output, [summary])
    return summary


def parse_csv(value: Optional[str]) -> Optional[list[str]]:
    """把逗号分隔的 CLI 过滤条件解析成列表。"""

    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or call LLM review prompts from an LLM review queue.")
    parser.add_argument("--mode", choices=["dry-run", "call"], default="dry-run")
    parser.add_argument("--queue-input", required=True)
    parser.add_argument("--prompts-output")
    parser.add_argument("--outputs-output")
    parser.add_argument("--traces-output")
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--priorities", default="P0", help="Comma-separated priorities. Default: P0.")
    parser.add_argument("--task-types", default="", help="Comma-separated task types. Empty means all task types.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum prompts to create. Use 0 for no limit.")
    parser.add_argument("--prompt-version", default="llm_review_prompt_v1")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--api-url", default="https://api.openai.com/v1/responses")
    parser.add_argument("--api-format", choices=["responses", "chat_completions"], default="responses")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-base-seconds", type=float, default=1.0)
    parser.add_argument("--max-response-bytes", type=int, default=2_000_000)
    parser.add_argument("--resume", action="store_true", help="Skip queue_ids that already have validated output rows.")
    parser.add_argument("--append-output", action="store_true", help="Append output and trace rows instead of replacing files.")
    parser.add_argument("--progress-every", type=int, default=50, help="Rewrite summary after every N attempted calls. Use 0 to only write at the end.")
    parser.add_argument("--flush-every", type=int, default=1, help="Flush output and trace files after every N attempted calls.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limit = None if args.limit == 0 else args.limit
    if args.mode == "dry-run":
        if not args.prompts_output:
            raise ValueError("--prompts-output is required in dry-run mode")
        summary = create_prompt_file(
            PromptFileRequest(
                queue_input=args.queue_input,
                prompts_output=args.prompts_output,
                summary_output=args.summary_output,
                selection=LLMQueueSelection(
                    priorities=parse_csv(args.priorities),
                    task_types=parse_csv(args.task_types),
                    limit=limit,
                ),
                prompt_version=args.prompt_version,
            )
        )
        print("prompt_items={prompt_items} priority={priority_counts} tasks={task_counts}".format(**summary))
        return

    if not args.outputs_output or not args.traces_output:
        raise ValueError("--outputs-output and --traces-output are required in call mode")
    summary = call_queue_items(
        paths=LLMRunPaths(args.queue_input, args.outputs_output, args.traces_output, args.summary_output),
        config=LLMRunConfig(
            prompt_version=args.prompt_version,
            model=args.model,
            api_url=args.api_url,
            api_key_env=args.api_key_env,
            env_file=args.env_file,
            api_format=args.api_format,
            temperature=args.temperature,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            retry_base_seconds=args.retry_base_seconds,
            max_response_bytes=args.max_response_bytes,
        ),
        selection=LLMQueueSelection(
            priorities=parse_csv(args.priorities),
            task_types=parse_csv(args.task_types),
            limit=limit,
        ),
        stream=LLMStreamOptions(
            resume=args.resume,
            append_output=args.append_output,
            progress_every=args.progress_every,
            flush_every=args.flush_every,
        ),
    )
    print("output_items={output_items} status={status_counts} tasks={task_counts}".format(**summary))


if __name__ == "__main__":
    main()

