from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from src.common.extraction_common import normalize_text, utc_now
from src.domain.candidate import ModelTrace
from src.domain.common import stable_id


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class RuleTraceInput:
    rule_version: str
    task_type: str
    block: JsonDict
    parsed_output: JsonDict
    confidence: float
    parameters: JsonDict
    elapsed_ms: int
    success: bool = True
    error: str | None = None
    target_table: str | None = None
    target_entity_id: str | None = None


def compact(value: Any, max_chars: int = 240) -> str:
    return normalize_text(value)[:max_chars].strip(" ,;:-")


def source_order(row: JsonDict) -> int:
    if "source_order" in row:
        return _safe_int(row.get("source_order"))
    payload = row.get("normalized_payload")
    if isinstance(payload, dict):
        return _safe_int(payload.get("source_order"))
    return _safe_int(row.get("order"))


def block_id_from_payload(row: JsonDict) -> str:
    payload = row.get("normalized_payload")
    if isinstance(payload, dict):
        return str(payload.get("block_id") or "")
    return str(row.get("source_block_id") or row.get("block_id") or "")


def mean_confidence(rows: Iterable[JsonDict], field: str = "extraction_confidence") -> float:
    values = [float(row.get(field) or 0.0) for row in rows]
    return round(sum(values) / len(values), 4) if values else 0.0


def build_rule_trace(data: RuleTraceInput) -> ModelTrace:
    return ModelTrace(
        model_trace_id=stable_id("model_trace", data.rule_version, data.block.get("block_id"), data.task_type),
        task_type=data.task_type,  # type: ignore[arg-type]
        method="rule",
        model_name=data.rule_version,
        input_entity_type="block",
        input_entity_id=str(data.block.get("block_id") or ""),
        input_text=str(data.block.get("text") or ""),
        model_version=data.rule_version,
        raw_output={"candidate_hints": data.block.get("candidate_hints", [])},
        parsed_output=data.parsed_output,
        confidence=data.confidence,
        parameters=data.parameters,
        runtime_ms=data.elapsed_ms,
        success=data.success,
        error_message=data.error,
        target_table=data.target_table,
        target_entity_id=data.target_entity_id,
        created_at=utc_now(),
    )


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
