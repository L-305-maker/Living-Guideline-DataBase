"""候选增强文件：对规则抽取得到的候选进行补充、归一化或后处理增强。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Callable

from src.common.extraction_common import JsonDict, utc_now

# 模块职责:
# 把已通过 LLM/自动质控的修正建议回填到候选记录上，同时保留原始 payload
# 和审计快照，方便后续人工复核或问题追踪。


def parsed_result(row: JsonDict) -> JsonDict:
    """读取 LLM 输出中已经解析好的结果对象；不存在时返回空对象。"""

    value = row.get("parsed_result")
    return value if isinstance(value, dict) else {}


def as_payload(row: JsonDict, field: str) -> JsonDict:
    """复制指定 payload 字段，避免调用方直接修改输入行。"""

    value = row.get(field)
    return deepcopy(value) if isinstance(value, dict) else {}


def index_by_id(rows, id_field: str) -> dict[str, JsonDict]:
    """按非空 ID 字段给 JSON 行建立索引，后出现的同 ID 行会覆盖前者。"""

    indexed: dict[str, JsonDict] = {}
    for row in rows:
        row_id = str(row.get(id_field) or "")
        if row_id:
            indexed[row_id] = row
    return indexed


def build_enhancement_payload(llm_output: JsonDict | None, qc: JsonDict | None, decision: str, method: str) -> JsonDict:
    """生成增强审计信息，说明候选记录是如何被 LLM/QC 修改的。"""

    return {
        "enhancement_method": method,
        "enhancement_status": decision,
        "llm_output_id": (llm_output or {}).get("llm_output_id", ""),
        "llm_model_trace_id": (llm_output or {}).get("model_trace_id", ""),
        "auto_qc_id": (qc or {}).get("auto_qc_id", ""),
        "auto_qc_decision": (qc or {}).get("auto_qc_decision", decision),
        "auto_qc_score": (qc or {}).get("auto_qc_score"),
        "auto_qc_reasons": (qc or {}).get("auto_qc_reasons", []),
        "recommended_action": (qc or {}).get("recommended_action", ""),
        "qc_policy_key": (qc or {}).get("qc_policy_key", "generic"),
        "qc_policy": (qc or {}).get("qc_policy", {}),
        "profile_context": (qc or {}).get("profile_context", {}),
        "enhanced_at": utc_now(),
    }


def compact_review_note(prefix: str, qc: JsonDict | None, llm_output: JsonDict | None) -> str:
    """根据 QC 与 LLM 拒绝原因生成紧凑的人工复核备注。"""

    result = parsed_result(llm_output or {})
    reasons = (qc or {}).get("auto_qc_reasons") or []
    reason_text = "; ".join(str(reason) for reason in reasons[:8])
    reject_reason = result.get("reject_reason")
    parts = [prefix]
    if (qc or {}).get("auto_qc_score") is not None:
        parts.append(f"auto_qc_score={(qc or {}).get('auto_qc_score')}")
    if reject_reason:
        parts.append(f"reject_reason={reject_reason}")
    if reason_text:
        parts.append(f"reasons={reason_text}")
    return " | ".join(parts)


def attach_payloads(
    candidate: JsonDict,
    llm_output: JsonDict | None,
    qc: JsonDict | None,
    decision: str,
    method: str,
    suggestion_builder: Callable[[JsonDict], JsonDict],
) -> JsonDict:
    """在候选副本上附加增强信息、LLM 建议和 QC 快照。"""

    enhanced = deepcopy(candidate)
    normalized = as_payload(enhanced, "normalized_payload")
    raw_payload = as_payload(enhanced, "raw_payload")
    normalized["enhancement"] = build_enhancement_payload(llm_output, qc, decision, method)
    if llm_output:
        normalized["llm_suggestion"] = suggestion_builder(llm_output)
    if qc:
        normalized["auto_qc"] = {
            "auto_qc_id": qc.get("auto_qc_id"),
            "decision": qc.get("auto_qc_decision"),
            "score": qc.get("auto_qc_score"),
            "reasons": qc.get("auto_qc_reasons", []),
            "recommended_action": qc.get("recommended_action"),
            "qc_policy_key": qc.get("qc_policy_key", "generic"),
            "qc_policy": qc.get("qc_policy", {}),
            "profile_context": qc.get("profile_context", {}),
        }
    if llm_output:
        raw_payload["llm_output_snapshot"] = {
            "llm_output_id": llm_output.get("llm_output_id"),
            "model_trace_id": llm_output.get("model_trace_id"),
            "status": llm_output.get("status"),
            "validation_errors": llm_output.get("validation_errors", []),
        }
    enhanced["normalized_payload"] = normalized
    enhanced["raw_payload"] = raw_payload
    return enhanced


def mark_needs_review(candidate: JsonDict, llm_output: JsonDict | None, qc: JsonDict | None, prefix: str) -> JsonDict:
    """返回标记为 needs_review 的候选副本，并写入简短原因。"""

    enhanced = deepcopy(candidate)
    enhanced["status"] = "needs_review"
    enhanced["review_note"] = compact_review_note(prefix, qc, llm_output)
    enhanced["updated_at"] = utc_now()
    return enhanced


def summarize_enhancements(rows: list[JsonDict]) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """统计增强后的最终状态、自动 QC 决策和 QC 策略分布。"""

    status_counts: Counter[str] = Counter()
    enhancement_counts: Counter[str] = Counter()
    policy_counts: Counter[str] = Counter()
    for row in rows:
        status_counts[str(row.get("status") or "unknown")] += 1
        enhancement = row.get("normalized_payload", {}).get("enhancement", {})
        if isinstance(enhancement, dict):
            enhancement_counts[str(enhancement.get("auto_qc_decision") or enhancement.get("enhancement_status") or "unknown")] += 1
            policy_counts[str(enhancement.get("qc_policy_key") or "generic")] += 1
    return dict(status_counts), dict(enhancement_counts), dict(policy_counts)

