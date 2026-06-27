"""版本更新文件：比较新旧推荐版本并生成 update log，记录推荐知识的变化。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.common.extraction_common import utc_now
from src.common.process_jsonl import iter_jsonl, write_jsonl
from src.domain.common import stable_id


JsonDict = Dict[str, Any]
COMPARE_FIELDS = ("recommendation_text", "direction", "strength", "certainty", "pico_id", "grade_candidate_id")


def recommendation_identity(row: JsonDict) -> str:
    return str(row.get("recommendation_id") or row.get("recommendation_candidate_id") or row.get("recommendation_version_id") or "")


def latest_by_recommendation(rows: Iterable[JsonDict]) -> Dict[str, JsonDict]:
    latest: Dict[str, JsonDict] = {}
    for row in rows:
        identity = recommendation_identity(row)
        if not identity:
            continue
        current = latest.get(identity)
        if current is None or _version_number(row) >= _version_number(current):
            latest[identity] = row
    return latest


def _version_number(row: JsonDict) -> int:
    raw = str(row.get("version_number") or "v0").lstrip("vV")
    try:
        return int(raw or 0)
    except ValueError:
        return 0


def changed_fields(previous: JsonDict, current: JsonDict) -> List[str]:
    return [
        field
        for field in COMPARE_FIELDS
        if str(previous.get(field) or "") != str(current.get(field) or "")
    ]


def update_type_for(previous: JsonDict | None, current: JsonDict) -> str:
    explicit_change = str(current.get("change_type") or "")
    if explicit_change in {"withdrawn", "reaffirmed"}:
        return explicit_change
    if previous is None:
        return "new_recommendation"
    fields = changed_fields(previous, current)
    if "recommendation_text" in fields:
        return "text_modified"
    if "strength" in fields:
        return "strength_changed"
    if "direction" in fields:
        return "direction_changed"
    if "certainty" in fields:
        return "certainty_changed"
    if fields:
        return "evidence_updated"
    return "unchanged"


def change_summary(previous: JsonDict | None, current: JsonDict) -> str:
    if previous is None:
        return "New recommendation version added."
    fields = changed_fields(previous, current)
    if not fields:
        return "No material recommendation change."
    return "Changed fields: " + ", ".join(fields)


def triggering_evidence_ids(current: JsonDict) -> List[str]:
    payload = current.get("normalized_payload")
    if not isinstance(payload, dict):
        return []
    evidence = payload.get("linked_evidence")
    if not isinstance(evidence, dict):
        return []
    ids = evidence.get("evidence_ids") or evidence.get("linked_evidence_ids")
    return [str(item) for item in ids] if isinstance(ids, list) else []


def build_update_logs(
    previous_versions: Iterable[JsonDict],
    current_versions: Iterable[JsonDict],
    *,
    updated_by: str = "version_diff",
) -> tuple[List[JsonDict], JsonDict]:
    """Build update logs that explain how current versions differ from previous versions."""

    previous_by_rec = latest_by_recommendation(previous_versions)
    current_rows = list(current_versions)
    logs: List[JsonDict] = []
    type_counts: Counter[str] = Counter()
    for current in current_rows:
        identity = recommendation_identity(current)
        previous = previous_by_rec.get(identity)
        update_type = update_type_for(previous, current)
        type_counts[update_type] += 1
        if update_type == "unchanged":
            continue
        current_version_id = str(current.get("recommendation_version_id") or "")
        old_version_id = str((previous or {}).get("recommendation_version_id") or "") or None
        log = {
            "update_log_id": stable_id("update_log", identity, current_version_id, update_type),
            "guideline_id": str(current.get("guideline_id") or ""),
            "new_recommendation_version_id": current_version_id,
            "recommendation_version_id": current_version_id,
            "old_recommendation_version_id": old_version_id,
            "update_type": update_type,
            "change_summary": change_summary(previous, current),
            "change_reason": "automated_version_comparison",
            "triggering_evidence_ids": triggering_evidence_ids(current),
            "updated_by": updated_by,
            "updated_at": utc_now(),
            "published_at": current.get("published_at"),
        }
        logs.append(log)
    return logs, {
        "previous_versions": len(previous_by_rec),
        "current_versions": len(latest_by_recommendation(current_rows)),
        "generated_update_logs": len(logs),
        "update_type_counts": dict(type_counts),
    }


def build_update_logs_file(
    previous_versions_input: str | Path,
    current_versions_input: str | Path,
    update_logs_output: str | Path,
    report_output: str | Path,
) -> JsonDict:
    logs, report = build_update_logs(iter_jsonl(previous_versions_input), list(iter_jsonl(current_versions_input)))
    write_jsonl(update_logs_output, logs)
    write_jsonl(report_output, [report])
    return report

