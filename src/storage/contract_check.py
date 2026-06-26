"""Storage 入库契约检查。

该模块不连接 PostgreSQL，只读取一个 run 目录下的 JSONL 文件，提前验证
流水线产物是否能满足 storage 层的基础入库契约。它补足 `pre_ingest_check`
之外的检查：字段别名、旧表必填列、复核 payload 保留和主键完整性。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from src.common.process_jsonl import iter_jsonl
from src.storage.generic_ingest import GENERIC_TABLE_CONFIG, aliased_value


JsonDict = Dict[str, Any]

RUN_TABLE_FILES = {
    "recommendation_candidates": "recommendation_candidates.jsonl",
    "grade_candidates": "grade_candidates.jsonl",
    "pico_questions": "pico_questions.jsonl",
    "evidence_items": "evidence_items.jsonl",
    "recommendation_versions": "recommendation_versions.jsonl",
}

# 这里列的是 storage 当前表结构中真正会影响基础入库的字段。
# 新版 JSONL 字段名和旧表列名不一致时，由 GENERIC_TABLE_CONFIG.aliases 兜底。
REQUIRED_STORAGE_FIELDS = {
    "recommendation_candidates": ("candidate_id", "model_trace_id", "statement"),
    "grade_candidates": ("grade_candidate_id", "model_trace_id", "source_grade_raw"),
    "pico_questions": ("pico_id", "clinical_question", "population", "intervention"),
    "evidence_items": ("evidence_id", "pico_id"),
    "recommendation_versions": ("recommendation_version_id", "recommendation_candidate_id", "guideline_id", "version_number", "recommendation_text"),
}

RECOMMENDED_PAYLOAD_FIELDS = {
    "recommendation_candidates": ("normalized_payload", "raw_payload"),
    "grade_candidates": ("normalized_payload", "raw_payload"),
    "pico_questions": ("normalized_payload",),
    "evidence_items": ("normalized_payload",),
    "recommendation_versions": ("normalized_payload", "raw_payload"),
}


def _rows(path: Path) -> List[JsonDict]:
    return list(iter_jsonl(path)) if path.exists() else []


def _duplicate_count(rows: Iterable[JsonDict], id_field: str) -> int:
    counts = Counter(str(row.get(id_field) or "") for row in rows if row.get(id_field))
    return sum(count - 1 for count in counts.values() if count > 1)


def _defaulted_value(row: JsonDict, table: str, field: str) -> Any:
    config = GENERIC_TABLE_CONFIG.get(table, {})
    aliases = config.get("aliases") or {}
    value = aliased_value(row, field, aliases)
    defaults = config.get("defaults") or {}
    if value is None and field in defaults:
        return defaults[field]
    return value


def _missing_count(rows: Sequence[JsonDict], table: str, field: str) -> int:
    return sum(1 for row in rows if _defaulted_value(row, table, field) in (None, ""))


def _payload_missing_count(rows: Sequence[JsonDict], field: str) -> int:
    return sum(1 for row in rows if not isinstance(row.get(field), dict))


def build_storage_contract_report(run_dir: str | Path) -> JsonDict:
    """返回 run 目录与 storage 入库契约的匹配报告。"""

    run_path = Path(run_dir)
    missing_files: List[str] = []
    table_reports: JsonDict = {}
    blocking_reasons: List[str] = []
    warning_reasons: List[str] = []

    for table, filename in RUN_TABLE_FILES.items():
        path = run_path / filename
        if not path.exists():
            missing_files.append(filename)
            blocking_reasons.append(f"missing_file:{filename}")
            table_reports[table] = {"path": str(path), "rows": 0, "missing_file": True}
            continue

        rows = _rows(path)
        config = GENERIC_TABLE_CONFIG[table]
        pk = str(config["pk"])
        missing_required = {
            field: _missing_count(rows, table, field)
            for field in REQUIRED_STORAGE_FIELDS[table]
        }
        missing_payload = {
            field: _payload_missing_count(rows, field)
            for field in RECOMMENDED_PAYLOAD_FIELDS.get(table, ())
        }
        duplicate_ids = _duplicate_count(rows, pk)
        missing_ids = sum(1 for row in rows if not row.get(pk))

        for field, count in missing_required.items():
            if count:
                blocking_reasons.append(f"{table}.{field}_missing:{count}")
        if missing_ids:
            blocking_reasons.append(f"{table}.{pk}_missing:{missing_ids}")
        if duplicate_ids:
            blocking_reasons.append(f"{table}.{pk}_duplicate:{duplicate_ids}")
        for field, count in missing_payload.items():
            if count:
                warning_reasons.append(f"{table}.{field}_not_dict:{count}")

        table_reports[table] = {
            "path": str(path),
            "rows": len(rows),
            "primary_key": pk,
            "missing_id_rows": missing_ids,
            "duplicate_id_rows": duplicate_ids,
            "missing_required_storage_fields": missing_required,
            "missing_recommended_payload_fields": missing_payload,
        }

    return {
        "run_dir": str(run_path),
        "passed": not blocking_reasons,
        "missing_files": missing_files,
        "blocking_reasons": blocking_reasons,
        "warning_reasons": warning_reasons,
        "table_reports": table_reports,
        "recommended_next_action": "ingest_or_run_live_smoke_test" if not blocking_reasons else "fix_storage_contract_before_ingest",
    }
