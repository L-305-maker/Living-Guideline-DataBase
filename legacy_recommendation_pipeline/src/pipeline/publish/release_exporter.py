"""发布阶段文件：把审核通过、闭包完整的候选版本导出为 publish-ready 正式发布包。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from src.common.extraction_common import utc_now
from src.common.process_jsonl import iter_jsonl, write_jsonl
from src.domain.common import stable_id
from src.pipeline.extraction.versioning.recommendation_state_builder import build_recommendation_state, is_publishable_version
from src.pipeline.update.version_diff import build_update_logs


JsonDict = Dict[str, Any]
RELEASE_EXPORTER_VERSION = "release_exporter_v1"
VALID_EVIDENCE_STATUSES = {"included", "accepted", "publish_ready"}
REVIEW_QUEUE_FILES = (
    "association_review_summary.jsonl",
    "evidence_pico_review_queue.jsonl",
    "evidence_review_queue.jsonl",
    "first_pass_review_queue.jsonl",
    "grade_review_queue.jsonl",
    "llm_priority_queue.jsonl",
    "pico_review_queue.jsonl",
    "recommendation_association_review_queue.jsonl",
    "recommendation_review_queue.jsonl",
    "review_batch_summary.jsonl",
)
TRACE_FILES = (
    "recommendation_traces.jsonl",
    "grade_traces.jsonl",
    "pico_traces.jsonl",
    "evidence_traces.jsonl",
)


def _read_rows(path: Path) -> List[JsonDict]:
    return list(iter_jsonl(path)) if path.exists() else []


def _read_map(path: Path, id_field: str, wanted_ids: set[str] | None = None) -> Dict[str, JsonDict]:
    rows: Dict[str, JsonDict] = {}
    if not path.exists():
        return rows
    for row in iter_jsonl(path):
        row_id = str(row.get(id_field) or "")
        if not row_id:
            continue
        if wanted_ids is None or row_id in wanted_ids:
            rows[row_id] = row
    return rows


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _payload(row: JsonDict) -> JsonDict:
    payload = row.get("normalized_payload")
    return payload if isinstance(payload, dict) else {}


def _linked_evidence_ids(version: JsonDict) -> List[str]:
    ids: List[str] = []
    for payload in (_payload(version), version.get("raw_payload") if isinstance(version.get("raw_payload"), dict) else {}):
        linked = payload.get("linked_evidence")
        if isinstance(linked, dict):
            for key in ("linked_evidence_ids", "evidence_ids"):
                values = linked.get(key)
                if isinstance(values, list):
                    ids.extend(str(value) for value in values if value)
            items = linked.get("evidence_items")
            if isinstance(items, list):
                ids.extend(str(item.get("evidence_id") or "") for item in items if isinstance(item, dict))
        values = payload.get("linked_evidence_ids")
        if isinstance(values, list):
            ids.extend(str(value) for value in values if value)
    return list(dict.fromkeys(item for item in ids if item))


def _has_source_span(row: JsonDict) -> bool:
    if not row.get("source_span") or not row.get("source_span_ref"):
        return False
    start = row.get("start_char")
    end = row.get("end_char")
    if start is None or end is None:
        return False
    try:
        return int(end) >= int(start)
    except (TypeError, ValueError):
        return False


def _evidence_blockers(evidence: JsonDict | None, *, version_pico_id: str) -> List[str]:
    """判断一条 evidence 是否能进入正式发布闭包。"""

    # 发布包只接受与版本 PICO 一致、状态 publish-ready、且有 source span 的证据。
    # 不满足条件的 evidence 不删除，而是带 blocking reason 进入 backlog。
    if evidence is None:
        return ["evidence_row_missing"]
    blockers: List[str] = []
    evidence_pico_id = str(evidence.get("pico_id") or "")
    status = str(evidence.get("screening_status") or "")
    if not evidence_pico_id:
        blockers.append("evidence_missing_pico_id")
    elif version_pico_id and evidence_pico_id != version_pico_id:
        blockers.append("evidence_pico_mismatch")
    if status not in VALID_EVIDENCE_STATUSES:
        blockers.append(f"evidence_status_not_publish_ready:{status or 'missing'}")
    if not evidence.get("source_span"):
        blockers.append("evidence_missing_source_span")
    return blockers


def _evidence_summary(evidence_rows: Sequence[JsonDict]) -> JsonDict:
    return {
        "association_reason": "release_export_filtered_publish_ready_evidence",
        "linked_evidence_count": len(evidence_rows),
        "linked_evidence_ids": [row.get("evidence_id") for row in evidence_rows],
        "study_design_counts": dict(Counter(str(row.get("study_design") or "unknown") for row in evidence_rows)),
        "effect_direction_counts": dict(Counter(str(row.get("effect_direction") or "uncertain") for row in evidence_rows)),
        "screening_status_counts": dict(Counter(str(row.get("screening_status") or "missing") for row in evidence_rows)),
        "evidence_items": [
            {
                "evidence_id": row.get("evidence_id"),
                "pico_id": row.get("pico_id"),
                "recommendation_candidate_id": row.get("recommendation_candidate_id"),
                "study_design": row.get("study_design"),
                "effect_direction": row.get("effect_direction"),
                "screening_status": row.get("screening_status"),
                "extraction_confidence": row.get("extraction_confidence"),
                "source_block_id": row.get("source_block_id"),
                "source_order": row.get("source_order"),
            }
            for row in evidence_rows
        ],
    }


def _release_version(version: JsonDict, valid_evidence: Sequence[JsonDict], *, published_at: str) -> JsonDict:
    """把通过正式 gate 的 RecommendationVersion 标记成发布态。"""

    release = deepcopy(version)
    release["quality_status"] = "publishable"
    release["published_at"] = published_at
    normalized = deepcopy(_payload(release))
    previous_gate = normalized.get("publish_gate") if isinstance(normalized.get("publish_gate"), dict) else {}
    previous_warnings = previous_gate.get("warning_reasons") if isinstance(previous_gate.get("warning_reasons"), list) else []
    normalized["linked_evidence"] = _evidence_summary(valid_evidence)
    normalized["publish_gate"] = {
        "quality_status": "publishable",
        "publishable": True,
        "blocking_reasons": [],
        "warning_reasons": previous_warnings,
        "formal_gate_scope": "publish_ready_release_package",
    }
    normalized["release"] = {
        "release_exporter_version": RELEASE_EXPORTER_VERSION,
        "published_at": published_at,
        "evidence_policy": "included_or_publish_ready_only",
        "linked_evidence_ids": [row.get("evidence_id") for row in valid_evidence],
    }
    release["normalized_payload"] = normalized
    raw_payload = deepcopy(release.get("raw_payload")) if isinstance(release.get("raw_payload"), dict) else {}
    raw_payload["release_exporter_version"] = RELEASE_EXPORTER_VERSION
    raw_payload["release_linked_evidence_count"] = len(valid_evidence)
    release["raw_payload"] = raw_payload
    return release


def _release_row(row: JsonDict, *, entity: str, published_at: str, status_field: str | None = None, status_value: str | None = None) -> JsonDict:
    release = deepcopy(row)
    if status_field and status_value:
        original_status = release.get(status_field)
        release[status_field] = status_value
    else:
        original_status = None
    normalized = deepcopy(_payload(release))
    release_info = {
        "release_exporter_version": RELEASE_EXPORTER_VERSION,
        "entity": entity,
        "published_at": published_at,
    }
    if original_status is not None:
        release_info["original_status"] = original_status
    normalized["release"] = release_info
    release["normalized_payload"] = normalized
    return release


def _record_ids_from_rows(rows: Iterable[JsonDict]) -> set[str]:
    ids: set[str] = set()
    for row in rows:
        for field in ("record_id", "source_record_id"):
            value = str(row.get(field) or "")
            if value:
                ids.add(value)
    return ids


def _collect_record_status(source_root: Path) -> tuple[set[str], set[str]]:
    ready_path = source_root / "gate" / "all_cleaned.ready.jsonl"
    bad_paths = [
        source_root / "gate" / "all_cleaned.needs_layout_repair.jsonl",
        source_root / "gate" / "all_cleaned.parse_failed.jsonl",
    ]
    ready_ids = {str(row.get("record_id") or "") for row in _read_rows(ready_path) if row.get("record_id")}
    bad_ids: set[str] = set()
    for path in bad_paths:
        bad_ids.update(str(row.get("record_id") or "") for row in _read_rows(path) if row.get("record_id"))
    return ready_ids, bad_ids


def _copy_review_queues(run_dir: Path, backlog_dir: Path) -> JsonDict:
    source = run_dir / "review_batch"
    target = backlog_dir / "review_queues"
    copied: Dict[str, str] = {}
    if not source.exists():
        return {"source_dir": str(source), "copied_files": copied, "missing_source_dir": True}
    target.mkdir(parents=True, exist_ok=True)
    for filename in REVIEW_QUEUE_FILES:
        source_file = source / filename
        if source_file.exists():
            target_file = target / filename
            shutil.copy2(source_file, target_file)
            copied[filename] = str(target_file)
    return {"source_dir": str(source), "copied_files": copied, "missing_source_dir": False}


def _minimal_trace(trace_id: str, *, target_entity_id: str = "") -> JsonDict:
    return {
        "model_trace_id": trace_id,
        "task_type": "release_trace_backfill",
        "method": "metadata_backfill",
        "model_name": "missing_trace_placeholder",
        "input_entity_type": "unknown",
        "input_entity_id": target_entity_id or trace_id,
        "input_text": "",
        "model_version": RELEASE_EXPORTER_VERSION,
        "prompt_version": None,
        "raw_output": {},
        "parsed_output": {},
        "confidence": 0.0,
        "parameters": {"reason": "source_trace_not_found_during_release_export"},
        "token_usage": {},
        "runtime_ms": 0,
        "code_version": None,
        "success": False,
        "error_message": "Source trace row was not found during release export.",
        "human_verified": False,
        "verified_by": None,
        "verified_at": None,
        "target_table": "release_package",
        "target_entity_id": target_entity_id,
        "created_at": utc_now(),
    }


def _collect_traces(source_root: Path, needed_trace_ids: set[str], trace_targets: Dict[str, str]) -> tuple[List[JsonDict], List[str]]:
    found: Dict[str, JsonDict] = {}
    downstream = source_root / "downstream"
    for filename in TRACE_FILES:
        path = downstream / filename
        if not path.exists():
            continue
        for row in iter_jsonl(path):
            trace_id = str(row.get("model_trace_id") or "")
            if trace_id in needed_trace_ids:
                found[trace_id] = row
    missing = sorted(needed_trace_ids - set(found))
    for trace_id in missing:
        found[trace_id] = _minimal_trace(trace_id, target_entity_id=trace_targets.get(trace_id, ""))
    return [found[trace_id] for trace_id in sorted(found)], missing


def _selected_cleaned_records(source_root: Path, wanted_record_ids: set[str]) -> tuple[List[JsonDict], set[str]]:
    rows: List[JsonDict] = []
    found: set[str] = set()
    path = source_root / "all_cleaned.jsonl"
    if not path.exists():
        return rows, wanted_record_ids
    for row in iter_jsonl(path):
        record_id = str(row.get("record_id") or "")
        if record_id in wanted_record_ids:
            rows.append(row)
            found.add(record_id)
    return rows, wanted_record_ids - found


def _seed_rows(cleaned_records: Sequence[JsonDict], seed_field: str, id_field: str) -> List[JsonDict]:
    by_id: Dict[str, JsonDict] = {}
    for row in cleaned_records:
        seed = row.get(seed_field)
        if not isinstance(seed, dict):
            continue
        seed_id = str(seed.get(id_field) or "")
        if seed_id:
            by_id[seed_id] = seed
    return [by_id[key] for key in sorted(by_id)]


def _inventory_counts(run_dir: Path) -> JsonDict:
    evidence_counts: Counter[str] = Counter()
    evidence_missing_pico = 0
    for row in iter_jsonl(run_dir / "evidence_items.jsonl"):
        evidence_counts[str(row.get("screening_status") or "missing")] += 1
        if not row.get("pico_id"):
            evidence_missing_pico += 1
    pico_counts = Counter(str(row.get("status") or "missing") for row in iter_jsonl(run_dir / "pico_questions.jsonl"))
    return {
        "evidence_screening_status_counts": dict(evidence_counts),
        "evidence_missing_pico_id": evidence_missing_pico,
        "pico_status_counts": dict(pico_counts),
    }


def export_release_package(
    run_dir: str | Path,
    *,
    source_root: str | Path | None = None,
    publish_dir: str | Path | None = None,
    backlog_dir: str | Path | None = None,
    published_at: str | None = None,
) -> JsonDict:
    """导出 publish_ready_v1 和 review_backlog_v1。

    这个函数是候选世界到正式发布世界的边界：只有闭包完整、可追溯、证据状态合格的版本
    会进入 publish_ready；其余版本和 evidence 会带原因进入 backlog。
    """

    run_path = Path(run_dir)
    source_path = Path(source_root) if source_root else run_path.parent
    publish_path = Path(publish_dir) if publish_dir else run_path / "publish_ready_v1"
    backlog_path = Path(backlog_dir) if backlog_dir else run_path / "review_backlog_v1"
    published_at = published_at or utc_now()
    publish_path.mkdir(parents=True, exist_ok=True)
    backlog_path.mkdir(parents=True, exist_ok=True)

    versions = _read_rows(run_path / "recommendation_versions.jsonl")
    referenced_evidence_ids = set()
    for version in versions:
        referenced_evidence_ids.update(_linked_evidence_ids(version))
    evidence_by_id = _read_map(run_path / "evidence_items.jsonl", "evidence_id", referenced_evidence_ids)
    ready_record_ids, bad_record_ids = _collect_record_status(source_path)
    inventory = _inventory_counts(run_path)

    release_versions: List[JsonDict] = []
    excluded_versions: List[JsonDict] = []
    excluded_evidence_rows: List[JsonDict] = []
    evidence_to_versions: Dict[str, List[str]] = defaultdict(list)
    release_evidence_ids: set[str] = set()
    version_blocker_counts: Counter[str] = Counter()
    filtered_evidence_reason_counts: Counter[str] = Counter()

    for version in versions:
        version_id = str(version.get("recommendation_version_id") or "")
        version_pico_id = str(version.get("pico_id") or "")
        blockers: List[str] = []

        # 先检查版本自身是否满足正式发布基本条件：publish gate、grade、PICO、source span、来源清洗状态。
        if not is_publishable_version(version):
            blockers.append("version_publish_gate_not_satisfied")
        if not version.get("recommendation_candidate_id"):
            blockers.append("version_missing_recommendation_candidate_id")
        if not version.get("grade_candidate_id"):
            blockers.append("version_missing_grade_candidate_id")
        if not version_pico_id:
            blockers.append("version_missing_pico_id")
        if not _has_source_span(version):
            blockers.append("version_missing_source_span_boundary")
        record_id = str(version.get("record_id") or "")
        if record_id in bad_record_ids:
            blockers.append("source_record_failed_cleaning_gate")
        if ready_record_ids and record_id and record_id not in ready_record_ids:
            blockers.append("source_record_not_in_ready_gate")

        valid_evidence: List[JsonDict] = []
        for evidence_id in _linked_evidence_ids(version):
            evidence = evidence_by_id.get(evidence_id)
            evidence_blockers = _evidence_blockers(evidence, version_pico_id=version_pico_id)
            if evidence_blockers:
                # evidence 不合格时只排除这条 evidence，并把原因写入 backlog，方便后续复核修复。
                for reason in evidence_blockers:
                    filtered_evidence_reason_counts[reason] += 1
                backlog_row = deepcopy(evidence) if evidence else {"evidence_id": evidence_id}
                backlog_row["release_backlog"] = {
                    "recommendation_version_id": version_id,
                    "blocking_reasons": evidence_blockers,
                    "release_exporter_version": RELEASE_EXPORTER_VERSION,
                }
                excluded_evidence_rows.append(backlog_row)
                continue
            valid_evidence.append(evidence)
        if not valid_evidence:
            blockers.append("version_without_publish_ready_evidence")

        if blockers:
            # 版本层面不闭合时，整条 RecommendationVersion 进入 backlog，不进入正式发布包。
            for blocker in blockers:
                version_blocker_counts[blocker] += 1
            blocked = deepcopy(version)
            blocked["release_backlog"] = {
                "blocking_reasons": blockers,
                "release_exporter_version": RELEASE_EXPORTER_VERSION,
            }
            excluded_versions.append(blocked)
            continue

        released = _release_version(version, valid_evidence, published_at=published_at)
        release_versions.append(released)
        for evidence in valid_evidence:
            evidence_id = str(evidence.get("evidence_id") or "")
            if evidence_id:
                evidence_to_versions[evidence_id].append(version_id)
                release_evidence_ids.add(evidence_id)

    release_candidate_ids = {str(row.get("recommendation_candidate_id") or "") for row in release_versions if row.get("recommendation_candidate_id")}
    release_grade_ids = {str(row.get("grade_candidate_id") or "") for row in release_versions if row.get("grade_candidate_id")}
    release_pico_ids = {str(row.get("pico_id") or "") for row in release_versions if row.get("pico_id")}
    release_evidence = [evidence_by_id[evidence_id] for evidence_id in sorted(release_evidence_ids) if evidence_id in evidence_by_id]
    release_pico_ids.update(str(row.get("pico_id") or "") for row in release_evidence if row.get("pico_id"))

    candidates_by_id = _read_map(run_path / "recommendation_candidates.jsonl", "candidate_id", release_candidate_ids)
    grades_by_id = _read_map(run_path / "grade_candidates.jsonl", "grade_candidate_id", release_grade_ids)
    picos_by_id = _read_map(run_path / "pico_questions.jsonl", "pico_id", release_pico_ids)

    release_candidates = [
        _release_row(candidates_by_id[candidate_id], entity="recommendation_candidate", published_at=published_at, status_field="status", status_value="accepted")
        for candidate_id in sorted(release_candidate_ids)
        if candidate_id in candidates_by_id
    ]
    release_grades = [
        _release_row(grades_by_id[grade_id], entity="grade_candidate", published_at=published_at, status_field="status", status_value="accepted")
        for grade_id in sorted(release_grade_ids)
        if grade_id in grades_by_id
    ]
    release_picos = [
        _release_row(picos_by_id[pico_id], entity="pico_question", published_at=published_at, status_field="status", status_value="active")
        for pico_id in sorted(release_pico_ids)
        if pico_id in picos_by_id
    ]
    release_evidence_rows: List[JsonDict] = []
    release_evidence_links: List[JsonDict] = []
    for evidence in release_evidence:
        released = _release_row(evidence, entity="evidence_item", published_at=published_at)
        linked_versions = sorted(set(evidence_to_versions.get(str(evidence.get("evidence_id") or ""), [])))
        normalized = deepcopy(_payload(released))
        normalized.setdefault("release", {})
        normalized["release"]["linked_recommendation_version_ids"] = linked_versions
        released["normalized_payload"] = normalized
        if len(linked_versions) == 1:
            released["recommendation_version_id"] = linked_versions[0]
        release_evidence_rows.append(released)
        for version_id in linked_versions:
            release_evidence_links.append(
                {
                    "link_id": stable_id("recommendation_version_evidence_link", version_id, evidence.get("evidence_id")),
                    "recommendation_version_id": version_id,
                    "evidence_id": evidence.get("evidence_id"),
                    "recommendation_candidate_id": evidence.get("recommendation_candidate_id"),
                    "pico_id": evidence.get("pico_id"),
                    "link_reason": "release_linked_evidence",
                    "link_status": "publish_ready",
                    "normalized_payload": {
                        "release_exporter_version": RELEASE_EXPORTER_VERSION,
                        "published_at": published_at,
                        "source": "RecommendationVersion.normalized_payload.linked_evidence",
                    },
                }
            )

    release_records_needed = _record_ids_from_rows(
        [*release_versions, *release_candidates, *release_grades, *release_picos, *release_evidence_rows]
    )
    # 正式发布包需要把依赖的 cleaned_records、guideline/paper seeds、model_traces 一并带上，
    # 这样 storage 入库时可以保持外键闭包完整。
    cleaned_records, missing_record_ids = _selected_cleaned_records(source_path, release_records_needed)
    guideline_seeds = _seed_rows(cleaned_records, "guideline_seed", "guideline_id")
    paper_seeds = _seed_rows(cleaned_records, "paper_seed", "paper_id")

    needed_trace_ids: set[str] = set()
    trace_targets: Dict[str, str] = {}
    for row in [*release_candidates, *release_grades, *release_evidence_rows]:
        trace_id = str(row.get("model_trace_id") or "")
        if trace_id:
            needed_trace_ids.add(trace_id)
            trace_targets[trace_id] = str(
                row.get("candidate_id") or row.get("grade_candidate_id") or row.get("evidence_id") or row.get("pico_id") or ""
            )
    release_traces, missing_trace_ids = _collect_traces(source_path, needed_trace_ids, trace_targets)

    release_recommendations = [build_recommendation_state(version, published_at=published_at) for version in release_versions]
    release_update_logs, update_log_report = build_update_logs([], release_versions, updated_by=RELEASE_EXPORTER_VERSION)
    for log in release_update_logs:
        log["published_at"] = published_at

    formal_blockers: List[str] = []
    if not release_versions:
        formal_blockers.append("no_release_recommendation_versions")
    missing_candidate_ids = sorted(release_candidate_ids - set(candidates_by_id))
    missing_grade_ids = sorted(release_grade_ids - set(grades_by_id))
    missing_pico_ids = sorted(release_pico_ids - set(picos_by_id))
    if missing_candidate_ids:
        formal_blockers.append("release_missing_recommendation_candidates")
    if missing_grade_ids:
        formal_blockers.append("release_missing_grade_candidates")
    if missing_pico_ids:
        formal_blockers.append("release_missing_pico_questions")
    if missing_record_ids:
        formal_blockers.append("release_missing_cleaned_records")

    formal_gate_report = {
        "release_exporter_version": RELEASE_EXPORTER_VERSION,
        "run_dir": str(run_path),
        "publish_dir": str(publish_path),
        "backlog_dir": str(backlog_path),
        "passed": not formal_blockers,
        "blocking_reasons": formal_blockers,
        "warning_reasons": {
            "missing_model_traces_backfilled": missing_trace_ids,
            "filtered_evidence_reason_counts": dict(filtered_evidence_reason_counts),
        },
        "release_counts": {
            "recommendation_versions": len(release_versions),
            "recommendations": len(release_recommendations),
            "recommendation_candidates": len(release_candidates),
            "grade_candidates": len(release_grades),
            "pico_questions": len(release_picos),
            "evidence_items": len(release_evidence_rows),
            "recommendation_version_evidence_links": len(release_evidence_links),
            "cleaned_records": len(cleaned_records),
            "model_traces": len(release_traces),
            "update_logs": len(release_update_logs),
        },
        "excluded_counts": {
            "recommendation_versions": len(excluded_versions),
            "evidence_items": len(excluded_evidence_rows),
        },
        "version_blocker_counts": dict(version_blocker_counts),
        "missing_ids": {
            "recommendation_candidate_ids": missing_candidate_ids,
            "grade_candidate_ids": missing_grade_ids,
            "pico_ids": missing_pico_ids,
            "record_ids": sorted(missing_record_ids),
        },
        "update_log_report": update_log_report,
    }

    manifest = {
        "release_exporter_version": RELEASE_EXPORTER_VERSION,
        "created_at": utc_now(),
        "published_at": published_at,
        "source_run_dir": str(run_path),
        "source_root": str(source_path),
        "publish_dir": str(publish_path),
        "backlog_dir": str(backlog_path),
        "formal_gate_passed": formal_gate_report["passed"],
        "formal_gate_blocking_reasons": formal_gate_report["blocking_reasons"],
        "release_counts": formal_gate_report["release_counts"],
        "excluded_counts": formal_gate_report["excluded_counts"],
        "inventory_counts": inventory,
        "rag_default_scope": "publish_ready_v1 only",
        "backlog_scope": "review_backlog_v1 excluded from default retrieval and formal ingest",
    }

    backlog_copy_report = _copy_review_queues(run_path, backlog_path)
    backlog_summary = {
        "release_exporter_version": RELEASE_EXPORTER_VERSION,
        "created_at": manifest["created_at"],
        "source_run_dir": str(run_path),
        "excluded_recommendation_versions": len(excluded_versions),
        "excluded_evidence_items_from_release_links": len(excluded_evidence_rows),
        "version_blocker_counts": dict(version_blocker_counts),
        "filtered_evidence_reason_counts": dict(filtered_evidence_reason_counts),
        "inventory_counts": inventory,
        "review_queue_copy": backlog_copy_report,
    }

    outputs = {
        "cleaned_records": publish_path / "cleaned_records.jsonl",
        "guideline_seeds": publish_path / "guideline_seeds.jsonl",
        "paper_seeds": publish_path / "paper_seeds.jsonl",
        "model_traces": publish_path / "model_traces.jsonl",
        "recommendation_candidates": publish_path / "recommendation_candidates.jsonl",
        "grade_candidates": publish_path / "grade_candidates.jsonl",
        "pico_questions": publish_path / "pico_questions.jsonl",
        "evidence_items": publish_path / "evidence_items.jsonl",
        "recommendation_version_evidence_links": publish_path / "recommendation_version_evidence_links.jsonl",
        "recommendation_versions": publish_path / "recommendation_versions.jsonl",
        "recommendations": publish_path / "recommendations.jsonl",
        "update_logs": publish_path / "update_logs.jsonl",
    }
    write_jsonl(outputs["cleaned_records"], cleaned_records)
    write_jsonl(outputs["guideline_seeds"], guideline_seeds)
    write_jsonl(outputs["paper_seeds"], paper_seeds)
    write_jsonl(outputs["model_traces"], release_traces)
    write_jsonl(outputs["recommendation_candidates"], release_candidates)
    write_jsonl(outputs["grade_candidates"], release_grades)
    write_jsonl(outputs["pico_questions"], release_picos)
    write_jsonl(outputs["evidence_items"], release_evidence_rows)
    write_jsonl(outputs["recommendation_version_evidence_links"], release_evidence_links)
    write_jsonl(outputs["recommendation_versions"], release_versions)
    write_jsonl(outputs["recommendations"], release_recommendations)
    write_jsonl(outputs["update_logs"], release_update_logs)

    write_jsonl(backlog_path / "excluded_recommendation_versions.jsonl", excluded_versions)
    write_jsonl(backlog_path / "excluded_evidence_items.jsonl", excluded_evidence_rows)
    _write_json(publish_path / "formal_gate_report.json", formal_gate_report)
    _write_json(publish_path / "release_manifest.json", manifest)
    _write_json(backlog_path / "backlog_summary.json", backlog_summary)
    (publish_path / "release_notes.md").write_text(
        "\n".join(
            [
                "# Publish Ready Release",
                "",
                f"- RecommendationVersion rows: {len(release_versions)}",
                f"- EvidenceItem rows: {len(release_evidence_rows)}",
                f"- RecommendationVersion-Evidence link rows: {len(release_evidence_links)}",
                f"- PICOQuestion rows: {len(release_picos)}",
                f"- Formal gate passed: {formal_gate_report['passed']}",
                "- Default RAG and formal ingest should point only at this directory.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "manifest": manifest,
        "formal_gate_report": formal_gate_report,
        "backlog_summary": backlog_summary,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a publish-ready release package and a separated review backlog.")
    parser.add_argument("--run-dir", required=True, help="Reviewed run directory containing recommendation/evidence JSONL files.")
    parser.add_argument("--source-root", help="Parent run root containing all_cleaned.jsonl, gate, and downstream traces.")
    parser.add_argument("--publish-dir", help="Output directory for publish_ready_v1.")
    parser.add_argument("--backlog-dir", help="Output directory for review_backlog_v1.")
    parser.add_argument("--published-at", help="Fixed published_at timestamp for reproducible exports.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_release_package(
        args.run_dir,
        source_root=args.source_root,
        publish_dir=args.publish_dir,
        backlog_dir=args.backlog_dir,
        published_at=args.published_at,
    )
    report = result["formal_gate_report"]
    print(
        "release_versions={versions} evidence_items={evidence} picos={picos} formal_gate_passed={passed} publish_dir={publish_dir}".format(
            versions=report["release_counts"]["recommendation_versions"],
            evidence=report["release_counts"]["evidence_items"],
            picos=report["release_counts"]["pico_questions"],
            passed=report["passed"],
            publish_dir=result["manifest"]["publish_dir"],
        )
    )


if __name__ == "__main__":
    main()

