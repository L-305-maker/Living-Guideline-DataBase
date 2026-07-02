"""入库前 JSONL 体检与 staging/formal 分流建议。

这个模块只读取流水线产物，不连接数据库、不写正式表。它的职责是回答三件事：
1. 当前 run 是否足够安全，可以进入 staging 表或人工评测流程；
2. 哪些数据绝不能直接进入正式活指南发布表；
3. 哪些字段/关联问题需要在 reviewer 或后续 pipeline 中优先处理。
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.common.process_jsonl import iter_jsonl, write_jsonl


JsonDict = Dict[str, Any]


def read_optional(path: Path) -> List[JsonDict]:
    """读取可选 JSONL 文件；不存在时返回空列表并由报告记录缺失。"""

    if not path.exists():
        return []
    return list(iter_jsonl(path))


def duplicate_count(rows: Iterable[JsonDict], id_field: str) -> int:
    """统计主键重复行数，用于提前发现 upsert 前的数据身份问题。"""

    counts = Counter(str(row.get(id_field) or "") for row in rows if row.get(id_field))
    return sum(count - 1 for count in counts.values() if count > 1)


def payload(row: JsonDict) -> JsonDict:
    value = row.get("normalized_payload")
    return value if isinstance(value, dict) else {}


def publishable_version(row: JsonDict) -> bool:
    """判断 RecommendationVersion 是否满足正式发布门禁。"""

    gate = payload(row).get("publish_gate")
    if isinstance(gate, dict) and gate.get("publishable") is False:
        return False
    return str(row.get("quality_status") or (gate or {}).get("quality_status") or "") == "publishable"


def status_counts(rows: Iterable[JsonDict], field: str) -> Dict[str, int]:
    return dict(Counter(str(row.get(field) or "missing") for row in rows))


def grade_association_quality_counts(rows: Iterable[JsonDict]) -> Dict[str, int]:
    return dict(Counter(str(payload(row).get("association_quality") or "unknown") for row in rows))


def entity_input_report(rows: List[JsonDict], id_field: str) -> JsonDict:
    return {
        "rows": len(rows),
        "missing_id_rows": sum(1 for row in rows if not row.get(id_field)),
        "duplicate_id_rows": duplicate_count(rows, id_field),
    }


def build_pre_ingest_report(
    run_dir: str | Path,
    grade_candidates_input: str | Path | None = None,
) -> JsonDict:
    """构建一次 run 的入库前报告。"""

    run_path = Path(run_dir)
    paths = {
        "recommendations": run_path / "recommendation_candidates.jsonl",
        "grades": Path(grade_candidates_input) if grade_candidates_input else run_path / "grade_candidates.jsonl",
        "picos": run_path / "pico_questions.jsonl",
        "evidence": run_path / "evidence_items.jsonl",
        "versions": run_path / "recommendation_versions.jsonl",
    }
    rows = {name: read_optional(path) for name, path in paths.items()}
    missing_files = [name for name, path in paths.items() if not path.exists()]

    recommendations = rows["recommendations"]
    grades = rows["grades"]
    picos = rows["picos"]
    evidence = rows["evidence"]
    versions = rows["versions"]
    publishable_versions = [row for row in versions if publishable_version(row)]

    formal_blockers: List[str] = []
    if not publishable_versions:
        formal_blockers.append("no_publishable_recommendation_versions")
    if missing_files:
        formal_blockers.append("missing_input_files")

    evidence_with_pico = sum(1 for row in evidence if row.get("pico_id"))
    evidence_without_pico = len(evidence) - evidence_with_pico
    if evidence_without_pico:
        formal_blockers.append("evidence_items_without_pico_must_stay_in_review")

    staging_blockers: List[str] = []
    for name, id_field in [
        ("recommendations", "candidate_id"),
        ("grades", "grade_candidate_id"),
        ("picos", "pico_id"),
        ("evidence", "evidence_id"),
    ]:
        entity_report = entity_input_report(rows[name], id_field)
        if entity_report["missing_id_rows"] or entity_report["duplicate_id_rows"]:
            staging_blockers.append(f"{name}_identity_errors")

    return {
        "run_dir": str(run_path),
        "input_files": {name: str(path) for name, path in paths.items()},
        "missing_files": missing_files,
        "staging_gate": {
            "passed": not staging_blockers,
            "blocking_reasons": staging_blockers,
            "recommendation_candidates_for_staging": len(recommendations),
            "grade_candidates_for_staging": len(grades),
            "pico_questions_for_staging": len(picos),
            "evidence_items_with_pico_for_staging": evidence_with_pico,
            "evidence_items_to_review_before_storage": evidence_without_pico,
        },
        "formal_publish_gate": {
            "passed": not formal_blockers,
            "blocking_reasons": formal_blockers,
            "publishable_recommendation_versions": len(publishable_versions),
            "review_queue_versions": len(versions) - len(publishable_versions),
        },
        "entity_reports": {
            "recommendations": {
                **entity_input_report(recommendations, "candidate_id"),
                "status_counts": status_counts(recommendations, "status"),
                "accepted_rows": sum(1 for row in recommendations if row.get("status") == "accepted"),
            },
            "grades": {
                **entity_input_report(grades, "grade_candidate_id"),
                "status_counts": status_counts(grades, "status"),
                "association_quality_counts": grade_association_quality_counts(grades),
                "missing_recommendation_candidate_id": sum(1 for row in grades if not row.get("recommendation_candidate_id")),
            },
            "picos": {
                **entity_input_report(picos, "pico_id"),
                "status_counts": status_counts(picos, "status"),
            },
            "evidence": {
                **entity_input_report(evidence, "evidence_id"),
                "screening_status_counts": status_counts(evidence, "screening_status"),
                "missing_pico_id": evidence_without_pico,
            },
            "versions": {
                **entity_input_report(versions, "recommendation_version_id"),
                "quality_status_counts": status_counts(versions, "quality_status"),
            },
        },
        "recommended_next_action": "load_staging_and_review_batch" if not staging_blockers else "fix_identity_errors_before_storage",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 run 产物是否适合 staging 入库或正式发布。")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--grade-candidates-input", default=None, help="可选：使用新版 GRADE 候选覆盖 run_dir 默认文件。")
    parser.add_argument("--output", required=True, help="输出单行 JSONL 报告。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_pre_ingest_report(args.run_dir, grade_candidates_input=args.grade_candidates_input)
    write_jsonl(args.output, [report])
    print(
        "staging_passed={staging} formal_passed={formal}".format(
            staging=report["staging_gate"]["passed"],
            formal=report["formal_publish_gate"]["passed"],
        )
    )


if __name__ == "__main__":
    main()
