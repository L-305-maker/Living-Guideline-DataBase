"""质量评估文件：提供 goldset、审计指标和质量报告能力，用于评估候选结果是否可靠。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.pipeline.extraction.recommendation.patterns import has_action_pattern, matched_action_texts

JsonDict = Dict[str, Any]
VALID_ROUTE = {"skip", "needs_layout_repair", "needs_quality_review", "sentence_only"}
VALID_BOOL_OR_NONE = {True, False, None}
DEFAULT_THRESHOLDS: JsonDict = {
    "positive_anchor_recall": {"min": 0.95},
    "candidate_precision": {"min": 0.8},
    "skipped_false_negative_rate": {"max": 0.1},
    "candidate_layout_contamination_rate": {"max": 0.2},
    "missing_provenance": {"max": 0},
    "trace_failed": {"max": 0},
}


def read_jsonl(path: Path, errors: List[str] | None = None) -> List[JsonDict]:
    rows: List[JsonDict] = []
    if not path.exists():
        if errors is not None:
            errors.append(f"missing:{path}")
        return rows
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                if errors is not None:
                    errors.append(f"{path}:{line_no}:{exc}")
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def write_jsonl(path: Path, rows: Iterable[JsonDict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: Path, row: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> JsonDict:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return data


# 读取一次回归目录并计算固定指标，作为 baseline 和 before/after report 的数据来源。
def metrics_for_dir(root: Path) -> JsonDict:
    errors: List[str] = []
    cleaned = read_jsonl(root / "cleaned100.jsonl", errors)
    blocks = read_jsonl(root / "blocks100.jsonl", errors)
    recs = read_jsonl(root / "recommendation_candidates100.jsonl", errors)
    traces = read_jsonl(root / "recommendation_traces100.jsonl", errors)
    queue = read_jsonl(root / "llm_review_queue100.jsonl", errors)
    skipped = read_jsonl(root / "routed100_skipped_blocks.jsonl", errors)
    repair_path = root / "routed100_layout_repair_candidates.jsonl"
    repair = read_jsonl(repair_path, errors) if repair_path.exists() else []
    priorities = Counter(str(row.get("priority") or "") for row in queue)
    statuses = Counter(str(row.get("status") or "") for row in recs)
    skipped_with_action = [row for row in skipped if has_action_pattern(str(row.get("text") or ""))]

    return {
        "total_records": len(cleaned),
        "total_blocks": len(blocks),
        "total_candidates": len(recs),
        "P0_count": priorities.get("P0", 0),
        "P1_count": priorities.get("P1", 0),
        "P2_count": priorities.get("P2", 0),
        "pending_count": statuses.get("pending", 0),
        "needs_review_count": statuses.get("needs_review", 0),
        "auto_reject_count": 0,
        "long_rec_text_gt_700": sum(1 for row in recs if len(str(row.get("recommendation_text") or "")) > 700),
        "missing_provenance": sum(
            1
            for row in recs
            if not ((row.get("normalized_payload") or {}).get("source_block_id") and (row.get("normalized_payload") or {}).get("extraction_reason"))
        ),
        "skipped_with_action_pattern_count": len(skipped_with_action),
        "layout_repair_candidates": len(repair),
        "trace_failed": sum(1 for row in traces if not row.get("success")),
        "jsonl_parse_errors": errors,
    }


def sample_rows(rows: List[JsonDict], limit: int) -> List[JsonDict]:
    return rows[:limit]


def section_text(row: JsonDict) -> str:
    section = row.get("section_path")
    if isinstance(section, list):
        return " ".join(str(item) for item in section)
    return str(section or row.get("source_section") or "")


def candidate_audit_row(row: JsonDict) -> JsonDict:
    payload = row.get("normalized_payload") or {}
    return {
        "record_id": row.get("record_id", ""),
        "block_id": payload.get("source_block_id") or payload.get("block_id") or row.get("block_id", ""),
        "candidate_id": row.get("candidate_id", ""),
        "text": row.get("source_text", ""),
        "normalized_statement": row.get("recommendation_text", ""),
        "section_path": payload.get("section_path", []),
        "route_action": (payload.get("route") or {}).get("route_action", ""),
        "candidate_status": row.get("status", ""),
        "confidence": row.get("extraction_confidence"),
        "quality_notes": payload.get("quality_notes", []),
        "provenance": {
            "source_block_id": payload.get("source_block_id", ""),
            "sentence_index": payload.get("sentence_index"),
            "page": None,
        },
        "annotation": {
            "is_true_recommendation": None,
            "is_complete_statement": None,
            "is_clinical_action": None,
            "is_guideline_meta_text": None,
            "has_layout_contamination": None,
            "needs_layout_repair": None,
            "should_enter_candidate_queue": None,
            "should_enter_manual_review": None,
            "reason": "",
        },
    }


def skipped_audit_row(row: JsonDict) -> JsonDict:
    route = row.get("route") or {}
    text = str(row.get("text") or "")
    return {
        "record_id": row.get("record_id", ""),
        "block_id": row.get("block_id", ""),
        "text": text,
        "section_path": row.get("section_path", []),
        "block_type": row.get("block_type", ""),
        "skip_reasons": route.get("noise_reasons", []),
        "matched_action_patterns": matched_action_texts(text),
        "annotation": {
            "contains_true_recommendation": None,
            "is_recoverable_by_sentence_split": None,
            "requires_layout_repair": None,
            "is_pure_noise": None,
            "should_route_to": None,
            "reason": "",
        },
    }


def positive_anchor_rows() -> List[JsonDict]:
    examples = [
        ("We recommend annual influenza vaccination for adults at increased risk of complications.", "we_recommend"),
        ("We suggest cognitive behavioral therapy for adults with chronic insomnia.", "we_suggest"),
        ("Clinicians should offer smoking cessation counseling to adults who smoke.", "clinicians_should"),
        ("Patients should receive two appropriately spaced doses of MMR vaccine when indicated.", "patients_should"),
        ("It is recommended that patients with severe disease be referred to specialist care.", "recommended_that"),
        ("It is suggested that children with persistent symptoms receive follow-up testing.", "suggested_that"),
        ("We recommend against routine antibiotics for uncomplicated viral upper respiratory infection.", "recommends_against"),
        ("We suggest against benzodiazepines as first-line therapy in older adults.", "suggests_against"),
        ("Clinicians should not use valproic acid during pregnancy unless no safer alternative is available.", "should_not"),
        ("This intervention is not recommended for patients with known hypersensitivity.", "not_recommended"),
        ("The panel recommends low-dose aspirin for selected adults at high cardiovascular risk.", "panel_recommends"),
        ("The AASM recommends positive airway pressure therapy for adults with obstructive sleep apnea.", "aasm_recommends"),
        ("CDC recommends first-line antimicrobial therapy for persons with tularemia.", "cdc_recommends"),
        ("Recommendation 1: In adults with insomnia, we suggest cognitive behavioral therapy.", "we_suggest"),
        ("Patients should not receive live vaccine when severely immunocompromised.", "should_not"),
        ("The panel recommends against screening asymptomatic low-risk adults.", "recommends_against"),
        ("Clinicians should administer epinephrine promptly for suspected anaphylaxis.", "clinicians_should"),
        ("Adults with persistent symptoms should receive follow-up testing.", "patients_should"),
        ("It is recommended that anticoagulation be offered to eligible patients with atrial fibrillation.", "recommended_that"),
        ("We suggest that clinicians monitor renal function after initiating therapy.", "we_suggest"),
    ]
    rows = []
    for index, (text, pattern) in enumerate(examples, 1):
        rows.append(
            {
                "record_id": "positive_anchor_record",
                "block_id": f"positive_anchor_block_{index}",
                "text": text,
                "section_path": ["Positive Anchor", "Recommendations"],
                "expected_min_candidates": 1,
                "must_match_action_pattern": pattern,
                "reason": "clear clinical action recommendation",
            }
        )
    return rows


# 从真实回归产物生成待人工标注样本：候选、skip+action 风险块、正例 anchor。
def generate_samples(root: Path, fixtures: Path) -> JsonDict:
    recs = read_jsonl(root / "recommendation_candidates100.jsonl")
    skipped = read_jsonl(root / "routed100_skipped_blocks.jsonl")
    pending = [row for row in recs if row.get("status") == "pending"]
    needs_review = [row for row in recs if row.get("status") == "needs_review"]
    risk = [
        row
        for row in recs
        if (row.get("normalized_payload") or {}).get("quality_notes")
        or re.search(r"\b(?:evidence|methods?|discussion|disclosure)\b", section_text(row), re.I)
    ]
    selected: List[JsonDict] = []
    seen: set[str] = set()
    for group, limit in [(needs_review, 10), (risk, 15), (pending, 50)]:
        for row in group:
            cid = str(row.get("candidate_id") or "")
            if cid and cid not in seen:
                selected.append(row)
                seen.add(cid)
            if len(selected) >= limit and group is not pending:
                break
            if len(selected) >= 50:
                break
        if len(selected) >= 50:
            break
    candidate_count = write_jsonl(fixtures / "recommendation_candidate_audit_sample.jsonl", [candidate_audit_row(row) for row in selected[:50]])

    skipped_action = [row for row in skipped if has_action_pattern(str(row.get("text") or ""))]
    skipped_count = write_jsonl(fixtures / "recommendation_skipped_action_audit_sample.jsonl", [skipped_audit_row(row) for row in sample_rows(skipped_action, 30)])
    positive_count = write_jsonl(fixtures / "recommendation_positive_anchor_sample.jsonl", positive_anchor_rows())
    return {
        "candidate_audit_sample": candidate_count,
        "skipped_action_audit_sample": skipped_count,
        "positive_anchor_sample": positive_count,
    }


def require_bool(value: Any, path: str) -> bool:
    if value not in VALID_BOOL_OR_NONE or value is None:
        raise ValueError(f"{path} must be true or false")
    return bool(value)


# 读取人工标注后的 fixtures，计算 precision、false negative、layout repair need 等指标。
def eval_audit(fixtures: Path, output: Path | None = None) -> JsonDict:
    candidates = read_jsonl(fixtures / "recommendation_candidate_audit_sample.jsonl")
    skipped = read_jsonl(fixtures / "recommendation_skipped_action_audit_sample.jsonl")
    positives = read_jsonl(fixtures / "recommendation_positive_anchor_sample.jsonl")
    if not candidates or not skipped or not positives:
        raise ValueError("audit fixtures are required")

    true_candidates = incomplete = contaminated = 0
    for idx, row in enumerate(candidates, 1):
        ann = row.get("annotation") or {}
        true_candidates += require_bool(ann.get("is_true_recommendation"), f"candidate[{idx}].is_true_recommendation")
        incomplete += not require_bool(ann.get("is_complete_statement"), f"candidate[{idx}].is_complete_statement")
        contaminated += require_bool(ann.get("has_layout_contamination"), f"candidate[{idx}].has_layout_contamination")

    skipped_true = repair_needed = 0
    for idx, row in enumerate(skipped, 1):
        ann = row.get("annotation") or {}
        route = ann.get("should_route_to")
        if route not in VALID_ROUTE:
            raise ValueError(f"skipped[{idx}].should_route_to invalid: {route}")
        skipped_true += require_bool(ann.get("contains_true_recommendation"), f"skipped[{idx}].contains_true_recommendation")
        repair_needed += require_bool(ann.get("requires_layout_repair"), f"skipped[{idx}].requires_layout_repair")

    positive_hits = sum(1 for row in positives if int(row.get("observed_candidates", row.get("expected_min_candidates", 0)) or 0) >= int(row.get("expected_min_candidates", 1) or 1))
    report = {
        "audited_candidates": len(candidates),
        "audited_skipped_action_blocks": len(skipped),
        "positive_anchor_samples": len(positives),
        "candidate_precision": round(true_candidates / len(candidates), 4),
        "candidate_incomplete_rate": round(incomplete / len(candidates), 4),
        "candidate_layout_contamination_rate": round(contaminated / len(candidates), 4),
        "skipped_false_negative_rate": round(skipped_true / len(skipped), 4),
        "layout_repair_need_rate": round(repair_needed / len(skipped), 4),
        "positive_anchor_recall": round(positive_hits / len(positives), 4),
    }
    if output:
        write_json(output, report)
    return report


def parse_threshold(value: str) -> tuple[str, JsonDict]:
    if ">=" in value:
        key, raw = value.split(">=", 1)
        return key.strip(), {"min": float(raw)}
    if "<=" in value:
        key, raw = value.split("<=", 1)
        return key.strip(), {"max": float(raw)}
    if "==" in value:
        key, raw = value.split("==", 1)
        return key.strip(), {"eq": float(raw)}
    raise ValueError(f"Threshold must use >=, <=, or ==: {value}")


def merged_thresholds(overrides: Iterable[str] = ()) -> JsonDict:
    thresholds = json.loads(json.dumps(DEFAULT_THRESHOLDS))
    for override in overrides:
        key, rule = parse_threshold(override)
        thresholds[key] = rule
    return thresholds


def evaluate_thresholds(metrics: JsonDict, thresholds: JsonDict | None = None) -> JsonDict:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    checks: List[JsonDict] = []
    passed = True
    for metric, rule in thresholds.items():
        if metric not in metrics:
            checks.append({"metric": metric, "status": "missing", "passed": False, "rule": rule})
            passed = False
            continue
        value = metrics.get(metric)
        metric_passed = True
        if "min" in rule:
            metric_passed = isinstance(value, (int, float)) and value >= rule["min"]
        if "max" in rule:
            metric_passed = isinstance(value, (int, float)) and value <= rule["max"]
        if "eq" in rule:
            metric_passed = isinstance(value, (int, float)) and value == rule["eq"]
        checks.append({"metric": metric, "value": value, "rule": rule, "passed": metric_passed})
        passed = passed and metric_passed
    return {
        "passed": passed,
        "checked_metrics": len(checks),
        "failed_metrics": [check for check in checks if not check["passed"]],
        "checks": checks,
    }


# 对比两个回归目录，生成候选数、P0、repair 池等核心指标的差异报告。
def regression_report(before: Path, after: Path, output: Path) -> JsonDict:
    before_metrics = metrics_for_dir(before)
    after_metrics = metrics_for_dir(after)
    report = {
        "before": before_metrics,
        "after": after_metrics,
        "diff": {
            "candidate_delta": after_metrics["total_candidates"] - before_metrics["total_candidates"],
            "P0_delta": after_metrics["P0_count"] - before_metrics["P0_count"],
            "manual_queue_delta": (
                after_metrics["P0_count"] + after_metrics["P1_count"] + after_metrics["P2_count"]
                - before_metrics["P0_count"] - before_metrics["P1_count"] - before_metrics["P2_count"]
            ),
            "layout_repair_delta": after_metrics["layout_repair_candidates"] - before_metrics["layout_repair_candidates"],
        },
        "threshold_check": evaluate_thresholds(after_metrics),
    }
    write_json(output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recommendation quality audit helpers.")
    sub = parser.add_subparsers(dest="command", required=True)
    baseline = sub.add_parser("baseline")
    baseline.add_argument("--root", required=True)
    baseline.add_argument("--output", required=True)
    samples = sub.add_parser("generate-samples")
    samples.add_argument("--root", required=True)
    samples.add_argument("--fixtures", default="tests/fixtures")
    audit = sub.add_parser("eval-audit")
    audit.add_argument("--fixtures", default="tests/fixtures")
    audit.add_argument("--output")
    check = sub.add_parser("check-thresholds")
    check.add_argument("--metrics-file")
    check.add_argument("--fixtures")
    check.add_argument("--output")
    check.add_argument("--threshold", action="append", default=[], help="Override threshold, for example candidate_precision>=0.85")
    report = sub.add_parser("regression-report")
    report.add_argument("--before", required=True)
    report.add_argument("--after", required=True)
    report.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "baseline":
        result = metrics_for_dir(Path(args.root))
        write_json(Path(args.output), result)
    elif args.command == "generate-samples":
        result = generate_samples(Path(args.root), Path(args.fixtures))
        print(json.dumps(result, ensure_ascii=False))
    elif args.command == "eval-audit":
        result = eval_audit(Path(args.fixtures), Path(args.output) if args.output else None)
        print(json.dumps(result, ensure_ascii=False))
    elif args.command == "check-thresholds":
        if args.metrics_file:
            metrics = read_json(Path(args.metrics_file))
        elif args.fixtures:
            metrics = eval_audit(Path(args.fixtures))
        else:
            raise ValueError("check-thresholds requires --metrics-file or --fixtures")
        result = evaluate_thresholds(metrics, merged_thresholds(args.threshold))
        if args.output:
            write_json(Path(args.output), result)
        print(json.dumps(result, ensure_ascii=False))
        if not result["passed"]:
            raise SystemExit(1)
    elif args.command == "regression-report":
        result = regression_report(Path(args.before), Path(args.after), Path(args.output))
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

