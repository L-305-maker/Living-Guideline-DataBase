from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.process_jsonl import iter_jsonl, write_jsonl_obj


JsonDict = dict[str, Any]

ACTION_SIGNAL_RE = re.compile(r"\b(recommend|suggest|should|guideline statement|practice parameter)\b", re.I)
BODY_HEADING_IN_REFERENCES_RE = re.compile(r"(?im)^\s*(chapter\s+\d+|appendix|annex)\b")
RESIDUAL_BOILERPLATE_RE = re.compile(
    r"(?i)(Allrightsreserved|SubjecttoNoticeofrights|Downloaded from|All rights reserved)"
)
NICE_PAGE_BOILERPLATE_RE = re.compile(
    r"(?i)(Seetheoriginalguidanceat|See\s*(?:https?://)?www\.nice\.org\.uk/guidance|"
    r"Seewww\.nice\.org\.uk/guidance)"
)
READABLE_PUNCT = set(".,;:()[]{}+-/%'\"&<>_=*#")


def garbled_text_score(text: str) -> float:
    sample = str(text or "")[:20000]
    if not sample.strip():
        return 1.0

    total = len(sample)
    weird = 0
    alpha_num = 0
    long_symbol_runs = 0
    current_symbol_run = 0
    for char in sample:
        if char.isalnum():
            alpha_num += 1
            current_symbol_run = 0
        elif char.isspace() or char in READABLE_PUNCT:
            current_symbol_run = 0
        else:
            weird += 1
            current_symbol_run += 1
            if current_symbol_run == 12:
                long_symbol_runs += 1

    weird_ratio = weird / max(total, 1)
    alpha_num_ratio = alpha_num / max(total, 1)
    run_score = min(1.0, long_symbol_runs / 8)
    low_text_score = max(0.0, 0.18 - alpha_num_ratio) / 0.18
    return round(max(weird_ratio, low_text_score, (run_score * 0.65) + (low_text_score * 0.35)), 4)


def classify_record(record: JsonDict) -> tuple[str, list[str], JsonDict]:
    raw = str(record.get("raw_content") or "")
    clean = str(record.get("clean_content") or record.get("content") or "")
    references = str(record.get("references_text") or "")
    audit = dict(dict(record.get("cleaning_log") or {}).get("audit") or {})
    audit_flags = set(audit.get("audit_flags") or [])
    source = str(dict(record.get("metadata") or {}).get("source") or record.get("source") or "").lower()

    raw_chars = len(raw)
    clean_chars = len(clean)
    references_chars = len(references)
    clean_ratio = clean_chars / max(raw_chars, 1) if raw_chars else 0.0
    garbled_score = garbled_text_score(clean)
    residual_boilerplate = bool(RESIDUAL_BOILERPLATE_RE.search(clean)) or (
        source == "nice" and bool(NICE_PAGE_BOILERPLATE_RE.search(clean))
    )
    zero_recommendations_with_signal = not record.get("recommendations") and bool(ACTION_SIGNAL_RE.search(clean))

    reasons: list[str] = []
    warnings: list[str] = []

    if raw_chars == 0:
        reasons.append("empty_raw_content")
    if clean_chars == 0:
        reasons.append("empty_clean_content")
    if raw_chars > 0 and clean_ratio < 0.2:
        reasons.append("very_low_clean_ratio")
    if clean_chars >= 1000 and garbled_score >= 0.35:
        reasons.append("garbled_text")

    route = "needs_source_reextract" if reasons else "ready"
    if route == "ready":
        if "removed_ratio_high" in audit_flags or "references_larger_than_clean_text" in audit_flags:
            reasons.append("cleaning_deletion_budget_warning")
        if residual_boilerplate:
            reasons.append("residual_boilerplate")
        if "references_contains_body_heading" in audit_flags:
            if "removed_ratio_high" in audit_flags or references_chars > clean_chars * 2 or BODY_HEADING_IN_REFERENCES_RE.search(references):
                reasons.append("references_may_contain_body")
            else:
                warnings.append("references_contains_body_heading")
        if zero_recommendations_with_signal:
            warnings.append("zero_recommendations_with_action_signal")
        if reasons:
            route = "needs_review"

    metrics = {
        "raw_chars": raw_chars,
        "clean_chars": clean_chars,
        "references_chars": references_chars,
        "clean_ratio": round(clean_ratio, 4),
        "garbled_text_score": garbled_score,
        "audit_flags": sorted(audit_flags),
        "warnings": warnings,
        "zero_recommendations_with_signal": zero_recommendations_with_signal,
        "residual_boilerplate": residual_boilerplate,
    }
    return route, reasons + warnings, metrics


def routed_record(record: JsonDict, route: str, reasons: list[str], metrics: JsonDict) -> JsonDict:
    output = dict(record)
    output["cleaning_quality_route"] = route
    output["cleaning_quality_reasons"] = reasons
    output["cleaning_quality_metrics"] = metrics
    return output


def audit_file(input_path: str | Path, output_dir: str | Path, sample_limit: int = 200) -> JsonDict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    route_paths = {
        "ready": output / "cleaned.ready.jsonl",
        "needs_review": output / "cleaned.needs_review.jsonl",
        "needs_source_reextract": output / "cleaned.needs_source_reextract.jsonl",
    }
    handles = {route: path.open("w", encoding="utf-8") for route, path in route_paths.items()}
    sample_handle = (output / "cleaning_quality_samples.jsonl").open("w", encoding="utf-8")
    try:
        route_counts: Counter[str] = Counter()
        reason_counts: Counter[str] = Counter()
        source_route_counts: dict[str, Counter[str]] = defaultdict(Counter)
        totals: Counter[str] = Counter()
        sample_counts: Counter[tuple[str, str]] = Counter()

        for index, record in enumerate(iter_jsonl(input_path), start=1):
            route, reasons, metrics = classify_record(record)
            source = str(dict(record.get("metadata") or {}).get("source") or record.get("source") or "unknown")
            route_counts[route] += 1
            source_route_counts[source][route] += 1
            totals["records"] += 1
            totals["raw_chars"] += metrics["raw_chars"]
            totals["clean_chars"] += metrics["clean_chars"]
            totals["references_chars"] += metrics["references_chars"]
            for reason in reasons:
                reason_counts[reason] += 1
                key = (route, reason)
                if sample_counts[key] < sample_limit:
                    sample_counts[key] += 1
                    write_jsonl_obj(
                        sample_handle,
                        {
                            "record_index": index,
                            "record_id": record.get("record_id"),
                            "source": source,
                            "title": dict(record.get("metadata") or {}).get("title") or record.get("title"),
                            "route": route,
                            "reason": reason,
                            "metrics": metrics,
                            "text_preview": str(record.get("clean_content") or "")[:500],
                        },
                    )
            write_jsonl_obj(handles[route], routed_record(record, route, reasons, metrics))

    finally:
        for handle in handles.values():
            handle.close()
        sample_handle.close()

    summary = {
        "input_path": str(input_path),
        "output_dir": str(output),
        "route_counts": dict(route_counts),
        "reason_counts": dict(reason_counts),
        "source_route_counts": {source: dict(counts) for source, counts in sorted(source_route_counts.items())},
        "totals": dict(totals),
    }
    with (output / "cleaning_quality_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route cleaned records into ready/review/reextract queues.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-limit", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = audit_file(args.input, args.output_dir, sample_limit=args.sample_limit)
    print(json.dumps(summary["route_counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
