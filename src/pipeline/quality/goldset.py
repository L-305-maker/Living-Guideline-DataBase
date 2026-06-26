from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.common.process_jsonl import iter_jsonl, write_jsonl


JsonDict = Dict[str, Any]
DEFAULT_FIELDS = ("recommendation_text", "direction", "strength", "certainty", "population", "intervention", "comparator")


def _entity_id(row: JsonDict) -> str:
    for field in ("entity_id", "recommendation_version_id", "candidate_id", "pico_id", "evidence_id", "grade_candidate_id"):
        value = row.get(field)
        if value:
            return str(value)
    return ""


def _expected(row: JsonDict) -> JsonDict:
    value = row.get("expected")
    return value if isinstance(value, dict) else row


def _index(rows: Iterable[JsonDict]) -> Dict[str, JsonDict]:
    indexed: Dict[str, JsonDict] = {}
    for row in rows:
        entity_id = _entity_id(row)
        if entity_id:
            indexed[entity_id] = row
    return indexed


def _same_value(left: Any, right: Any) -> bool:
    if left is None and right in {"", None}:
        return True
    if right is None and left in {"", None}:
        return True
    return str(left).strip().lower() == str(right).strip().lower()


def _offset_match(expected: JsonDict, predicted: JsonDict) -> bool | None:
    raw_start = expected.get("raw_start_char")
    raw_end = expected.get("raw_end_char")
    if raw_start is None or raw_end is None:
        return None
    return raw_start == predicted.get("raw_start_char", predicted.get("start_char")) and raw_end == predicted.get("raw_end_char", predicted.get("end_char"))


def evaluate_goldset(
    gold_rows: Iterable[JsonDict],
    prediction_rows: Iterable[JsonDict],
    *,
    fields: Iterable[str] = DEFAULT_FIELDS,
) -> JsonDict:
    """用人工 goldset 评估预测实体。

    评估分两层：
    1. entity precision/recall/f1 判断实体是否抽到、是否多抽；
    2. field_accuracy 和 raw_offset_accuracy 判断字段值与原文定位是否正确。
    """

    gold = _index(gold_rows)
    predictions = _index(prediction_rows)
    expected_ids = set(gold)
    predicted_ids = set(predictions)
    matched_ids = expected_ids & predicted_ids
    field_totals: Counter[str] = Counter()
    field_hits: Counter[str] = Counter()
    offset_total = 0
    offset_hits = 0
    mismatches: List[JsonDict] = []

    for entity_id in sorted(matched_ids):
        expected = _expected(gold[entity_id])
        predicted = predictions[entity_id]
        for field in fields:
            if field not in expected:
                continue
            field_totals[field] += 1
            if _same_value(expected.get(field), predicted.get(field)):
                field_hits[field] += 1
            else:
                mismatches.append(
                    {
                        "entity_id": entity_id,
                        "field": field,
                        "expected": expected.get(field),
                        "predicted": predicted.get(field),
                    }
                )
        offset_match = _offset_match(expected, predicted)
        if offset_match is not None:
            offset_total += 1
            offset_hits += int(offset_match)

    precision = len(matched_ids) / len(predicted_ids) if predicted_ids else 0.0
    recall = len(matched_ids) / len(expected_ids) if expected_ids else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    field_accuracy = {
        field: round(field_hits[field] / total, 4)
        for field, total in field_totals.items()
    }
    return {
        "gold_entities": len(expected_ids),
        "predicted_entities": len(predicted_ids),
        "matched_entities": len(matched_ids),
        "missing_prediction_ids": sorted(expected_ids - predicted_ids),
        "extra_prediction_ids": sorted(predicted_ids - expected_ids),
        "entity_precision": round(precision, 4),
        "entity_recall": round(recall, 4),
        "entity_f1": round(f1, 4),
        "field_accuracy": field_accuracy,
        "raw_offset_accuracy": round(offset_hits / offset_total, 4) if offset_total else None,
        "mismatches": mismatches,
    }


def evaluate_goldset_file(gold_input: str | Path, predictions_input: str | Path, report_output: str | Path) -> JsonDict:
    report = evaluate_goldset(iter_jsonl(gold_input), iter_jsonl(predictions_input))
    write_jsonl(report_output, [report])
    return report
