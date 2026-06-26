from __future__ import annotations

from typing import Any, Dict, List


JsonDict = Dict[str, Any]


def _valid(rows: List[JsonDict]) -> List[JsonDict]:
    return [row for row in rows if (row.get("validation") or {}).get("is_valid")]


def map_to_recommendation_versions(payload: JsonDict) -> List[JsonDict]:
    rows = _valid(payload.get("entities", {}).get("recommendation_versions", []))
    return [
        {
            "recommendation_version_id": row["recommendation_version_id"],
            "recommendation_id": row["recommendation_id"],
            "recommendation_candidate_id": row["recommendation_id"],
            "guideline_id": row.get("guideline_id") or "unknown_guideline",
            "version_number": "v1",
            "recommendation_text": row["recommendation_text"],
            "record_id": row.get("source_record_id"),
            "quality_status": "candidate_accepted",
            "recommendation_code": row.get("recommendation_code"),
            "direction": row.get("direction"),
            "strength": row.get("strength"),
            "certainty": row.get("certainty"),
            "rationale": row.get("rationale"),
            "remarks": row.get("remark"),
            "population": row.get("population"),
            "intervention": row.get("intervention"),
            "comparator": row.get("comparator"),
            "source_text": row.get("source_span"),
            "source_span": row.get("source_span"),
            "source_span_ref": row.get("source_span_ref") or row.get("recommendation_id"),
            "start_char": row.get("start_char"),
            "end_char": row.get("end_char"),
            "source_section": row.get("source_section"),
            "normalized_payload": {
                "source_span": row.get("source_span"),
                "source_span_ref": row.get("recommendation_id"),
                "start_char": row.get("start_char"),
                "end_char": row.get("end_char"),
                "quality_score": row.get("quality_score"),
            },
            "raw_payload": row,
        }
        for row in rows
    ]


def map_payload_to_business_tables(payload: JsonDict) -> JsonDict:
    entities = payload.get("entities", {})
    return {
        "model_traces": list(entities.get("model_traces", [])),
        "recommendation_versions": map_to_recommendation_versions(payload),
        "pico_questions": _valid(entities.get("pico_questions", [])),
        "evidence_items": _valid(entities.get("evidence_items", [])),
        "grade_assessments": _valid(entities.get("grade_assessments", [])),
        "update_logs": list(entities.get("update_logs", [])),
    }
