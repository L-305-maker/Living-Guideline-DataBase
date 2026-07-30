"""JSON schemas expected from model adapters."""

from __future__ import annotations

EXTRACTION_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["is_formal_recommendation", "recommendation_type", "recommendation_text", "field_evidence"],
    "properties": {
        "is_formal_recommendation": {"type": "boolean"},
        "recommendation_type": {"type": "string"},
        "recommendation_text": {"type": "string"},
        "direction": {"type": ["string", "null"]},
        "strength": {"type": ["string", "null"]},
        "certainty": {"type": ["string", "null"]},
        "population": {"type": ["string", "null"]},
        "interventions": {"type": "array", "items": {"type": "string"}},
        "dosage": {"type": ["string", "null"]},
        "duration": {"type": ["string", "null"]},
        "conditions": {"type": "array", "items": {"type": "string"}},
        "field_evidence": {"type": "object"},
    },
}

VERIFICATION_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["agrees_is_formal_recommendation", "recommended_route"],
    "properties": {
        "agrees_is_formal_recommendation": {"type": ["boolean", "null"]},
        "field_agreements": {"type": "object"},
        "field_conflicts": {"type": "object"},
        "unsupported_fields": {"type": "array", "items": {"type": "string"}},
        "logic_errors": {"type": "array", "items": {"type": "string"}},
        "recommended_route": {"type": "string"},
    },
}
