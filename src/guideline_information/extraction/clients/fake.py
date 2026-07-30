"""Fake structured client for tests and offline smoke runs."""

from __future__ import annotations

import json
from collections import deque
from typing import Any, Iterable

from src.guideline_information.extraction.clients.base import ModelResponse
from src.guideline_information.extraction.json_parser import parse_json_object


class FakeStructuredModelClient:
    def __init__(self, responses: Iterable[str | dict[str, Any]] | None = None) -> None:
        self.responses = deque(responses or [])
        self.calls: list[dict[str, Any]] = []
        self.model_name = "fake-structured-model"

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, object],
        request_id: str,
    ) -> ModelResponse:
        self.calls.append({"request_id": request_id, "system_prompt": system_prompt, "user_prompt": user_prompt})
        payload = self.responses.popleft() if self.responses else self._heuristic_payload(user_prompt)
        raw_text = json.dumps(payload, ensure_ascii=False) if isinstance(payload, dict) else str(payload)
        parsed, repaired = parse_json_object(raw_text)
        return ModelResponse(
            request_id=request_id,
            provider="fake",
            model_name=self.model_name,
            raw_text=raw_text,
            parsed_json=parsed,
            usage={"input_tokens": 0, "output_tokens": 0},
            latency_ms=0,
            finish_reason="stop",
            parsed_with_repair=repaired,
        )

    def _heuristic_payload(self, user_prompt: str) -> dict[str, Any]:
        text = _candidate_line(user_prompt)
        lower = text.lower()
        is_recommendation = "recommend" in lower or "should" in lower or "suggest" in lower
        if "verification task" in lower:
            return {
                "agrees_is_formal_recommendation": is_recommendation,
                "field_agreements": {"recommendation_text": True},
                "field_conflicts": {},
                "unsupported_fields": [],
                "logic_errors": [],
                "recommended_route": "AUTO_ACCEPT" if is_recommendation else "REJECT",
            }
        quote = _first_sentence(text)
        return {
            "is_formal_recommendation": is_recommendation,
            "recommendation_type": "FORMAL_RECOMMENDATION" if is_recommendation else "NOT_RECOMMENDATION",
            "recommendation_text": quote if is_recommendation else "",
            "direction": "FOR" if is_recommendation else None,
            "strength": "NOT_STATED",
            "certainty": "NOT_STATED",
            "population": "NOT_STATED",
            "interventions": [],
            "dosage": None,
            "duration": None,
            "conditions": [],
            "field_evidence": {
                "recommendation_text": [
                    {
                        "value_original": quote,
                        "value_normalized": quote,
                        "source_type": "EXPLICIT_IN_RECOMMENDATION",
                        "source_block_key": "",
                        "quote": quote,
                        "span_start": 0,
                        "span_end": len(quote),
                        "confidence_signal": "fake_rule",
                    }
                ]
            },
        }


def _candidate_line(prompt: str) -> str:
    marker = "Candidate text:"
    if marker not in prompt:
        return prompt
    return prompt.split(marker, 1)[1].splitlines()[0].strip()


def _first_sentence(text: str) -> str:
    stripped = " ".join((text or "").split())
    for sep in [". ", "\n"]:
        if sep in stripped:
            return stripped.split(sep, 1)[0].strip() + ("." if sep == ". " else "")
    return stripped[:500]
