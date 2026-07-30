"""Strict JSON object parsing for model output."""

from __future__ import annotations

import json
import re
from typing import Any

from src.guideline_information.extraction.clients.base import ModelFormatError

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S | re.I)


def parse_json_object(text: str) -> tuple[dict[str, Any], bool]:
    raw = (text or "").strip()
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ModelFormatError("Model output must be a JSON object")
        return parsed, False
    except json.JSONDecodeError:
        pass

    match = _CODE_FENCE_RE.search(raw)
    candidate = match.group(1) if match else _extract_first_object(raw)
    if candidate is None:
        raise ModelFormatError("Model output did not contain a JSON object")
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ModelFormatError(f"Invalid JSON object in model output: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ModelFormatError("Model output must be a JSON object")
    return parsed, True


def _extract_first_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None
