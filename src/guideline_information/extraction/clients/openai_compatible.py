"""OpenAI-compatible structured JSON client.

This module reads credentials only from environment variables and never logs the
API key. It is not used by tests.
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from typing import Any

from src.guideline_information.extraction.clients.base import (
    ModelConfigurationError,
    ModelNetworkError,
    ModelRateLimitError,
    ModelResponse,
)
from src.guideline_information.extraction.json_parser import parse_json_object
from src.guideline_information.extraction.model_config import effective_max_output_tokens, effective_timeout_seconds, get_model_config_value


class OpenAICompatibleStructuredModelClient:
    def __init__(self, *, max_retries: int = 2) -> None:
        self.base_url = get_model_config_value("GUIDELINE_LLM_BASE_URL").rstrip("/")
        self.api_key = get_model_config_value("GUIDELINE_LLM_API_KEY")
        self.model_name = get_model_config_value("GUIDELINE_LLM_MODEL")
        self.timeout = effective_timeout_seconds()
        self.max_output_tokens = effective_max_output_tokens()
        self.thinking_mode = get_model_config_value("GUIDELINE_LLM_THINKING", default="disabled").lower()
        self.max_retries = max(0, max_retries)
        if not self.base_url or not self.api_key or not self.model_name:
            raise ModelConfigurationError(
                "OpenAI-compatible client requires GUIDELINE_LLM_BASE_URL, GUIDELINE_LLM_API_KEY, and GUIDELINE_LLM_MODEL"
            )

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, object],
        request_id: str,
    ) -> ModelResponse:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": self.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        if _uses_deepseek_v4(self.base_url, self.model_name) and self.thinking_mode in {"enabled", "disabled"}:
            payload["thinking"] = {"type": self.thinking_mode}
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        url = f"{self.base_url}/chat/completions"
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            start = time.perf_counter()
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                latency_ms = int((time.perf_counter() - start) * 1000)
                choice = data.get("choices", [{}])[0]
                raw_text = choice.get("message", {}).get("content", "")
                parsed, repaired = parse_json_object(raw_text)
                return ModelResponse(
                    request_id=request_id,
                    provider="openai-compatible",
                    model_name=self.model_name,
                    raw_text=raw_text,
                    parsed_json=parsed,
                    usage=data.get("usage") or {},
                    latency_ms=latency_ms,
                    finish_reason=choice.get("finish_reason") or "",
                    response_metadata={"id": data.get("id"), "created": data.get("created"), "retry_count": attempt},
                    parsed_with_repair=repaired,
                )
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code == 429:
                    if attempt >= self.max_retries:
                        raise ModelRateLimitError("Model provider returned HTTP 429") from exc
                elif 400 <= exc.code < 500:
                    raise ModelNetworkError(f"Model provider returned HTTP {exc.code}") from exc
            except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise ModelNetworkError("Model provider request failed") from exc
            time.sleep(min(2 ** attempt, 4))
        raise ModelNetworkError("Model provider request failed") from last_error


def _uses_deepseek_v4(base_url: str, model_name: str) -> bool:
    return "deepseek" in (base_url or "").lower() and (model_name or "").startswith("deepseek-v4")
