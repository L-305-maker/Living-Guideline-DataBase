from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.common.extraction_common import JsonDict


DEFAULT_ENV_FILE = ".env"
DEFAULT_RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
DEFAULT_MAX_RESPONSE_BYTES = 2_000_000


@dataclass(frozen=True)
class LLMRequest:
    prompt: str
    model: str
    api_url: str
    api_key_env: str
    env_file: str | Path
    api_format: str
    temperature: float
    timeout_seconds: int
    max_retries: int = 3
    retry_base_seconds: float = 1.0
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES


HttpPost = Callable[..., Any]


def load_env_file(env_file: str | Path = DEFAULT_ENV_FILE, override: bool = False) -> list[str]:
    env_path = Path(env_file)
    if not env_path.exists():
        return []

    loaded: list[str] = []
    with env_path.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            if override or key not in os.environ:
                os.environ[key] = value
                loaded.append(key)
    return loaded


def resolve_api_key(api_key_env: str = "OPENAI_API_KEY", env_file: str | Path = DEFAULT_ENV_FILE) -> str:
    load_env_file(env_file)
    for key in dict.fromkeys([api_key_env, "OPENAI_API_KEY", "API_KEY"]):
        value = os.getenv(key)
        if value:
            return value
    raise RuntimeError(f"Missing API key. Checked environment variables: {api_key_env}, OPENAI_API_KEY, API_KEY")


def build_llm_payload(prompt: str, model: str, api_format: str, temperature: float) -> JsonDict:
    if api_format == "responses":
        return {
            "model": model,
            "input": prompt,
            "temperature": temperature,
        }
    if api_format == "chat_completions":
        return {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You return strict JSON only for clinical guideline data extraction review.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
    raise ValueError(f"Unsupported api_format: {api_format}")


def _short_response_detail(text: str, limit: int = 500) -> str:
    detail = text.strip().replace("\n", " ")
    if len(detail) > limit:
        return f"{detail[:limit]}..."
    return detail


def _retry_delay_seconds(response: Any, attempt: int, base_seconds: float) -> float:
    retry_after = getattr(response, "headers", {}).get("Retry-After") if response is not None else None
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass
    return max(0.0, base_seconds) * (2 ** max(0, attempt - 1))


def _response_request_id(response: Any) -> str:
    headers = getattr(response, "headers", {}) or {}
    return str(headers.get("x-request-id") or headers.get("request-id") or "")


def post_llm_request(request: LLMRequest, post: HttpPost | None = None, sleep: Callable[[float], None] = time.sleep) -> JsonDict:
    api_key = resolve_api_key(api_key_env=request.api_key_env, env_file=request.env_file)

    import requests

    post_func = post or requests.post
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = build_llm_payload(request.prompt, request.model, request.api_format, request.temperature)
    attempts = max(1, request.max_retries + 1)
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        response = None
        try:
            response = post_func(
                request.api_url,
                headers=headers,
                json=payload,
                timeout=request.timeout_seconds,
            )
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= attempts:
                raise RuntimeError(f"LLM API request failed after {attempt} attempts: {exc}") from exc
            sleep(_retry_delay_seconds(None, attempt, request.retry_base_seconds))
            continue

        request_id = _response_request_id(response)
        response_size = len(getattr(response, "content", b"") or b"")
        if response_size > request.max_response_bytes:
            suffix = f" request_id={request_id}" if request_id else ""
            raise RuntimeError(
                f"LLM API response exceeded {request.max_response_bytes} bytes "
                f"({response_size} bytes).{suffix}"
            )

        if response.ok:
            parsed: Any = response.json()
            if not isinstance(parsed, dict):
                suffix = f" request_id={request_id}" if request_id else ""
                raise RuntimeError(f"LLM API response must be a JSON object.{suffix}")
            if request_id and "_request_id" not in parsed:
                parsed["_request_id"] = request_id
            if attempt > 1 and "_retry_attempts" not in parsed:
                parsed["_retry_attempts"] = attempt - 1
            return parsed

        if response.status_code in DEFAULT_RETRY_STATUSES and attempt < attempts:
            sleep(_retry_delay_seconds(response, attempt, request.retry_base_seconds))
            continue

        detail = _short_response_detail(response.text)
        suffix = f" request_id={request_id}" if request_id else ""
        raise RuntimeError(f"LLM API request failed with HTTP {response.status_code}:{suffix} {detail}")

    if last_error is not None:
        raise RuntimeError(f"LLM API request failed: {last_error}") from last_error
    raise RuntimeError("LLM API request failed without a response")
