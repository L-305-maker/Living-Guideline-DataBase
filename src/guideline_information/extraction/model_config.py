"""Safe model configuration loading from environment and local .env."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REQUIRED_KEYS = [
    "GUIDELINE_LLM_BASE_URL",
    "GUIDELINE_LLM_API_KEY",
    "GUIDELINE_LLM_MODEL",
    "GUIDELINE_LLM_TIMEOUT_SECONDS",
]
ALIASES = {
    "GUIDELINE_LLM_BASE_URL": ("GUIDELINE_LLM_BASE_URL",),
    "GUIDELINE_LLM_API_KEY": ("GUIDELINE_LLM_API_KEY", "API_KEY"),
    "GUIDELINE_LLM_MODEL": ("GUIDELINE_LLM_MODEL",),
    "GUIDELINE_LLM_TIMEOUT_SECONDS": ("GUIDELINE_LLM_TIMEOUT_SECONDS",),
    "GUIDELINE_LLM_MAX_OUTPUT_TOKENS": ("GUIDELINE_LLM_MAX_OUTPUT_TOKENS",),
}
DEFAULT_TIMEOUT_SECONDS = "120"
DEFAULT_MAX_OUTPUT_TOKENS = "4096"


def get_model_config_value(name: str, *, cwd: str | Path | None = None, default: str = "") -> str:
    for key in ALIASES.get(name, (name,)):
        value = os.getenv(key)
        if value:
            return value
    dotenv = read_dotenv(cwd=cwd)
    for key in ALIASES.get(name, (name,)):
        value = dotenv.get(key)
        if value:
            return value
    return default


def get_api_key_source(*, cwd: str | Path | None = None) -> str:
    dotenv = read_dotenv(cwd=cwd)
    if os.getenv("GUIDELINE_LLM_API_KEY") or dotenv.get("GUIDELINE_LLM_API_KEY"):
        return "GUIDELINE_LLM_API_KEY"
    if os.getenv("API_KEY") or dotenv.get("API_KEY"):
        return "API_KEY"
    return ""


def effective_timeout_seconds(*, cwd: str | Path | None = None) -> int:
    raw = get_model_config_value("GUIDELINE_LLM_TIMEOUT_SECONDS", cwd=cwd, default=DEFAULT_TIMEOUT_SECONDS)
    try:
        value = int(float(raw))
    except ValueError as exc:
        raise ValueError("GUIDELINE_LLM_TIMEOUT_SECONDS must be numeric") from exc
    if value <= 0:
        raise ValueError("GUIDELINE_LLM_TIMEOUT_SECONDS must be positive")
    return value




def effective_max_output_tokens(*, cwd: str | Path | None = None) -> int:
    raw = get_model_config_value("GUIDELINE_LLM_MAX_OUTPUT_TOKENS", cwd=cwd, default=DEFAULT_MAX_OUTPUT_TOKENS)
    try:
        value = int(float(raw))
    except ValueError as exc:
        raise ValueError("GUIDELINE_LLM_MAX_OUTPUT_TOKENS must be numeric") from exc
    if value <= 0:
        raise ValueError("GUIDELINE_LLM_MAX_OUTPUT_TOKENS must be positive")
    return value


def model_configured_flags(*, cwd: str | Path | None = None) -> dict[str, bool]:
    return {
        "base_url_configured": bool(get_model_config_value("GUIDELINE_LLM_BASE_URL", cwd=cwd)),
        "api_key_configured": bool(get_model_config_value("GUIDELINE_LLM_API_KEY", cwd=cwd)),
        "model_configured": bool(get_model_config_value("GUIDELINE_LLM_MODEL", cwd=cwd)),
        "timeout_configured": bool(effective_timeout_seconds(cwd=cwd)),
        "max_output_tokens_configured": bool(effective_max_output_tokens(cwd=cwd)),
    }


def model_config_audit(*, probe: bool = False, show_models: bool = False, cwd: str | Path | None = None) -> dict[str, Any]:
    base_url = get_model_config_value("GUIDELINE_LLM_BASE_URL", cwd=cwd).rstrip("/")
    api_key = get_model_config_value("GUIDELINE_LLM_API_KEY", cwd=cwd)
    model = get_model_config_value("GUIDELINE_LLM_MODEL", cwd=cwd)
    timeout = effective_timeout_seconds(cwd=cwd)
    max_output_tokens = effective_max_output_tokens(cwd=cwd)
    flags = model_configured_flags(cwd=cwd)
    missing = []
    if not flags["base_url_configured"]:
        missing.append("GUIDELINE_LLM_BASE_URL")
    if not flags["api_key_configured"]:
        missing.append("GUIDELINE_LLM_API_KEY")
    if not flags["model_configured"]:
        missing.append("GUIDELINE_LLM_MODEL")
    audit: dict[str, Any] = {
        "provider": "openai-compatible" if base_url else "UNCONFIRMED",
        "provider_confirmed": bool(base_url),
        "base_url_configured": flags["base_url_configured"],
        "api_key_configured": flags["api_key_configured"],
        "api_key_source": get_api_key_source(cwd=cwd),
        "model_configured": flags["model_configured"],
        "timeout_configured": flags["timeout_configured"],
        "timeout_seconds": timeout,
        "max_output_tokens": max_output_tokens,
        "authentication_valid": None,
        "configured_model_accessible": None,
        "network_called": False,
        "probe_status": "NOT_RUN",
        "missing": missing,
    }
    if not probe:
        return audit
    if not base_url:
        audit["probe_status"] = "BLOCKED_PROVIDER_UNCONFIRMED"
        return audit
    if missing:
        audit["probe_status"] = "BLOCKED_MODEL_CONFIGURATION_MISSING"
        return audit
    result = _probe_models_endpoint(base_url=base_url, api_key=api_key, model=model, timeout=timeout, show_models=show_models)
    audit.update(result)
    return audit


def read_dotenv(*, cwd: str | Path | None = None) -> dict[str, str]:
    path = Path(cwd or Path.cwd()) / ".env"
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _probe_models_endpoint(*, base_url: str, api_key: str, model: str, timeout: int, show_models: bool) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            return {
                "authentication_valid": False,
                "configured_model_accessible": False,
                "network_called": True,
                "http_status": exc.code,
                "probe_status": "BLOCKED_AUTHENTICATION_FAILED",
            }
        return {
            "authentication_valid": None,
            "configured_model_accessible": False,
            "network_called": True,
            "http_status": exc.code,
            "probe_status": "BLOCKED_MODEL_UNAVAILABLE",
        }
    except TimeoutError:
        return {"authentication_valid": None, "configured_model_accessible": False, "network_called": True, "probe_status": "TIMEOUT"}
    except Exception:
        return {"authentication_valid": None, "configured_model_accessible": False, "network_called": True, "probe_status": "NETWORK_ERROR"}
    ids = [str(item.get("id") or "") for item in payload.get("data") or [] if isinstance(item, dict)]
    accessible = model in ids
    result: dict[str, Any] = {
        "authentication_valid": True,
        "configured_model_accessible": accessible,
        "network_called": True,
        "http_status": status,
        "probe_status": "PASS" if accessible else "BLOCKED_CONFIGURED_MODEL_UNAVAILABLE",
    }
    if show_models:
        result["models"] = ids
    return result
