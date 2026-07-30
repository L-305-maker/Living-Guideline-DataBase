"""Provider-neutral structured model client boundary."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import Field

from src.guideline_information.models import FrozenModel


class ModelResponse(FrozenModel):
    request_id: str
    provider: str
    model_name: str
    raw_text: str
    parsed_json: dict[str, Any]
    usage: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = 0
    finish_reason: str = ""
    response_metadata: dict[str, Any] = Field(default_factory=dict)
    parsed_with_repair: bool = False


class StructuredModelClient(Protocol):
    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, object],
        request_id: str,
    ) -> ModelResponse:
        ...


class ModelClientError(RuntimeError):
    failure_type = "MODEL_CLIENT_ERROR"


class ModelConfigurationError(ModelClientError):
    failure_type = "MODEL_CONFIGURATION_ERROR"


class ModelNetworkError(ModelClientError):
    failure_type = "MODEL_NETWORK_ERROR"


class ModelFormatError(ModelClientError):
    failure_type = "MODEL_FORMAT_ERROR"


class ModelRateLimitError(ModelClientError):
    failure_type = "MODEL_RATE_LIMIT"


class ModelRefusalError(ModelClientError):
    failure_type = "MODEL_REFUSAL"
