import time
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from legal_state.json_grammar import compact_json_grammar

__all__ = ["ModelCallResult", "ModelClient"]


class ModelCallResult(BaseModel):
    """一次成功模型调用的原始文本与实验元数据。"""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    raw_text: str
    model: str = Field(min_length=1)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_seconds: float = Field(ge=0)
    finish_reason: str | None = None
    reasoning_tokens: int | None = Field(default=None, ge=0)


class _ResponseMessage(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    content: str


class _ResponseChoice(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    message: _ResponseMessage
    finish_reason: str | None = None


class _ResponseUsage(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    completion_tokens_details: dict[str, int | object] | None = None


class _ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    model: str = Field(min_length=1)
    choices: list[_ResponseChoice] = Field(min_length=1)
    usage: _ResponseUsage


class ModelClient:
    """同步调用一个固定的 OpenAI-compatible Chat Completions 模型。"""

    def __init__(
        self,
        api_key: str,
        model: str,
        endpoint: str,
        *,
        timeout_seconds: float = 60.0,
        temperature: float = 0,
        max_output_tokens: int = 2048,
        provider_options: dict[str, object] | None = None,
        compact_json: bool = False,
        compact_json_scope: Literal["all", "audit"] = "all",
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        if not model:
            raise ValueError("model must not be empty")
        if not endpoint:
            raise ValueError("endpoint must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        reserved = {"model", "messages", "temperature", "max_tokens", "n"}
        if provider_options is not None and reserved.intersection(provider_options):
            raise ValueError("provider_options cannot override controlled parameters")

        self._api_key = api_key
        self._model = model
        self._endpoint = endpoint
        self._timeout_seconds = timeout_seconds
        self._temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._compact_json = compact_json
        if compact_json_scope not in {"all", "audit"}:
            raise ValueError("Unknown compact JSON scope")
        self._compact_json_scope = compact_json_scope
        self._provider_options = (
            {"thinking": {"type": "disabled"}}
            if provider_options is None
            else dict(provider_options)
        )
        self.last_response: str | None = None

    def generate(
        self,
        prompt: str,
        *,
        max_output_tokens: int | None = None,
        response_format: dict[str, object] | None = None,
    ) -> ModelCallResult:
        """发送一次请求；不解析、修复或重试模型生成的文本。"""
        if not prompt:
            raise ValueError("prompt must not be empty")
        limit = (
            self.max_output_tokens if max_output_tokens is None else max_output_tokens
        )
        if limit <= 0:
            raise ValueError("max_output_tokens must be positive")
        self.last_response = None

        json_options = {}
        compact_request = self._compact_json and response_format is not None
        if compact_request and self._compact_json_scope == "audit":
            schema = response_format.get("json_schema", {}).get("schema", {})
            compact_request = schema.get("properties", {}).get("operation", {}).get("const") == "AUDIT_OPTIONS"
        if compact_request:
            if response_format.get("type") != "json_schema":
                raise ValueError("Compact JSON requires a json_schema response format")
            json_options = {"structured_outputs": {
                "grammar": compact_json_grammar(response_format["json_schema"]["schema"]),
            }}

        started_at = time.perf_counter()
        response = httpx.post(
            self._endpoint,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                **self._provider_options,
                **json_options,
                **({"response_format": response_format}
                   if response_format is not None and not compact_request else {}),
                "temperature": self._temperature,
                "max_tokens": limit,
                "n": 1,
            },
            timeout=self._timeout_seconds,
        )
        self.last_response = response.text
        response.raise_for_status()
        payload = _ChatCompletionResponse.model_validate_json(response.content)
        latency_seconds = time.perf_counter() - started_at

        return ModelCallResult(
            raw_text=payload.choices[0].message.content,
            model=payload.model,
            input_tokens=payload.usage.prompt_tokens,
            output_tokens=payload.usage.completion_tokens,
            latency_seconds=latency_seconds,
            finish_reason=payload.choices[0].finish_reason,
            reasoning_tokens=(
                (payload.usage.completion_tokens_details or {}).get("reasoning_tokens")
            ),
        )
