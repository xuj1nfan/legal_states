import time

import httpx
from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelCallResult", "ModelClient"]


class ModelCallResult(BaseModel):
    """一次成功模型调用的原始文本与实验元数据。"""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    raw_text: str
    model: str = Field(min_length=1)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_seconds: float = Field(ge=0)


class _ResponseMessage(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    content: str


class _ResponseChoice(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    message: _ResponseMessage


class _ResponseUsage(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)


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
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        if not model:
            raise ValueError("model must not be empty")
        if not endpoint:
            raise ValueError("endpoint must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        self._api_key = api_key
        self._model = model
        self._endpoint = endpoint
        self._timeout_seconds = timeout_seconds

    def generate(self, prompt: str) -> ModelCallResult:
        """发送一次请求；不解析、修复或重试模型生成的文本。"""
        if not prompt:
            raise ValueError("prompt must not be empty")

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
                "thinking": {"type":"disabled"},
                "temperature": 0,
                "max_tokens": 2048,
                "n": 1,
            },
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        payload = _ChatCompletionResponse.model_validate_json(response.content)
        latency_seconds = time.perf_counter() - started_at

        return ModelCallResult(
            raw_text=payload.choices[0].message.content,
            model=payload.model,
            input_tokens=payload.usage.prompt_tokens,
            output_tokens=payload.usage.completion_tokens,
            latency_seconds=latency_seconds,
        )
