import time

import httpx
import pytest
from pydantic import ValidationError

from legal_state.model import ModelCallResult, ModelClient

ENDPOINT = "https://model.example.test/v1/chat/completions"


def make_response(payload: dict[str, object], status_code: int = 200) -> httpx.Response:
    request = httpx.Request("POST", ENDPOINT)
    return httpx.Response(status_code, request=request, json=payload)


def valid_response(content: str = '{"operation":"STOP"}') -> dict[str, object]:
    return {
        "id": "call-1",
        "model": "fixed-model-2026-01-01",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 8,
            "total_tokens": 128,
        },
    }


def test_generate_returns_raw_text_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_text = "```json\n{not valid action json}\n```"
    captured: dict[str, object] = {}

    def fake_post(
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, object],
        timeout: float,
    ) -> httpx.Response:
        captured.update(
            url=url,
            headers=headers,
            json=json,
            timeout=timeout,
        )
        return make_response(valid_response(raw_text))

    times = iter([10.0, 10.25])
    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(time, "perf_counter", lambda: next(times))
    client = ModelClient(
        api_key="secret-key",
        model="fixed-model",
        endpoint=ENDPOINT,
        timeout_seconds=30.0,
    )

    result = client.generate("生成下一步行动")

    assert result == ModelCallResult(
        raw_text=raw_text,
        model="fixed-model-2026-01-01",
        input_tokens=120,
        output_tokens=8,
        latency_seconds=0.25,
    )
    assert captured == {
        "url": ENDPOINT,
        "headers": {
            "Authorization": "Bearer secret-key",
            "Content-Type": "application/json",
        },
        "json": {
            "model": "fixed-model",
            "messages": [{"role": "user", "content": "生成下一步行动"}],
            "temperature": 0,
            "max_tokens": 2048,
            "n": 1,
        },
        "timeout": 30.0,
    }


def test_generate_propagates_http_error_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    def fake_post(
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, object],
        timeout: float,
    ) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return make_response({"error": "rate limited"}, status_code=429)

    monkeypatch.setattr(httpx, "post", fake_post)
    client = ModelClient("secret-key", "fixed-model", ENDPOINT)

    with pytest.raises(httpx.HTTPStatusError):
        client.generate("prompt")

    assert call_count == 1


@pytest.mark.parametrize(
    "payload",
    [
        {
            "model": "fixed-model",
            "choices": [{"message": {"content": '{"operation":"STOP"}'}}],
        },
        {
            "model": "fixed-model",
            "choices": [],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
        {
            "model": "fixed-model",
            "choices": [{"message": {"content": None}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    ],
)
def test_generate_rejects_incomplete_provider_response(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
) -> None:
    def fake_post(
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, object],
        timeout: float,
    ) -> httpx.Response:
        return make_response(payload)

    monkeypatch.setattr(httpx, "post", fake_post)
    client = ModelClient("secret-key", "fixed-model", ENDPOINT)

    with pytest.raises(ValidationError):
        client.generate("prompt")


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("", "model", ENDPOINT), "api_key"),
        (("key", "", ENDPOINT), "model"),
        (("key", "model", ""), "endpoint"),
    ],
)
def test_client_rejects_empty_configuration(
    arguments: tuple[str, str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ModelClient(*arguments)


def test_client_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        ModelClient("key", "model", ENDPOINT, timeout_seconds=0)


def test_generate_rejects_empty_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> httpx.Response:
        raise AssertionError("HTTP request should not be sent")

    monkeypatch.setattr(httpx, "post", fail_if_called)
    client = ModelClient("key", "model", ENDPOINT)

    with pytest.raises(ValueError, match="prompt"):
        client.generate("")
