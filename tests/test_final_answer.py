import json
from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from legal_state.final_answer import (
    FinalAnswer,
    FinalAnswerError,
    build_final_answer_prompt,
    generate_final_answer,
    parse_final_answer_json,
)
from legal_state.model import ModelCallResult


class ScriptedModelClient:
    def __init__(self, responses: Sequence[str | Exception]) -> None:
        self._responses = iter(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> ModelCallResult:
        self.prompts.append(prompt)
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        call_number = len(self.prompts)
        return ModelCallResult(
            raw_text=response,
            model="scripted-final-answer-model",
            input_tokens=200 + call_number,
            output_tokens=20 + call_number,
            latency_seconds=call_number / 10,
        )


def test_final_answer_accepts_valid_answer() -> None:
    answer = parse_final_answer_json('{"answer":"乙应返还借款本金。"}')

    assert answer == FinalAnswer(answer="乙应返还借款本金。")


@pytest.mark.parametrize(
    "payload",
    [
        '{"answer":""}',
        '{"answer":"答案","unexpected":true}',
        '```json\n{"answer":"答案"}\n```',
        "普通文本答案",
        "{\"reasoning\":\"答案\"}",
    ],
)
def test_parse_final_answer_rejects_invalid_json_or_schema(payload: str) -> None:
    with pytest.raises(ValidationError):
        parse_final_answer_json(payload)


def test_build_prompt_contains_all_inputs_and_no_method_names() -> None:
    prompt = build_final_answer_prompt(
        case_text="甲向乙交付十万元。",
        question="乙是否应返还借款？",
        provided_knowledge=("借款人应按期返还借款。",),
        reasoning_artifact='{"conclusions":[{"content":"乙应返还借款"}]}',
    )

    input_data = json.loads(prompt.split("输入数据：\n", maxsplit=1)[1])
    assert input_data == {
        "case_text": "甲向乙交付十万元。",
        "question": "乙是否应返还借款？",
        "provided_knowledge": ["借款人应按期返还借款。"],
        "reasoning_artifact": '{"conclusions":[{"content":"乙应返还借款"}]}',
    }
    assert "CoT" not in prompt
    assert "Generic State" not in prompt
    assert "Legal State" not in prompt
    assert "只输出严格 JSON" in prompt
    assert "不得补充推理记录中没有出现的法律依据或结论" in prompt


def test_build_prompt_is_deterministic() -> None:
    arguments = (
        "案件文本",
        "原始问题",
        ("知识一", "知识二"),
        "推理记录",
    )

    assert build_final_answer_prompt(*arguments) == build_final_answer_prompt(
        *arguments
    )


def test_legal_state_json_is_passed_as_an_ordinary_artifact_string() -> None:
    artifact = json.dumps(
        {
            "issues": [],
            "facts": [],
            "knowledge": [],
            "relations": [],
            "conclusions": [],
        },
        ensure_ascii=False,
        sort_keys=True,
    )

    prompt = build_final_answer_prompt("案件", "问题", (), artifact)
    input_data = json.loads(prompt.split("输入数据：\n", maxsplit=1)[1])

    assert input_data["reasoning_artifact"] == artifact
    assert isinstance(input_data["reasoning_artifact"], str)


def test_generate_final_answer_calls_model_once_and_records_result() -> None:
    raw_output = '{"answer":"乙应返还借款本金。"}'
    client = ScriptedModelClient([raw_output])

    result = generate_final_answer(
        case_text="案件",
        question="问题",
        provided_knowledge=("知识",),
        reasoning_artifact="推理记录",
        model_client=client,  # type: ignore[arg-type]
    )

    assert len(client.prompts) == 1
    assert client.prompts[0] == result.prompt
    assert result.raw_model_output == raw_output
    assert result.parsed_answer == FinalAnswer(answer="乙应返还借款本金。")
    assert result.model == "scripted-final-answer-model"
    assert result.input_tokens == 201
    assert result.output_tokens == 21
    assert result.latency_seconds == 0.1


def test_model_call_failure_is_not_retried() -> None:
    model_error = RuntimeError("provider unavailable")
    client = ScriptedModelClient([model_error])

    with pytest.raises(FinalAnswerError) as caught:
        generate_final_answer(
            case_text="案件",
            question="问题",
            provided_knowledge=(),
            reasoning_artifact="推理记录",
            model_client=client,  # type: ignore[arg-type]
        )

    error = caught.value
    assert len(client.prompts) == 1
    assert error.stage == "model_call"
    assert error.model_call is None
    assert error.__cause__ is model_error


def test_parse_failure_is_not_retried_or_repaired_and_preserves_raw_output() -> None:
    raw_output = "```json\n{\"answer\":\"答案\"}\n```"
    client = ScriptedModelClient([raw_output])

    with pytest.raises(FinalAnswerError) as caught:
        generate_final_answer(
            case_text="案件",
            question="问题",
            provided_knowledge=(),
            reasoning_artifact="推理记录",
            model_client=client,  # type: ignore[arg-type]
        )

    error = caught.value
    assert len(client.prompts) == 1
    assert error.stage == "parse_final_answer"
    assert error.model_call is not None
    assert error.model_call.raw_text == raw_output
    assert error.__cause__ is not None
