import json

import pytest
from pydantic import ValidationError

from legal_state.actions import ActionName
from legal_state.prompts import build_action_prompt
from legal_state.schemas import Fact, Issue, LegalState


@pytest.fixture
def state() -> LegalState:
    return LegalState(
        issues=[Issue(id="I1", question="乙是否应返还本金")],
        facts=[Fact(id="F1", content="甲已交付借款", source="case")],
    )


def test_build_action_prompt_contains_inputs_and_only_allowed_formats(
    state: LegalState,
) -> None:
    prompt = build_action_prompt(
        case_text="甲向乙交付十万元。",
        question="乙是否应返还借款？",
        state=state,
        allowed_operations=[ActionName.BIND_FACT, "COMMIT"],
    )

    input_data = json.loads(prompt.split("输入数据：\n", maxsplit=1)[1])
    assert input_data == {
        "case_text": "甲向乙交付十万元。",
        "question": "乙是否应返还借款？",
        "legal_state": state.model_dump(mode="json"),
    }
    assert '"operation":"BIND_FACT"' in prompt
    assert '"operation":"COMMIT"' in prompt
    assert '"operation":"EXPAND_ISSUE"' not in prompt
    assert '"operation":"RESOLVE"' not in prompt
    assert "只输出一个合法 JSON 对象" in prompt
    assert "不要输出 Markdown、解释或思维链" in prompt
    assert "不得增加额外字段" in prompt


def test_build_action_prompt_deduplicates_allowed_operations(state: LegalState) -> None:
    prompt = build_action_prompt("案件", "问题", state, ["RESOLVE", "RESOLVE"])
    assert prompt.count('"operation":"RESOLVE"') == 1


@pytest.mark.parametrize("allowed", [[], ["SEARCH"]])
def test_build_action_prompt_rejects_invalid_allowed_operations(
    state: LegalState, allowed: list[str]
) -> None:
    with pytest.raises(ValueError):
        build_action_prompt("案件", "问题", state, allowed)


def test_build_action_prompt_revalidates_mutated_state(state: LegalState) -> None:
    state.facts[0].id = "I1"
    with pytest.raises(ValidationError, match="Duplicate object ID"):
        build_action_prompt("案件", "问题", state, ["BIND_FACT"])
