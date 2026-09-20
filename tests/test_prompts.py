import json

import pytest
from pydantic import ValidationError

from legal_state.actions import ActionName
from legal_state.prompts import build_action_prompt
from legal_state.schemas import Conclusion, Fact, Issue, IssueStatus, LegalState


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
    assert '"operation":"STOP"' not in prompt
    assert "只输出一个合法 JSON 对象" in prompt
    assert "不要输出 Markdown、解释或思维链" in prompt
    assert "不得增加额外字段" in prompt
    assert "事实、知识或阶段性结论" in prompt


def test_build_action_prompt_includes_stop_format_and_guidance(
    state: LegalState,
) -> None:
    prompt = build_action_prompt("案件", "问题", state, [ActionName.STOP])
    assert '"operation":"STOP"' in prompt
    assert "当你认为当前 LegalState 已经无需继续更新时" in prompt
    assert "否则应选择其他允许的操作继续更新状态" in prompt
    assert "STOP 仅表示结束状态构建，不生成也不等同于最终答案" in prompt


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


def _target_section(prompt: str) -> str:
    return prompt.split(
        "当前操作可使用的争点目标：\n", maxsplit=1
    )[1].split("\n\n允许的 JSON 格式：", maxsplit=1)[0]


def test_build_action_prompt_lists_targets_for_each_operation() -> None:
    state = LegalState(
        issues=[
            Issue(id="I1", question="争点一", status=IssueStatus.REASONING),
            Issue(id="I2", question="争点二"),
            Issue(id="I3", question="争点三", status=IssueStatus.RESOLVED),
        ],
        facts=[Fact(id="F1", content="事实", source="case")],
        conclusions=[
            Conclusion(id="C1", issue_id="I1", content="结论", support=["F1"])
        ],
    )

    targets = _target_section(
        build_action_prompt(
            "案件",
            "问题",
            state,
            [
                "EXPAND_ISSUE",
                "BIND_FACT",
                "COMMIT",
                "RESOLVE",
                "STOP",
            ],
        )
    )
    target_lines = targets.splitlines()

    assert "- BIND_FACT: I1, I2" in target_lines
    assert "- COMMIT: I1" in target_lines
    assert "- RESOLVE: I1" in target_lines
    assert "- EXPAND_ISSUE: 可新增争点；parent_issue 可为 null 或已有 issue ID: I1, I2, I3" in target_lines
    assert "- STOP: 不需要 issue_id" in target_lines
    assert "I3" not in next(line for line in target_lines if line.startswith("- BIND_FACT:"))
    assert "I3" not in next(line for line in target_lines if line.startswith("- COMMIT:"))
    assert "I3" not in next(line for line in target_lines if line.startswith("- RESOLVE:"))


def test_build_action_prompt_excludes_reasoning_issue_without_conclusion_from_resolve() -> None:
    state = LegalState(
        issues=[Issue(id="I1", question="争点", status=IssueStatus.REASONING)],
        facts=[Fact(id="F1", content="事实", source="case")],
    )

    targets = _target_section(
        build_action_prompt("案件", "问题", state, ["COMMIT", "RESOLVE"])
    )

    assert "- COMMIT: I1" in targets
    assert "- RESOLVE: 无合法目标争点" in targets


def test_build_action_prompt_lists_only_allowed_operation_targets(
    state: LegalState,
) -> None:
    targets = _target_section(
        build_action_prompt("案件", "问题", state, ["BIND_FACT", "COMMIT"])
    )

    assert targets.splitlines() == ["- BIND_FACT: I1", "- COMMIT: 无合法目标争点"]
    assert "RESOLVE" not in targets
    assert "STOP" not in targets
    assert "EXPAND_ISSUE" not in targets


def test_build_action_prompt_shows_no_bind_target_without_facts() -> None:
    state = LegalState(issues=[Issue(id="I1", question="争点")])

    targets = _target_section(
        build_action_prompt("案件", "问题", state, ["BIND_FACT"])
    )

    assert targets == "- BIND_FACT: 无合法目标争点"
