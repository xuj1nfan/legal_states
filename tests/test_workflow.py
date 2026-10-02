"""Sequential workflow phase gates and failure preservation."""

import json

import pytest

from legal_state.actions import ActionName
from legal_state.model import ModelCallResult
from legal_state.operations import bind_fact, commit, expand_issue, resolve
from legal_state.runner import (
    RunnerStepError,
    determine_allowed_operations,
    run_legal_state,
)
from legal_state.schemas import Fact, Issue, LegalState


class Client:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return ModelCallResult(
            raw_text=json.dumps(next(self.actions)),
            model="scripted",
            input_tokens=100,
            output_tokens=10,
            latency_seconds=0,
            finish_reason="stop",
        )


@pytest.fixture
def state():
    return LegalState(facts=[Fact(id="F1", content="材料", source="case")])


def test_sequential_phases_require_conclusion_and_resolution_before_stop(state):
    states = [state]
    states.append(expand_issue(states[-1], "争点"))
    states.append(bind_fact(states[-1], "I1", ["F1"]))
    states.append(commit(states[-1], "I1", "判断及理由", ["F1"]))
    states.append(resolve(states[-1], "I1"))
    assert [determine_allowed_operations(s, workflow="sequential") for s in states] == [
        (ActionName.EXPAND_ISSUE,),
        (ActionName.BIND_FACT,),
        (ActionName.COMMIT,),
        (ActionName.RESOLVE,),
        (ActionName.EXPAND_ISSUE, ActionName.STOP),
    ]
    assert ActionName.BIND_FACT in determine_allowed_operations(states[2])
    assert ActionName.COMMIT in determine_allowed_operations(states[3])


def test_sequential_runner_completes_and_keeps_full_immutable_trajectory(state):
    snapshot = state.model_dump()
    actions = [
        {"operation": "EXPAND_ISSUE", "question": "争点"},
        {"operation": "BIND_FACT", "issue_id": "I1", "fact_ids": ["F1"]},
        {
            "operation": "COMMIT",
            "issue_id": "I1",
            "conclusion": "判断及理由",
            "support": ["F1"],
        },
        {"operation": "RESOLVE", "issue_id": "I1"},
        {"operation": "STOP"},
    ]
    client = Client(actions)
    result = run_legal_state("材料", "问题", state, client, 24, workflow="sequential")
    assert result.termination_reason.value == "stop"
    assert len(result.steps) == 5
    assert result.final_state.issues[0].status.value == "resolved"
    assert not result.steps[1].before_state.relations
    assert not result.steps[2].before_state.conclusions
    assert state.model_dump() == snapshot


@pytest.mark.parametrize(
    "bad_action,phase",
    [
        ({"operation": "STOP"}, "empty"),
        ({"operation": "BIND_FACT", "issue_id": "I1", "fact_ids": ["F1"]}, "bound"),
        (
            {
                "operation": "COMMIT",
                "issue_id": "I1",
                "conclusion": "重复",
                "support": [],
            },
            "concluded",
        ),
    ],
)
def test_sequential_invalid_phase_actions_fail_without_state_change_or_retry(
    state, bad_action, phase
):
    if phase != "empty":
        state = bind_fact(expand_issue(state, "争点"), "I1", ["F1"])
    if phase == "concluded":
        state = commit(state, "I1", "已有结论", ["F1"])
    snapshot = state.model_dump()
    client = Client([bad_action])
    with pytest.raises(RunnerStepError) as caught:
        run_legal_state("材料", "问题", state, client, 24, workflow="sequential")
    assert caught.value.failed_step.stage == "check_allowed_operation"
    assert caught.value.failed_step.before_state.model_dump() == snapshot
    assert len(client.prompts) == 1
    assert state.model_dump() == snapshot


def test_sequential_focus_rejects_another_otherwise_valid_issue_target(state):
    state.issues = [
        Issue(id="I1", question="先处理"),
        Issue(id="I2", question="后处理"),
    ]
    client = Client([{"operation": "BIND_FACT", "issue_id": "I2", "fact_ids": ["F1"]}])
    with pytest.raises(RunnerStepError, match="expected issue target 'I1'") as caught:
        run_legal_state("材料", "问题", state, client, 24, workflow="sequential")
    assert caught.value.failed_step.before_state.relations == []
    target_section = (
        client.prompts[0].split("当前操作可使用的争点目标：\n")[1].split("\n\n")[0]
    )
    assert target_section == "- BIND_FACT: I1"


def test_sequential_requires_material_but_free_workflow_stays_available():
    state = LegalState(issues=[Issue(id="I1", question="争点")])
    with pytest.raises(ValueError, match="needs material"):
        determine_allowed_operations(state, workflow="sequential")
    assert ActionName.STOP in determine_allowed_operations(state)
