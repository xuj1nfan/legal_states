from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from legal_state.actions import (
    ActionName,
    BindFactAction,
    CommitAction,
    ExpandIssueAction,
    ResolveAction,
    StopAction,
)
from legal_state.model import ModelCallResult
from legal_state.operations import IssueNotFoundError
from legal_state.runner import (
    RunnerStepError,
    TerminationReason,
    determine_allowed_operations,
    run_legal_state,
)
from legal_state.schemas import (
    Conclusion,
    Fact,
    Issue,
    IssueStatus,
    Knowledge,
    LegalState,
)


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
            model="scripted-model",
            input_tokens=100 + call_number,
            output_tokens=10 + call_number,
            latency_seconds=call_number / 10,
        )


def test_determine_allowed_operations_for_empty_state() -> None:
    assert determine_allowed_operations(LegalState()) == (
        ActionName.EXPAND_ISSUE,
        ActionName.STOP,
    )


def test_determine_allowed_operations_for_open_issue_with_fact() -> None:
    state = LegalState(
        issues=[Issue(id="I1", question="争点")],
        facts=[Fact(id="F1", content="事实", source="case")],
    )

    assert determine_allowed_operations(state) == (
        ActionName.EXPAND_ISSUE,
        ActionName.BIND_FACT,
        ActionName.STOP,
    )


def test_determine_allowed_operations_for_reasoning_issue() -> None:
    state = LegalState(
        issues=[
            Issue(
                id="I1",
                question="争点",
                status=IssueStatus.REASONING,
            )
        ]
    )

    assert determine_allowed_operations(state) == (
        ActionName.EXPAND_ISSUE,
        ActionName.COMMIT,
        ActionName.STOP,
    )


def test_determine_allowed_operations_for_reasoning_issue_with_conclusion() -> None:
    state = LegalState(
        issues=[
            Issue(
                id="I1",
                question="争点",
                status=IssueStatus.REASONING,
            )
        ],
        facts=[Fact(id="F1", content="事实", source="case")],
        conclusions=[
            Conclusion(
                id="C1",
                issue_id="I1",
                content="结论",
                support=["F1"],
            )
        ],
    )

    assert determine_allowed_operations(state) == (
        ActionName.EXPAND_ISSUE,
        ActionName.BIND_FACT,
        ActionName.COMMIT,
        ActionName.RESOLVE,
        ActionName.STOP,
    )


def test_determine_allowed_operations_for_resolved_only_state() -> None:
    state = LegalState(
        issues=[
            Issue(
                id="I1",
                question="争点",
                status=IssueStatus.RESOLVED,
            )
        ]
    )

    assert determine_allowed_operations(state) == (
        ActionName.EXPAND_ISSUE,
        ActionName.STOP,
    )


def test_runner_records_complete_model_driven_trajectory() -> None:
    initial_state = LegalState(
        facts=[Fact(id="F1", content="甲已经交付借款", source="case")],
        knowledge=[
            Knowledge(
                id="K1",
                content="借款人应按期返还借款",
                source="provided_rule",
            )
        ],
    )
    original = initial_state.model_dump()
    responses = [
        '{"operation":"EXPAND_ISSUE","question":"乙是否应还款","parent_issue":null}',
        '{"operation":"BIND_FACT","issue_id":"I1","fact_ids":["F1"]}',
        (
            '{"operation":"COMMIT","issue_id":"I1",'
            '"conclusion":"乙应返还借款","support":["F1","K1"]}'
        ),
        '{"operation":"RESOLVE","issue_id":"I1"}',
        '{"operation":"STOP"}',
    ]
    client = ScriptedModelClient(responses)

    result = run_legal_state(
        case_text="甲向乙交付借款，乙到期未还。",
        question="乙是否应返还借款？",
        initial_state=initial_state,
        model_client=client,  # type: ignore[arg-type]
        max_steps=8,
    )

    assert result.case_text == "甲向乙交付借款，乙到期未还。"
    assert result.question == "乙是否应返还借款？"
    assert result.termination_reason is TerminationReason.STOP
    assert result.max_steps == 8
    assert len(result.steps) == 5
    assert [step.step_index for step in result.steps] == list(range(5))
    assert [type(step.parsed_action) for step in result.steps] == [
        ExpandIssueAction,
        BindFactAction,
        CommitAction,
        ResolveAction,
        StopAction,
    ]
    assert [step.raw_model_output for step in result.steps] == responses
    assert [step.model for step in result.steps] == ["scripted-model"] * 5
    assert [step.input_tokens for step in result.steps] == [101, 102, 103, 104, 105]
    assert [step.output_tokens for step in result.steps] == [11, 12, 13, 14, 15]
    assert [step.latency_seconds for step in result.steps] == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert all(step.prompt == prompt for step, prompt in zip(result.steps, client.prompts))
    assert result.final_state.issues[0].status is IssueStatus.RESOLVED
    assert result.steps[-1].before_state == result.steps[-1].after_state
    assert result.steps[-1].before_state is not result.steps[-1].after_state
    assert result.initial_state is not initial_state
    assert initial_state.model_dump() == original

    for previous, following in zip(result.steps, result.steps[1:]):
        assert previous.after_state == following.before_state
        assert previous.after_state is not following.before_state


def test_runner_returns_max_steps_with_complete_trace() -> None:
    client = ScriptedModelClient(
        [
            (
                '{"operation":"EXPAND_ISSUE","question":"争点一",'
                '"parent_issue":null}'
            ),
            (
                '{"operation":"EXPAND_ISSUE","question":"争点二",'
                '"parent_issue":null}'
            ),
        ]
    )

    result = run_legal_state(
        "案件",
        "问题",
        LegalState(),
        client,  # type: ignore[arg-type]
        max_steps=2,
    )

    assert result.termination_reason is TerminationReason.MAX_STEPS
    assert len(result.steps) == 2
    assert len(result.final_state.issues) == 2
    assert len(client.prompts) == 2


def test_stop_on_last_available_step_takes_precedence_over_max_steps() -> None:
    client = ScriptedModelClient(['{"operation":"STOP"}'])

    result = run_legal_state(
        "案件",
        "问题",
        LegalState(),
        client,  # type: ignore[arg-type]
        max_steps=1,
    )

    assert result.termination_reason is TerminationReason.STOP
    assert len(result.steps) == 1


def test_runner_rejects_non_positive_max_steps_before_model_call() -> None:
    client = ScriptedModelClient([])

    with pytest.raises(ValueError, match="max_steps"):
        run_legal_state(
            "案件",
            "问题",
            LegalState(),
            client,  # type: ignore[arg-type]
            max_steps=0,
        )

    assert client.prompts == []


def test_parse_failure_preserves_completed_steps_and_model_output() -> None:
    client = ScriptedModelClient(
        [
            (
                '{"operation":"EXPAND_ISSUE","question":"争点",'
                '"parent_issue":null}'
            ),
            "not json",
        ]
    )

    with pytest.raises(RunnerStepError) as caught:
        run_legal_state(
            "案件",
            "问题",
            LegalState(),
            client,  # type: ignore[arg-type]
            max_steps=3,
        )

    error = caught.value
    assert len(error.completed_steps) == 1
    assert error.failed_step.step_index == 1
    assert error.failed_step.stage == "parse_action"
    assert error.failed_step.raw_model_output == "not json"
    assert error.failed_step.parsed_action is None
    assert error.failed_step.model == "scripted-model"
    assert isinstance(error.__cause__, ValidationError)
    assert len(client.prompts) == 2


def test_disallowed_action_fails_before_executor() -> None:
    client = ScriptedModelClient(
        ['{"operation":"BIND_FACT","issue_id":"I1","fact_ids":["F1"]}']
    )

    with pytest.raises(RunnerStepError) as caught:
        run_legal_state(
            "案件",
            "问题",
            LegalState(),
            client,  # type: ignore[arg-type]
            max_steps=1,
        )

    failed = caught.value.failed_step
    assert failed.stage == "check_allowed_operation"
    assert isinstance(failed.parsed_action, BindFactAction)
    assert ActionName.BIND_FACT not in failed.allowed_operations
    assert isinstance(caught.value.__cause__, ValueError)


def test_executor_failure_preserves_parsed_action() -> None:
    state = LegalState(
        issues=[Issue(id="I1", question="争点")],
        facts=[Fact(id="F1", content="事实", source="case")],
    )
    client = ScriptedModelClient(
        ['{"operation":"BIND_FACT","issue_id":"missing","fact_ids":["F1"]}']
    )

    with pytest.raises(RunnerStepError) as caught:
        run_legal_state(
            "案件",
            "问题",
            state,
            client,  # type: ignore[arg-type]
            max_steps=1,
        )

    failed = caught.value.failed_step
    assert failed.stage == "apply_action"
    assert isinstance(failed.parsed_action, BindFactAction)
    assert failed.parsed_action.issue_id == "missing"
    assert isinstance(caught.value.__cause__, IssueNotFoundError)
    assert len(client.prompts) == 1


def test_model_failure_preserves_completed_steps_and_current_prompt() -> None:
    model_error = RuntimeError("provider unavailable")
    client = ScriptedModelClient(
        [
            (
                '{"operation":"EXPAND_ISSUE","question":"争点",'
                '"parent_issue":null}'
            ),
            model_error,
        ]
    )

    with pytest.raises(RunnerStepError) as caught:
        run_legal_state(
            "案件正文",
            "案件问题",
            LegalState(),
            client,  # type: ignore[arg-type]
            max_steps=3,
        )

    error = caught.value
    failed = error.failed_step
    assert len(error.completed_steps) == 1
    assert failed.step_index == 1
    assert failed.stage == "model_call"
    assert failed.before_state == error.completed_steps[0].after_state
    assert "案件正文" in failed.prompt
    assert "案件问题" in failed.prompt
    assert failed.raw_model_output is None
    assert failed.parsed_action is None
    assert failed.model is None
    assert failed.input_tokens is None
    assert failed.output_tokens is None
    assert failed.latency_seconds is None
    assert error.__cause__ is model_error
    assert len(client.prompts) == 2
