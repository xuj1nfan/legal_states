import pytest
from pydantic import ValidationError

from legal_state.actions import (
    ActionName,
    BindFactAction,
    CommitAction,
    ExpandIssueAction,
    ResolveAction,
    StopAction,
    parse_action_json,
)
from legal_state.executor import apply_action
from legal_state.operations import InvalidTransitionError, IssueNotFoundError
from legal_state.schemas import Fact, Issue, IssueStatus, Knowledge, LegalState


@pytest.fixture
def initial_state() -> LegalState:
    return LegalState(
        issues=[Issue(id="I1", question="乙是否应返还本金")],
        facts=[Fact(id="F1", content="甲已交付借款", source="case")],
        knowledge=[
            Knowledge(
                id="K1",
                content="借款人应当按约定返还借款",
                source="provided_rule",
            )
        ],
    )


def test_apply_expand_issue_creates_issue() -> None:
    state = LegalState()
    action = parse_action_json(
        '{"operation":"EXPAND_ISSUE","question":"合同是否有效",'
        '"parent_issue":null}'
    )

    result = apply_action(state, action)

    assert len(result.issues) == 1
    assert result.issues[0].question == "合同是否有效"
    assert result.issues[0].status is IssueStatus.OPEN


def test_apply_bind_fact_uses_existing_operation(initial_state: LegalState) -> None:
    action = BindFactAction(
        operation=ActionName.BIND_FACT,
        issue_id="I1",
        fact_ids=["F1"],
    )

    result = apply_action(initial_state, action)

    assert result.issues[0].status is IssueStatus.REASONING
    assert result.relations[0].issue_id == "I1"
    assert result.relations[0].fact_ids == ["F1"]


def test_apply_commit_creates_conclusion(initial_state: LegalState) -> None:
    reasoning_state = apply_action(
        initial_state,
        BindFactAction(
            operation=ActionName.BIND_FACT,
            issue_id="I1",
            fact_ids=["F1"],
        ),
    )
    action = CommitAction(
        operation=ActionName.COMMIT,
        issue_id="I1",
        conclusion="乙应返还本金",
        support=["F1", "K1"],
    )

    result = apply_action(reasoning_state, action)

    assert result.conclusions[0].issue_id == "I1"
    assert result.conclusions[0].content == "乙应返还本金"
    assert result.conclusions[0].support == ["F1", "K1"]


def test_apply_resolve_changes_issue_status(initial_state: LegalState) -> None:
    reasoning_state = apply_action(
        initial_state,
        BindFactAction(
            operation=ActionName.BIND_FACT,
            issue_id="I1",
            fact_ids=["F1"],
        ),
    )
    ready_state = apply_action(
        reasoning_state,
        CommitAction(
            operation=ActionName.COMMIT,
            issue_id="I1",
            conclusion="乙应返还本金",
            support=["F1", "K1"],
        ),
    )

    result = apply_action(
        ready_state,
        ResolveAction(operation=ActionName.RESOLVE, issue_id="I1"),
    )

    assert result.issues[0].status is IssueStatus.RESOLVED


def test_apply_stop_returns_equal_independent_snapshot(
    initial_state: LegalState,
) -> None:
    result = apply_action(
        initial_state,
        StopAction(operation=ActionName.STOP),
    )

    assert result == initial_state
    assert result is not initial_state
    for field in LegalState.model_fields:
        assert getattr(result, field) is not getattr(initial_state, field)


def test_apply_stop_allows_open_issues(initial_state: LegalState) -> None:
    result = apply_action(
        initial_state,
        StopAction(operation=ActionName.STOP),
    )
    assert result.issues[0].status is IssueStatus.OPEN


def test_apply_stop_revalidates_mutated_state(initial_state: LegalState) -> None:
    initial_state.facts[0].id = "I1"
    with pytest.raises(ValidationError, match="Duplicate object ID"):
        apply_action(initial_state, StopAction(operation=ActionName.STOP))


def test_apply_action_preserves_invalid_transition_error(
    initial_state: LegalState,
) -> None:
    action = CommitAction(
        operation=ActionName.COMMIT,
        issue_id="I1",
        conclusion="乙应返还本金",
        support=["F1"],
    )

    with pytest.raises(InvalidTransitionError, match="COMMIT.*open.*reasoning"):
        apply_action(initial_state, action)


def test_apply_action_preserves_invalid_reference_error(
    initial_state: LegalState,
) -> None:
    action = BindFactAction(
        operation=ActionName.BIND_FACT,
        issue_id="I1",
        fact_ids=["missing"],
    )

    with pytest.raises(ValidationError, match="missing"):
        apply_action(initial_state, action)


def test_apply_expand_preserves_missing_parent_error(
    initial_state: LegalState,
) -> None:
    action = ExpandIssueAction(
        operation=ActionName.EXPAND_ISSUE,
        question="子争点",
        parent_issue="missing",
    )

    with pytest.raises(IssueNotFoundError, match="EXPAND_ISSUE.*missing"):
        apply_action(initial_state, action)
