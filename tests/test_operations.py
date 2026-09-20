"""测试状态操作、非法状态转移，以及不同快照互不影响。"""

from collections.abc import Callable

import pytest
from pydantic import TypeAdapter, ValidationError

from legal_state.operations import (
    InvalidTransitionError,
    IssueNotFoundError,
    StateOperationError,
    bind_fact,
    commit,
    expand_issue,
    resolve,
)
from legal_state.schemas import (
    Conclusion,
    Fact,
    Issue,
    IssueStatus,
    Knowledge,
    LegalState,
    StateTransition,
)


@pytest.fixture
def initial_state() -> LegalState:
    return LegalState(
        issues=[
            Issue(id="I1", question="乙是否应返还本金"),
            Issue(id="I2", question="是否需要支付利息"),
        ],
        facts=[
            Fact(id="F1", content="甲交付借款本金", source="case"),
            Fact(id="F2", content="期限届满乙未偿还", source="case"),
        ],
        knowledge=[
            Knowledge(
                id="K1", content="借款人应按约定期限还款", source="provided_rule"
            )
        ],
    )


@pytest.fixture
def reasoning_state(initial_state: LegalState) -> LegalState:
    return bind_fact(initial_state, "I1", ["F1"])


@pytest.fixture
def ready_state(reasoning_state: LegalState) -> LegalState:
    return commit(reasoning_state, "I1", "乙应返还本金", ["F1", "K1"])


def test_expand_from_empty_state() -> None:
    initial = LegalState()
    expanded = expand_issue(initial, "借款合同是否成立")
    assert expanded.issues[0].id == "I1"
    assert expanded.issues[0].status is IssueStatus.OPEN
    assert expanded.issues[0].parent_issue is None
    assert initial.issues == []


def test_expand_child_issue(initial_state: LegalState) -> None:
    expanded = expand_issue(initial_state, "借款合同是否有效", parent_issue="I1")
    assert expanded.issues[-1].id == "I3"
    assert expanded.issues[-1].parent_issue == "I1"
    assert expanded.issues[-1].question == "借款合同是否有效"
    assert expanded.issues[0].status is IssueStatus.OPEN


def test_expand_does_not_reopen_resolved_parent(ready_state: LegalState) -> None:
    resolved = resolve(ready_state, "I1")
    expanded = expand_issue(resolved, "新发现的子争点", parent_issue="I1")
    assert expanded.issues[0].status is IssueStatus.RESOLVED
    assert expanded.issues[-1].status is IssueStatus.OPEN


def test_expand_missing_parent(initial_state: LegalState) -> None:
    with pytest.raises(IssueNotFoundError, match="EXPAND_ISSUE.*missing"):
        expand_issue(initial_state, "子争点", parent_issue="missing")


def test_expand_empty_question(initial_state: LegalState) -> None:
    with pytest.raises(ValidationError, match="question"):
        expand_issue(initial_state, "")


def test_bind_moves_open_to_reasoning(initial_state: LegalState) -> None:
    fact_ids = ["F1", "F2"]
    bound = bind_fact(initial_state, "I1", fact_ids)
    assert bound.issues[0].status is IssueStatus.REASONING
    assert bound.issues[1].status is IssueStatus.OPEN
    assert bound.relations[0].id == "A1"
    assert bound.relations[0].issue_id == "I1"
    assert bound.relations[0].fact_ids == fact_ids
    assert bound.relations[0].knowledge_ids == []
    assert bound.relations[0].description
    fact_ids.append("missing")
    assert bound.relations[0].fact_ids == ["F1", "F2"]


def test_bind_during_reasoning_preserves_previous_relation(
    reasoning_state: LegalState,
) -> None:
    bound = bind_fact(reasoning_state, "I1", ["F2"])
    assert bound.issues[0].status is IssueStatus.REASONING
    assert [item.id for item in bound.relations] == ["A1", "A2"]
    assert [item.fact_ids for item in bound.relations] == [["F1"], ["F2"]]


def test_bind_requires_fact_ids(initial_state: LegalState) -> None:
    with pytest.raises(StateOperationError, match="BIND_FACT.*at least one"):
        bind_fact(initial_state, "I1", [])


@pytest.mark.parametrize("target", ["missing", "K1", "I1"])
def test_bind_invalid_fact_reference(initial_state: LegalState, target: str) -> None:
    before = initial_state.model_dump()
    with pytest.raises(ValidationError, match="fact_ids"):
        bind_fact(initial_state, "I1", [target])
    assert initial_state.model_dump() == before


def test_commit_creates_conclusion(reasoning_state: LegalState) -> None:
    support = ["F1", "K1"]
    committed = commit(reasoning_state, "I1", "乙应返还本金", support)
    conclusion = committed.conclusions[0]
    assert conclusion.id == "C1"
    assert conclusion.issue_id == "I1"
    assert conclusion.content == "乙应返还本金"
    assert conclusion.support == support
    assert committed.issues[0].status is IssueStatus.REASONING
    support.append("missing")
    assert conclusion.support == ["F1", "K1"]


def test_commit_preserves_previous_conclusions(ready_state: LegalState) -> None:
    committed = commit(ready_state, "I1", "进一步的阶段结论", ["F2", "K1"])
    assert [item.id for item in committed.conclusions] == ["C1", "C2"]
    assert committed.conclusions[0].content == "乙应返还本金"


def test_commit_allows_empty_support(reasoning_state: LegalState) -> None:
    committed = commit(reasoning_state, "I1", "暂定结论", [])
    assert committed.conclusions[0].support == []


def test_commit_empty_conclusion(reasoning_state: LegalState) -> None:
    with pytest.raises(ValidationError, match="content"):
        commit(reasoning_state, "I1", "", ["F1"])


@pytest.mark.parametrize("target", ["missing", "I1", "A1"])
def test_commit_invalid_support(ready_state: LegalState, target: str) -> None:
    before = ready_state.model_dump()
    with pytest.raises(ValidationError, match="support"):
        commit(ready_state, "I1", "阶段结论", [target])
    assert ready_state.model_dump() == before


def test_commit_allows_previous_conclusion_as_support(
    reasoning_state: LegalState,
) -> None:
    first = commit(reasoning_state, "I1", "乙应返还本金", ["F1", "K1"])
    before = first.model_dump()

    second = commit(first, "I1", "乙还应承担迟延责任", ["C1", "F2", "K1"])

    assert second is not first
    assert second.conclusions[-1].id == "C2"
    assert second.conclusions[-1].support == ["C1", "F2", "K1"]
    assert first.model_dump() == before
    assert LegalState.model_validate(second.model_dump()) == second


def test_commit_requires_reasoning(initial_state: LegalState) -> None:
    with pytest.raises(InvalidTransitionError, match="COMMIT.*open.*reasoning"):
        commit(initial_state, "I1", "阶段结论", ["F1"])


def test_resolve_with_conclusion(ready_state: LegalState) -> None:
    resolved = resolve(ready_state, "I1")
    assert resolved.issues[0].status is IssueStatus.RESOLVED
    assert resolved.issues[1].status is IssueStatus.OPEN
    assert resolved.conclusions == ready_state.conclusions
    assert resolved.relations == ready_state.relations


def test_resolve_requires_reasoning(initial_state: LegalState) -> None:
    with pytest.raises(InvalidTransitionError, match="RESOLVE.*open.*reasoning"):
        resolve(initial_state, "I1")


def test_resolve_requires_conclusion(reasoning_state: LegalState) -> None:
    with pytest.raises(InvalidTransitionError, match="RESOLVE.*I1.*no conclusion"):
        resolve(reasoning_state, "I1")


def test_resolve_requires_conclusion_for_target_issue(
    reasoning_state: LegalState,
) -> None:
    candidate = reasoning_state.model_dump()
    candidate["conclusions"] = [
        Conclusion(id="C1", issue_id="I2", content="另一争点的结论", support=[])
    ]
    state = LegalState.model_validate(candidate)
    with pytest.raises(InvalidTransitionError, match="I1.*no conclusion"):
        resolve(state, "I1")


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda state: bind_fact(state, "I1", ["F1"]), id="BIND_FACT"),
        pytest.param(
            lambda state: commit(state, "I1", "阶段结论", ["F1"]), id="COMMIT"
        ),
        pytest.param(lambda state: resolve(state, "I1"), id="RESOLVE"),
    ],
)
def test_resolved_issue_rejects_operations(
    ready_state: LegalState, operation: Callable[[LegalState], LegalState]
) -> None:
    resolved = resolve(ready_state, "I1")
    before = resolved.model_dump()
    with pytest.raises(InvalidTransitionError, match="I1.*resolved"):
        operation(resolved)
    assert resolved.model_dump() == before


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(
            lambda state: expand_issue(state, "子争点", parent_issue="missing"),
            id="EXPAND_ISSUE",
        ),
        pytest.param(
            lambda state: bind_fact(state, "missing", ["F1"]), id="BIND_FACT"
        ),
        pytest.param(
            lambda state: commit(state, "missing", "结论", ["F1"]), id="COMMIT"
        ),
        pytest.param(lambda state: resolve(state, "missing"), id="RESOLVE"),
    ],
)
def test_unknown_issue(
    ready_state: LegalState, operation: Callable[[LegalState], LegalState]
) -> None:
    with pytest.raises(IssueNotFoundError, match="missing"):
        operation(ready_state)


@pytest.mark.parametrize(
    ("operation", "prefix", "collection"),
    [
        pytest.param(
            lambda state: expand_issue(state, "新争点"),
            "I",
            "issues",
            id="EXPAND_ISSUE",
        ),
        pytest.param(
            lambda state: bind_fact(state, "I1", ["F2"]),
            "A",
            "relations",
            id="BIND_FACT",
        ),
        pytest.param(
            lambda state: commit(state, "I1", "新结论", ["F1"]),
            "C",
            "conclusions",
            id="COMMIT",
        ),
    ],
)
def test_ids_are_deterministic_and_skip_global_collisions(
    ready_state: LegalState,
    operation: Callable[[LegalState], LegalState],
    prefix: str,
    collection: str,
) -> None:
    candidate = ready_state.model_dump()
    candidate["facts"].extend(
        [
            Fact(id=f"{prefix}3", content="占用较小编号", source="case"),
            Fact(id=f"{prefix}10", content="占用最大编号", source="case"),
        ]
    )
    state = LegalState.model_validate(candidate)
    first = operation(state)
    repeated = operation(state)
    assert first == repeated
    assert getattr(first, collection)[-1].id == f"{prefix}11"
    assert getattr(operation(first), collection)[-1].id == f"{prefix}12"


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda state: expand_issue(state, "新争点"), id="EXPAND_ISSUE"),
        pytest.param(lambda state: bind_fact(state, "I1", ["F2"]), id="BIND_FACT"),
        pytest.param(
            lambda state: commit(state, "I1", "新结论", ["F1"]), id="COMMIT"
        ),
        pytest.param(lambda state: resolve(state, "I1"), id="RESOLVE"),
    ],
)
def test_operations_return_independent_validated_snapshots(
    ready_state: LegalState, operation: Callable[[LegalState], LegalState]
) -> None:
    before = ready_state.model_dump()
    result = operation(ready_state)
    assert result is not ready_state
    assert ready_state.model_dump() == before
    assert LegalState.model_validate_json(result.model_dump_json()) == result
    for field in LegalState.model_fields:
        original_items = getattr(ready_state, field)
        result_items = getattr(result, field)
        assert original_items is not result_items
        for original, updated in zip(original_items, result_items):
            assert original is not updated
    result.facts[0].content = "修改新状态的事实"
    result.relations[0].fact_ids.append("missing")
    result.conclusions[0].support.clear()
    assert ready_state.model_dump() == before


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda state: expand_issue(state, "新争点"), id="EXPAND_ISSUE"),
        pytest.param(lambda state: bind_fact(state, "I1", ["F2"]), id="BIND_FACT"),
        pytest.param(
            lambda state: commit(state, "I1", "新结论", ["F1"]), id="COMMIT"
        ),
        pytest.param(lambda state: resolve(state, "I1"), id="RESOLVE"),
    ],
)
def test_operations_revalidate_mutated_input(
    ready_state: LegalState, operation: Callable[[LegalState], LegalState]
) -> None:
    ready_state.relations[0].fact_ids.append("missing")
    before = ready_state.model_dump()
    with pytest.raises(ValidationError, match="missing"):
        operation(ready_state)
    assert ready_state.model_dump() == before


def test_complete_state_trace(initial_state: LegalState) -> None:
    r0 = initial_state
    r1 = bind_fact(r0, "I1", ["F1", "F2"])
    r2 = commit(r1, "I1", "乙应返还本金", ["F1", "F2", "K1"])
    r3 = resolve(r2, "I1")
    assert [state.issues[0].status for state in [r0, r1, r2, r3]] == [
        IssueStatus.OPEN,
        IssueStatus.REASONING,
        IssueStatus.REASONING,
        IssueStatus.RESOLVED,
    ]
    assert [len(state.relations) for state in [r0, r1, r2, r3]] == [0, 1, 1, 1]
    assert [len(state.conclusions) for state in [r0, r1, r2, r3]] == [0, 0, 1, 1]

    trajectory = [
        StateTransition(operation="BIND_FACT", before=r0, after=r1),
        StateTransition(operation="COMMIT", before=r1, after=r2),
        StateTransition(operation="RESOLVE", before=r2, after=r3),
    ]
    assert trajectory[0].after == trajectory[1].before
    assert trajectory[1].after == trajectory[2].before
    adapter = TypeAdapter(list[StateTransition])
    restored = adapter.validate_json(adapter.dump_json(trajectory))
    assert restored == trajectory
    assert restored[0].before == r0
    assert restored[-1].after == r3
