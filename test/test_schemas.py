"""测试法律状态的数据结构和快照完整性。"""

import json

import pytest
from pydantic import BaseModel, ValidationError

from legal_state.schemas import (
    Conclusion,
    Fact,
    Issue,
    IssueStatus,
    Knowledge,
    LegalState,
    Relation,
    StateTransition,
)


@pytest.fixture
def complete_state() -> LegalState:
    return LegalState(
        issues=[
            Issue(
                id="I1",
                question="乙是否负有返还借款本金的义务",
                status=IssueStatus.RESOLVED,
            )
        ],
        facts=[
            Fact(id="F1", content="甲向乙出借10万元", source="case"),
            Fact(id="F2", content="双方约定一年后偿还", source="case"),
            Fact(id="F3", content="期限届满乙未偿还", source="case"),
        ],
        knowledge=[
            Knowledge(
                id="K1",
                content="借款人应当按照约定期限返还借款",
                source="provided_rule",
            )
        ],
        relations=[
            Relation(
                id="A1",
                issue_id="I1",
                fact_ids=["F1", "F2", "F3"],
                knowledge_ids=["K1"],
                description="案件事实结合给定规则支持返还借款本金的义务",
            )
        ],
        conclusions=[
            Conclusion(
                id="C1",
                issue_id="I1",
                content="乙负有返还借款本金的义务",
                support=["F1", "F3", "K1"],
            )
        ],
    )


def validate_update(state: LegalState, **updates: object) -> LegalState:
    return LegalState.model_validate({**state.model_dump(), **updates})


def test_empty_state() -> None:
    assert LegalState().model_dump() == {
        "issues": [],
        "facts": [],
        "knowledge": [],
        "relations": [],
        "conclusions": [],
    }


def test_default_lists_are_independent() -> None:
    first = LegalState()
    second = LegalState()
    for field in LegalState.model_fields:
        assert getattr(first, field) is not getattr(second, field)
    first.issues.append(Issue(id="I1", question="借款合同是否成立"))
    assert second.issues == []


def test_issue_defaults() -> None:
    issue = Issue(id="I1", question="借款合同是否成立")
    assert issue.status is IssueStatus.OPEN
    assert issue.parent_issue is None


@pytest.mark.parametrize("status", ["open", "reasoning", "resolved"])
def test_valid_status(status: str) -> None:
    issue = Issue.model_validate({"id": "I1", "question": "争点", "status": status})
    assert issue.status is IssueStatus(status)
    assert issue.model_dump(mode="json")["status"] == status


def test_invalid_status() -> None:
    with pytest.raises(ValidationError, match="status"):
        Issue.model_validate({"id": "I1", "question": "争点", "status": "closed"})


def test_initial_borrowing_state(complete_state: LegalState) -> None:
    state = LegalState(
        issues=[Issue(id="I1", question="乙是否负有返还借款本金的义务")],
        facts=complete_state.facts,
    )
    assert state.issues[0].status is IssueStatus.OPEN
    assert len(state.facts) == 3
    assert state.knowledge == state.relations == state.conclusions == []


def test_complete_borrowing_state(complete_state: LegalState) -> None:
    assert LegalState.model_validate(complete_state.model_dump()) == complete_state
    assert complete_state.conclusions[0].support == ["F1", "F3", "K1"]


@pytest.mark.parametrize("model", [Issue, Fact, Knowledge, Relation, Conclusion])
def test_missing_required_fields(model: type[BaseModel]) -> None:
    with pytest.raises(ValidationError):
        model.model_validate({"id": "test"})


@pytest.mark.parametrize(
    "collection", ["issues", "facts", "knowledge", "relations", "conclusions"]
)
def test_unknown_nested_field(complete_state: LegalState, collection: str) -> None:
    data = complete_state.model_dump()
    data[collection][0]["unexpected"] = "value"
    with pytest.raises(ValidationError, match="unexpected"):
        LegalState.model_validate(data)


def test_unknown_top_level_field() -> None:
    with pytest.raises(ValidationError, match="unexpected"):
        LegalState.model_validate({"unexpected": []})


@pytest.mark.parametrize(
    "collection", ["issues", "facts", "knowledge", "relations", "conclusions"]
)
def test_empty_object_id(complete_state: LegalState, collection: str) -> None:
    data = complete_state.model_dump()
    data[collection][0]["id"] = ""
    with pytest.raises(ValidationError, match="id"):
        LegalState.model_validate(data)


@pytest.mark.parametrize(
    ("collection", "field"),
    [
        ("issues", "question"),
        ("facts", "content"),
        ("facts", "source"),
        ("knowledge", "content"),
        ("knowledge", "source"),
        ("relations", "description"),
        ("conclusions", "content"),
    ],
)
def test_empty_text_field(
    complete_state: LegalState, collection: str, field: str
) -> None:
    data = complete_state.model_dump()
    data[collection][0][field] = ""
    with pytest.raises(ValidationError) as error:
        LegalState.model_validate(data)
    assert error.value.errors()[0]["loc"] == (collection, 0, field)
    assert error.value.errors()[0]["type"] == "string_too_short"


def test_custom_ids_and_sources() -> None:
    state = LegalState(
        issues=[Issue(id="borrower-duty", question="是否应偿还借款")],
        facts=[Fact(id="loan-delivery", content="已交付借款", source="case_annotation")],
        knowledge=[
            Knowledge(
                id="repayment-rule", content="应按期还款", source="manual_reference"
            )
        ],
        conclusions=[
            Conclusion(
                id="repayment-duty",
                issue_id="borrower-duty",
                content="应当还款",
                support=["loan-delivery", "repayment-rule"],
            )
        ],
    )
    assert state.knowledge[0].source == "manual_reference"


@pytest.mark.parametrize(
    "collection", ["issues", "facts", "knowledge", "relations", "conclusions"]
)
def test_duplicate_id_in_collection(complete_state: LegalState, collection: str) -> None:
    data = complete_state.model_dump()
    data[collection].append(data[collection][0].copy())
    with pytest.raises(ValidationError, match="Duplicate object ID"):
        LegalState.model_validate(data)


def test_duplicate_id_across_collections() -> None:
    with pytest.raises(ValidationError, match="Duplicate object ID 'shared'"):
        LegalState(
            issues=[Issue(id="shared", question="争点")],
            facts=[Fact(id="shared", content="事实", source="case")],
        )


def test_parent_may_appear_later() -> None:
    state = LegalState(
        issues=[
            Issue(id="I2", question="借款合同是否有效", parent_issue="I1"),
            Issue(id="I1", question="乙是否应返还本金"),
        ]
    )
    assert state.issues[0].parent_issue == "I1"


def test_missing_parent_issue() -> None:
    with pytest.raises(
        ValidationError, match="Object 'I2' field 'parent_issue'.*'missing'"
    ):
        LegalState(issues=[Issue(id="I2", question="子争点", parent_issue="missing")])


def test_parent_cannot_reference_itself() -> None:
    with pytest.raises(ValidationError, match="'I1'.*'parent_issue'.*itself"):
        LegalState(issues=[Issue(id="I1", question="争点", parent_issue="I1")])


def test_two_node_parent_cycle() -> None:
    with pytest.raises(
        ValidationError, match="parent_issue cycle detected: I1 -> I2 -> I1"
    ):
        LegalState(
            issues=[
                Issue(id="I1", question="争点一", parent_issue="I2"),
                Issue(id="I2", question="争点二", parent_issue="I1"),
            ]
        )


def test_three_node_parent_cycle() -> None:
    with pytest.raises(
        ValidationError, match="parent_issue cycle detected: I1 -> I2 -> I3 -> I1"
    ):
        LegalState(
            issues=[
                Issue(id="I1", question="争点一", parent_issue="I2"),
                Issue(id="I2", question="争点二", parent_issue="I3"),
                Issue(id="I3", question="争点三", parent_issue="I1"),
            ]
        )


def test_parent_chain_leading_into_cycle() -> None:
    with pytest.raises(
        ValidationError, match="parent_issue cycle detected: I1 -> I2 -> I1"
    ):
        LegalState(
            issues=[
                Issue(id="I0", question="循环外的子争点", parent_issue="I1"),
                Issue(id="I1", question="争点一", parent_issue="I2"),
                Issue(id="I2", question="争点二", parent_issue="I1"),
            ]
        )


def test_acyclic_hierarchy_with_shared_parents_and_multiple_roots() -> None:
    state = LegalState(
        issues=[
            Issue(id="I3", question="三级争点", parent_issue="I2"),
            Issue(id="I4", question="另一个子争点", parent_issue="I1"),
            Issue(id="I2", question="二级争点", parent_issue="I1"),
            Issue(id="I1", question="根争点"),
            Issue(id="I5", question="独立的根争点"),
        ]
    )
    assert len(state.issues) == 5


@pytest.mark.parametrize(
    ("field", "target", "error_field"),
    [
        ("issue_id", "missing", "issue_id"),
        ("issue_id", "F1", "issue_id"),
        ("fact_ids", "missing", "fact_ids[0]"),
        ("fact_ids", "K1", "fact_ids[0]"),
        ("knowledge_ids", "missing", "knowledge_ids[0]"),
        ("knowledge_ids", "F1", "knowledge_ids[0]"),
    ],
)
def test_invalid_relation_reference(
    complete_state: LegalState, field: str, target: str, error_field: str
) -> None:
    data = complete_state.relations[0].model_dump()
    data[field] = target if field == "issue_id" else [target]
    relation = Relation.model_validate(data)
    with pytest.raises(ValidationError) as error:
        validate_update(complete_state, relations=[relation])
    message = str(error.value)
    assert "Object 'A1'" in message
    assert repr(error_field) in message
    assert repr(target) in message


@pytest.mark.parametrize("target", ["missing", "F1"])
def test_invalid_conclusion_issue(complete_state: LegalState, target: str) -> None:
    conclusion = Conclusion(
        id="C1", issue_id=target, content="阶段结论", support=["F1"]
    )
    with pytest.raises(ValidationError, match="Object 'C1' field 'issue_id'"):
        validate_update(complete_state, conclusions=[conclusion])


@pytest.mark.parametrize("target", ["missing", "I1", "A1", "C1", "C2"])
def test_invalid_conclusion_support(complete_state: LegalState, target: str) -> None:
    conclusions = [
        Conclusion(id="C1", issue_id="I1", content="阶段结论", support=[target]),
        Conclusion(id="C2", issue_id="I1", content="其他结论", support=[]),
    ]
    with pytest.raises(ValidationError) as error:
        validate_update(complete_state, conclusions=conclusions)
    message = str(error.value)
    assert "Object 'C1' field 'support[0]'" in message
    assert repr(target) in message


def test_empty_references_and_unsupported_conclusion() -> None:
    state = LegalState(
        issues=[Issue(id="I1", question="尚未解决的争点")],
        relations=[
            Relation(
                id="A1",
                issue_id="I1",
                fact_ids=[],
                knowledge_ids=[],
                description="初步分析",
            )
        ],
        conclusions=[Conclusion(id="C1", issue_id="I1", content="暂定结论", support=[])],
    )
    assert state.conclusions[0].support == []
    assert state.issues[0].status is IssueStatus.OPEN


def test_resolved_issue_without_conclusion() -> None:
    state = LegalState(
        issues=[Issue(id="I1", question="争点", status=IssueStatus.RESOLVED)]
    )
    assert state.conclusions == []


def test_multiple_integrity_errors_are_reported() -> None:
    with pytest.raises(ValidationError) as error:
        LegalState(
            relations=[
                Relation(
                    id="A1",
                    issue_id="I-missing",
                    fact_ids=["F-missing"],
                    knowledge_ids=["K-missing"],
                    description="分析",
                )
            ]
        )
    for target in ["I-missing", "F-missing", "K-missing"]:
        assert repr(target) in str(error.value)


def test_json_round_trip(complete_state: LegalState) -> None:
    serialized = complete_state.model_dump_json()
    assert LegalState.model_validate_json(serialized) == complete_state
    payload = json.loads(serialized)
    assert payload["issues"][0]["status"] == "resolved"
    assert payload["facts"][0]["content"] == "甲向乙出借10万元"
    assert set(payload) == {"issues", "facts", "knowledge", "relations", "conclusions"}


def test_json_parsing_rejects_invalid_reference(complete_state: LegalState) -> None:
    payload = complete_state.model_dump(mode="json")
    payload["conclusions"][0]["support"] = ["missing"]
    with pytest.raises(ValidationError, match="missing"):
        LegalState.model_validate_json(json.dumps(payload))


def test_json_schema() -> None:
    schema = LegalState.model_json_schema()
    assert set(schema["properties"]) == {
        "issues",
        "facts",
        "knowledge",
        "relations",
        "conclusions",
    }
    assert schema["$defs"]["IssueStatus"]["enum"] == ["open", "reasoning", "resolved"]
    assert schema["additionalProperties"] is False


def test_revalidate_after_mutating_lists(complete_state: LegalState) -> None:
    complete_state.facts.pop(0)
    with pytest.raises(ValidationError, match="F1"):
        LegalState.model_validate(complete_state.model_dump())


def test_state_transition(complete_state: LegalState) -> None:
    before = LegalState()
    transition = StateTransition(
        operation="EXPAND_ISSUE", before=before, after=complete_state
    )
    assert transition.operation == "EXPAND_ISSUE"
    assert transition.before == before
    assert transition.after == complete_state
    assert set(transition.model_dump()) == {"operation", "before", "after"}


@pytest.mark.parametrize("field", ["operation", "before", "after"])
def test_state_transition_requires_fields(field: str) -> None:
    data = {"operation": "EXPAND_ISSUE", "before": {}, "after": {}}
    del data[field]
    with pytest.raises(ValidationError, match=field):
        StateTransition.model_validate(data)


@pytest.mark.parametrize("operation", ["", 123, None])
def test_state_transition_invalid_operation(operation: object) -> None:
    with pytest.raises(ValidationError, match="operation"):
        StateTransition.model_validate(
            {"operation": operation, "before": {}, "after": {}}
        )


def test_state_transition_unknown_field() -> None:
    with pytest.raises(ValidationError, match="arguments"):
        StateTransition.model_validate(
            {"operation": "BIND_FACT", "before": {}, "after": {}, "arguments": {}}
        )


@pytest.mark.parametrize("field", ["before", "after"])
def test_state_transition_invalid_snapshot(
    complete_state: LegalState, field: str
) -> None:
    data = StateTransition(
        operation="RESOLVE", before=complete_state, after=complete_state
    ).model_dump()
    data[field]["conclusions"][0]["support"] = ["missing"]
    with pytest.raises(ValidationError) as error:
        StateTransition.model_validate(data)
    assert error.value.errors()[0]["loc"] == (field,)
    assert "missing" in str(error.value)


def test_state_transition_json_round_trip(complete_state: LegalState) -> None:
    transition = StateTransition(
        operation="RESOLVE", before=complete_state, after=complete_state
    )
    serialized = transition.model_dump_json()
    restored = StateTransition.model_validate_json(serialized)
    assert restored == transition
    assert restored.after.issues[0].status is IssueStatus.RESOLVED
