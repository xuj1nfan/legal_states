import pytest
from pydantic import ValidationError

from legal_state.actions import (
    ActionName,
    BindFactAction,
    CommitAction,
    ExpandIssueAction,
    ResolveAction,
    parse_action_json,
)


@pytest.mark.parametrize(
    ("payload", "expected_type"),
    [
        (
            (
                '{"operation":"EXPAND_ISSUE","question":"利息约定是否有效",'
                '"parent_issue":"I1"}'
            ),
            ExpandIssueAction,
        ),
        (
            (
                '{"operation":"BIND_FACT","issue_id":"I1",'
                '"fact_ids":["F1","F2"]}'
            ),
            BindFactAction,
        ),
        (
            (
                '{"operation":"COMMIT","issue_id":"I1",'
                '"conclusion":"乙应返还本金","support":["F1","K1"]}'
            ),
            CommitAction,
        ),
        (
            '{"operation":"RESOLVE","issue_id":"I1"}',
            ResolveAction,
        ),
    ],
)
def test_parse_valid_action_json(payload: str, expected_type: type) -> None:
    action = parse_action_json(payload)
    assert isinstance(action, expected_type)
    assert action.operation is ActionName(action.operation)


@pytest.mark.parametrize(
    "payload",
    [
        '{"operation":"SEARCH","query":"借款返还"}',
        '{"operation":"BIND_FACT","fact_ids":["F1"]}',
        '{"operation":"COMMIT","issue_id":"I1","conclusion":"结论"}',
        '{"operation":"RESOLVE","issue_id":"I1","reason":"已有结论"}',
    ],
)
def test_parse_rejects_invalid_missing_or_extra_fields(payload: str) -> None:
    with pytest.raises(ValidationError):
        parse_action_json(payload)


def test_action_models_forbid_extra_fields() -> None:
    with pytest.raises(ValidationError, match="unexpected"):
        BindFactAction.model_validate_json(
            '{"operation":"BIND_FACT","issue_id":"I1",'
            '"fact_ids":["F1"],"unexpected":true}'
        )


@pytest.mark.parametrize(
    "payload",
    [
        '{"operation":"BIND_FACT","issue_id":1,"fact_ids":["F1"]}',
        '{"operation":"BIND_FACT","issue_id":"I1","fact_ids":[]}',
        (
            '{"operation":"COMMIT","issue_id":"I1",'
            '"conclusion":"结论","support":[1]}'
        ),
        '{"operation":"RESOLVE","issue_id":""}',
    ],
)
def test_parse_rejects_wrong_types_and_empty_required_values(payload: str) -> None:
    with pytest.raises(ValidationError):
        parse_action_json(payload)
