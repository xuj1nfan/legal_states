from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from legal_state.schemas import OptionAssessment, OptionDecision

__all__ = [
    "Action",
    "ActionName",
    "AssessOptionAction",
    "AuditOptionsAction",
    "BindFactAction",
    "CommitAction",
    "ExpandIssueAction",
    "ResolveAction",
    "StopAction",
    "parse_action_json",
]


class ActionName(StrEnum):
    EXPAND_ISSUE = "EXPAND_ISSUE"
    BIND_FACT = "BIND_FACT"
    COMMIT = "COMMIT"
    RESOLVE = "RESOLVE"
    STOP = "STOP"
    ASSESS_OPTION = "ASSESS_OPTION"
    AUDIT_OPTIONS = "AUDIT_OPTIONS"


class _ActionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ExpandIssueAction(_ActionModel):
    operation: Literal[ActionName.EXPAND_ISSUE]
    question: str = Field(min_length=1)
    parent_issue: str | None = None
    scope: str | None = Field(
        default=None, min_length=1, exclude_if=lambda value: value is None
    )


class BindFactAction(_ActionModel):
    operation: Literal[ActionName.BIND_FACT]
    issue_id: str = Field(min_length=1)
    fact_ids: list[str] = Field(min_length=1)


class CommitAction(_ActionModel):
    operation: Literal[ActionName.COMMIT]
    issue_id: str = Field(min_length=1)
    conclusion: str = Field(min_length=1)
    support: list[str]


class ResolveAction(_ActionModel):
    operation: Literal[ActionName.RESOLVE]
    issue_id: str = Field(min_length=1)


class StopAction(_ActionModel):
    operation: Literal[ActionName.STOP]


class AssessOptionAction(_ActionModel):
    operation: Literal[ActionName.ASSESS_OPTION]
    assessment: OptionAssessment


class AuditOptionsAction(_ActionModel):
    operation: Literal[ActionName.AUDIT_OPTIONS]
    decision: OptionDecision


Action = Annotated[
    ExpandIssueAction | BindFactAction | CommitAction | ResolveAction | StopAction
    | AssessOptionAction | AuditOptionsAction,
    Field(discriminator="operation"),
]

_ACTION_ADAPTER = TypeAdapter(Action)


def parse_action_json(data: str | bytes | bytearray) -> Action:
    """解析并严格校验模型生成的单个 JSON 行动。"""
    return _ACTION_ADAPTER.validate_json(data, strict=True)
