from typing import assert_never

from legal_state.actions import (
    Action,
    AssessOptionAction,
    AuditOptionsAction,
    BindFactAction,
    CommitAction,
    ExpandIssueAction,
    FrameQuestionAction,
    ResolveAction,
    StopAction,
)
from legal_state.assessment import assess_option, audit_options, frame_question
from legal_state.operations import bind_fact, commit, expand_issue, resolve
from legal_state.schemas import LegalState

__all__ = ["apply_action"]


def apply_action(state: LegalState, action: Action) -> LegalState:
    """将一个已校验行动分派给对应的状态操作。"""
    if isinstance(action, FrameQuestionAction):
        return frame_question(state, action.frame)
    if isinstance(action, AssessOptionAction):
        return assess_option(state, action.assessment)
    if isinstance(action, AuditOptionsAction):
        return audit_options(state, action.decision)
    if isinstance(action, ExpandIssueAction):
        return expand_issue(
            state, action.question, action.parent_issue, scope=action.scope
        )
    if isinstance(action, BindFactAction):
        return bind_fact(state, action.issue_id, action.fact_ids)
    if isinstance(action, CommitAction):
        return commit(state, action.issue_id, action.conclusion, action.support)
    if isinstance(action, ResolveAction):
        return resolve(state, action.issue_id)
    if isinstance(action, StopAction):
        return LegalState.model_validate(state.model_dump())
    assert_never(action)
