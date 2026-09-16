import re

from legal_state.schemas import (
    Conclusion,
    Issue,
    IssueStatus,
    LegalState,
    Relation,
)

__all__ = [
    "StateOperationError",
    "IssueNotFoundError",
    "InvalidTransitionError",
    "expand_issue",
    "bind_fact",
    "commit",
    "resolve",
]


class StateOperationError(ValueError):
    """操作参数不符合要求。"""


class IssueNotFoundError(StateOperationError):
    """目标争点或父争点不存在。"""


class InvalidTransitionError(StateOperationError):
    """争点状态不允许此操作，或缺少必要结论。"""


def _validated_snapshot(state: LegalState) -> LegalState:
    # 转成字典再加载，确保已被修改的内部对象也会重新校验。
    return LegalState.model_validate(state.model_dump())


def _next_id(state: LegalState, prefix: str) -> str:
    objects = [
        *state.issues,
        *state.facts,
        *state.knowledge,
        *state.relations,
        *state.conclusions,
    ]
    next_number = 1
    for obj in objects:
        match = re.fullmatch(f"{prefix}([0-9]+)", obj.id)
        if match is not None:
            next_number = max(next_number, int(match.group(1)) + 1)
    return f"{prefix}{next_number}"


def _find_issue(state: LegalState, issue_id: str, operation: str) -> Issue:
    for issue in state.issues:
        if issue.id == issue_id:
            return issue
    raise IssueNotFoundError(f"{operation}: issue {issue_id!r} does not exist")


def _require_status(
    issue: Issue, operation: str, *allowed: IssueStatus
) -> None:
    if issue.status not in allowed:
        expected = ", ".join(status.value for status in allowed)
        raise InvalidTransitionError(
            f"{operation}: issue {issue.id!r} has status {issue.status.value!r}; "
            f"expected {expected}"
        )


def expand_issue(
    state: LegalState, question: str, parent_issue: str | None = None
) -> LegalState:
    """新增 open 争点，自动分配 I 前缀的 ID。
    指定的父争点必须存在；父争点保持原状态，即使它已经解决。
    """
    candidate = _validated_snapshot(state)
    if parent_issue is not None:
        _find_issue(candidate, parent_issue, "EXPAND_ISSUE")
    candidate.issues.append(
        Issue(
            id=_next_id(candidate, "I"),
            question=question,
            parent_issue=parent_issue,
        )
    )
    return _validated_snapshot(candidate)


def bind_fact(state: LegalState, issue_id: str, fact_ids: list[str]) -> LegalState:
    """记录事实绑定关系，将 open 争点转为 reasoning。
    reasoning 争点可继续绑定，每次新增一条关系。事实列表不能为空，
    且只能引用已有事实。绑定只记录关联，不表示事实已证明法律结论。
    """
    candidate = _validated_snapshot(state)
    issue = _find_issue(candidate, issue_id, "BIND_FACT")
    _require_status(issue, "BIND_FACT", IssueStatus.OPEN, IssueStatus.REASONING)
    if not fact_ids:
        raise StateOperationError("BIND_FACT: at least one fact ID is required")
    candidate.relations.append(
        Relation(
            id=_next_id(candidate, "A"),
            issue_id=issue_id,
            fact_ids=fact_ids,
            knowledge_ids=[],
            description=f"将事实 {fact_ids!r} 绑定到争点 {issue_id}",
        )
    )
    issue.status = IssueStatus.REASONING
    return _validated_snapshot(candidate)


def commit(
    state: LegalState, issue_id: str, conclusion: str, support: list[str]
) -> LegalState:
    """为 reasoning 争点新增结论，自动分配 C 前缀的 ID。
    支持依据只能引用已有事实或知识，也可为空。写入结论后不改变争点状态。
    """
    candidate = _validated_snapshot(state)
    issue = _find_issue(candidate, issue_id, "COMMIT")
    _require_status(issue, "COMMIT", IssueStatus.REASONING)
    candidate.conclusions.append(
        Conclusion(
            id=_next_id(candidate, "C"),
            issue_id=issue_id,
            content=conclusion,
            support=support,
        )
    )
    return _validated_snapshot(candidate)


def resolve(state: LegalState, issue_id: str) -> LegalState:
    """只有 reasoning 争点已有对应结论时，才可标记为 resolved。"""
    candidate = _validated_snapshot(state)
    issue = _find_issue(candidate, issue_id, "RESOLVE")
    _require_status(issue, "RESOLVE", IssueStatus.REASONING)
    if not any(item.issue_id == issue_id for item in candidate.conclusions):
        raise InvalidTransitionError(
            f"RESOLVE: issue {issue_id!r} has no conclusion"
        )
    issue.status = IssueStatus.RESOLVED
    return _validated_snapshot(candidate)
