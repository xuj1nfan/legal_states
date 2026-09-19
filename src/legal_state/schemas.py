from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "Conclusion",
    "Fact",
    "Issue",
    "IssueStatus",
    "Knowledge",
    "LegalState",
    "Relation",
    "StateTransition",
]


class IssueStatus(StrEnum):
    OPEN = "open"
    REASONING = "reasoning"
    RESOLVED = "resolved"


class _SchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _StateObject(_SchemaModel):
    id: str = Field(min_length=1)


class Issue(_StateObject):
    """法律争点，可指定父争点。"""

    question: str = Field(min_length=1)
    status: IssueStatus = IssueStatus.OPEN
    parent_issue: str | None = None


class Fact(_StateObject):
    """案件事实及其来源。"""

    content: str = Field(min_length=1)
    source: str = Field(min_length=1)


class Knowledge(_StateObject):
    """已获得的法律知识及其来源。"""

    content: str = Field(min_length=1)
    source: str = Field(min_length=1)


class Relation(_StateObject):
    """争点、事实和法律知识之间的推理关系。"""

    issue_id: str
    fact_ids: list[str]
    knowledge_ids: list[str]
    description: str = Field(min_length=1)


class Conclusion(_StateObject):
    """阶段性结论，支持依据为事实或知识的 ID。"""

    issue_id: str
    content: str = Field(min_length=1)
    support: list[str]


class LegalState(_SchemaModel):
    """经过校验的状态快照，修改字段后需重新校验。
    只检查当前状态的结构和引用，不判断法律结论、依据是否充分或前后状态是否一致。
    """

    issues: list[Issue] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)
    knowledge: list[Knowledge] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    conclusions: list[Conclusion] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_integrity(self) -> Self:
        objects: list[_StateObject] = [
            *self.issues,
            *self.facts,
            *self.knowledge,
            *self.relations,
            *self.conclusions,
        ]
        errors: list[str] = []
        seen_ids: set[str] = set()
        for obj in objects:
            if obj.id in seen_ids:
                errors.append(f"Duplicate object ID {obj.id!r}")
            seen_ids.add(obj.id)

        issue_ids = {issue.id for issue in self.issues}
        fact_ids = {fact.id for fact in self.facts}
        knowledge_ids = {item.id for item in self.knowledge}
        support_ids = fact_ids | knowledge_ids

        def check_reference(
            object_id: str, field: str, target: str, allowed_ids: set[str]
        ) -> None:
            if target not in allowed_ids:
                errors.append(
                    f"Object {object_id!r} field {field!r} "
                    f"has invalid reference {target!r}"
                )

        for issue in self.issues:
            if issue.parent_issue is not None:
                check_reference(
                    issue.id, "parent_issue", issue.parent_issue, issue_ids
                )
                if issue.parent_issue == issue.id:
                    errors.append(
                        f"Object {issue.id!r} field 'parent_issue' "
                        f"cannot reference itself ({issue.parent_issue!r})"
                    )

        for relation in self.relations:
            check_reference(relation.id, "issue_id", relation.issue_id, issue_ids)
            for index, fact_id in enumerate(relation.fact_ids):
                check_reference(
                    relation.id, f"fact_ids[{index}]", fact_id, fact_ids
                )
            for index, knowledge_id in enumerate(relation.knowledge_ids):
                check_reference(
                    relation.id,
                    f"knowledge_ids[{index}]",
                    knowledge_id,
                    knowledge_ids,
                )

        for conclusion in self.conclusions:
            check_reference(
                conclusion.id, "issue_id", conclusion.issue_id, issue_ids
            )
            for index, support_id in enumerate(conclusion.support):
                check_reference(
                    conclusion.id,
                    f"support[{index}]",
                    support_id,
                    support_ids,
                )

        if errors:
            raise ValueError("; ".join(errors))
        return self

    @model_validator(mode="after")
    def validate_issue_hierarchy(self) -> Self:
        """沿父争点逐步检查循环，每个争点最多遍历一次。"""
        parents = {issue.id: issue.parent_issue for issue in self.issues}
        completed: set[str] = set()

        for issue in self.issues:
            path: list[str] = []
            path_ids: set[str] = set()
            current: str | None = issue.id

            while current is not None and current not in completed:
                if current in path_ids:
                    cycle = path[path.index(current):] + [current]
                    raise ValueError(
                        "Issue parent_issue cycle detected: " + " -> ".join(cycle)
                    )
                path.append(current)
                path_ids.add(current)
                current = parents[current]

            completed.update(path)

        return self


class StateTransition(_SchemaModel):
    """记录一次操作及其前后状态，不判断该操作是否符合状态转移规则。"""

    operation: str = Field(min_length=1)
    before: LegalState
    after: LegalState
