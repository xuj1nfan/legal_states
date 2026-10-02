from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "Conclusion",
    "Fact",
    "Issue",
    "IssueStatus",
    "Knowledge",
    "LegalState",
    "MaterialSpan",
    "OptionAssessment",
    "OptionDecision",
    "OptionReview",
    "Relation",
    "RuleCondition",
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
    scope: str | None = Field(
        default=None, min_length=1, exclude_if=lambda value: value is None
    )


class MaterialSpan(_SchemaModel):
    """原问题中的引文位置；选项作用域不表示材料中的命题已成立。

    start/end 使用 Python 字符串索引，end 不包含在引文中。
    scope=None 表示题干；其他值表示独立选项，不能互相混作案情。
    """

    kind: Literal["quoted_material"] = "quoted_material"
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    scope: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.end <= self.start:
            raise ValueError("Material span must have end > start")
        return self


class Fact(_StateObject):
    """案件事实或带有 material 元数据的原文材料；引文不等于已证事实。"""

    content: str = Field(min_length=1)
    source: str = Field(min_length=1)
    material: MaterialSpan | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def validate_material_length(self) -> Self:
        if self.material and self.material.end - self.material.start != len(
            self.content
        ):
            raise ValueError("Material span length must match quoted content")
        return self


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
    """阶段性结论，支持依据可为事实、法律知识或已有阶段性结论的 ID。"""

    issue_id: str
    content: str = Field(min_length=1)
    support: list[str]


OptionLetter = Literal["A", "B", "C", "D"]
OptionVerdict = Literal["meets_question", "does_not_meet", "uncertain"]


class RuleCondition(_SchemaModel):
    """模型声明的规则条件及原文依据；不表示规则已获外部核验。"""

    condition: str = Field(min_length=1)
    evidence: list[str]
    finding: str = Field(min_length=1)


class OptionAssessment(_SchemaModel):
    """按题目问法判断一个选项，保留规则、条件映射及例外。"""

    option: OptionLetter
    rule: str = Field(min_length=1)
    rule_source: Literal["model_recall", "provided_knowledge"]
    conditions: list[RuleCondition] = Field(min_length=1, max_length=3)
    exception: str = Field(min_length=1)
    verdict: OptionVerdict

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if not any(condition.evidence for condition in self.conditions):
            raise ValueError("Assessment needs at least one material reference")
        return self


class OptionReview(_SchemaModel):
    option: OptionLetter
    reason: str = Field(min_length=1)


class OptionDecision(_SchemaModel):
    """复核可修订初判，原始评估仍保留在轨迹中。"""

    reviews: list[OptionReview] = Field(min_length=4, max_length=4)
    answer: Literal["A", "B", "C", "D", "UNKNOWN"]
    rationale: str = Field(min_length=1)
    counterargument: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        if {review.option for review in self.reviews} != set("ABCD"):
            raise ValueError("Audit must review every option exactly once")
        return self


class LegalState(_SchemaModel):
    """经过校验的状态快照，修改字段后需重新校验。
    只检查当前状态的结构和引用，不判断法律结论、依据是否充分或前后状态是否一致。
    """

    issues: list[Issue] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)
    knowledge: list[Knowledge] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    conclusions: list[Conclusion] = Field(default_factory=list)
    option_assessments: list[OptionAssessment] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )
    option_decision: OptionDecision | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    draft_reasoning: str | None = Field(
        default=None, min_length=1, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def validate_option_assessments(self) -> Self:
        options = [item.option for item in self.option_assessments]
        if len(options) != len(set(options)):
            raise ValueError("Duplicate option assessment")
        facts = {fact.id: fact for fact in self.facts}
        knowledge_ids = {item.id for item in self.knowledge}
        for item in self.option_assessments:
            for condition in item.conditions:
                for reference in condition.evidence:
                    if reference not in facts:
                        raise ValueError("Condition evidence must reference quoted facts")
                    material = facts[reference].material
                    if material and material.scope not in (None, item.option):
                        raise ValueError("Condition evidence crosses option scopes")
            if item.rule_source == "provided_knowledge" and not knowledge_ids:
                raise ValueError("No provided knowledge is available for this rule")
        if self.option_decision and set(options) != set("ABCD"):
            raise ValueError("Decision requires assessments for all four options")
        return self

    def support_scopes(self, support: list[str]) -> set[str]:
        """返回引用及其结论依赖的选项作用域，不判断引用是否足够。"""
        facts = {fact.id: fact for fact in self.facts}
        conclusions = {item.id: item for item in self.conclusions}
        issues = {issue.id: issue for issue in self.issues}
        scopes: set[str] = set()
        pending = list(support)
        visited: set[str] = set()
        while pending:
            reference = pending.pop()
            if reference in visited:
                continue
            visited.add(reference)
            if reference in facts:
                material = facts[reference].material
                if material and material.scope is not None:
                    scopes.add(material.scope)
            elif reference in conclusions:
                conclusion = conclusions[reference]
                scope = issues[conclusion.issue_id].scope
                if scope is not None:
                    scopes.add(scope)
                pending.extend(conclusion.support)
        return scopes

    def allowed_support_ids(self, issue_id: str) -> list[str]:
        issue = next(issue for issue in self.issues if issue.id == issue_id)
        references = [
            *(fact.id for fact in self.facts),
            *(item.id for item in self.knowledge),
            *(item.id for item in self.conclusions),
        ]
        return [
            reference
            for reference in references
            if issue.scope is None or self.support_scopes([reference]) <= {issue.scope}
        ]

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
        conclusion_ids = {item.id for item in self.conclusions}
        support_ids = fact_ids | knowledge_ids | conclusion_ids

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
                check_reference(issue.id, "parent_issue", issue.parent_issue, issue_ids)
                if issue.parent_issue == issue.id:
                    errors.append(
                        f"Object {issue.id!r} field 'parent_issue' "
                        f"cannot reference itself ({issue.parent_issue!r})"
                    )

        for relation in self.relations:
            check_reference(relation.id, "issue_id", relation.issue_id, issue_ids)
            for index, fact_id in enumerate(relation.fact_ids):
                check_reference(relation.id, f"fact_ids[{index}]", fact_id, fact_ids)
            for index, knowledge_id in enumerate(relation.knowledge_ids):
                check_reference(
                    relation.id,
                    f"knowledge_ids[{index}]",
                    knowledge_id,
                    knowledge_ids,
                )

        for conclusion in self.conclusions:
            check_reference(conclusion.id, "issue_id", conclusion.issue_id, issue_ids)
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
    def validate_material_scopes(self) -> Self:
        """隔离选项材料，并检查经中间结论传递的跨选项引用。"""
        available = {
            fact.material.scope
            for fact in self.facts
            if fact.material and fact.material.scope is not None
        }
        issues = {issue.id: issue for issue in self.issues}
        for issue in self.issues:
            if issue.scope is not None and issue.scope not in available:
                raise ValueError(f"Issue {issue.id!r} has unknown material scope")
            if issue.parent_issue:
                parent = issues[issue.parent_issue]
                if parent.scope is not None and issue.scope != parent.scope:
                    raise ValueError("Child issue must preserve its parent's scope")
        for relation in self.relations:
            scope = issues[relation.issue_id].scope
            if scope is not None and not self.support_scopes(relation.fact_ids) <= {
                scope
            }:
                raise ValueError(f"Relation {relation.id!r} crosses option scopes")
        for conclusion in self.conclusions:
            scope = issues[conclusion.issue_id].scope
            if scope is not None and not self.support_scopes(conclusion.support) <= {
                scope
            }:
                raise ValueError(f"Conclusion {conclusion.id!r} crosses option scopes")
        return self

    @model_validator(mode="after")
    def validate_conclusion_dependencies(self) -> Self:
        """检查阶段性结论之间的支持依据关系不能形成循环。"""
        conclusion_ids = {conclusion.id for conclusion in self.conclusions}
        dependencies = {
            conclusion.id: [
                support_id
                for support_id in conclusion.support
                if support_id in conclusion_ids
            ]
            for conclusion in self.conclusions
        }
        completed: set[str] = set()
        visiting: set[str] = set()

        def visit(conclusion_id: str, path: list[str]) -> None:
            if conclusion_id in visiting:
                cycle = path[path.index(conclusion_id) :] + [conclusion_id]
                raise ValueError(
                    "Conclusion support cycle detected: " + " -> ".join(cycle)
                )
            if conclusion_id in completed:
                return

            visiting.add(conclusion_id)
            path.append(conclusion_id)
            for dependency_id in dependencies[conclusion_id]:
                visit(dependency_id, path)
            path.pop()
            visiting.remove(conclusion_id)
            completed.add(conclusion_id)

        for conclusion in self.conclusions:
            visit(conclusion.id, [])

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
                    cycle = path[path.index(current) :] + [current]
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
