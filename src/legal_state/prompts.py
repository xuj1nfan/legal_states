import json
from collections.abc import Iterable

from legal_state.actions import ActionName
from legal_state.schemas import IssueStatus, LegalState

__all__ = ["build_action_prompt"]


_ACTION_FORMATS: dict[ActionName, dict[str, object]] = {
    ActionName.EXPAND_ISSUE: {
        "operation": "EXPAND_ISSUE",
        "question": "新的法律争点",
        "parent_issue": None,
    },
    ActionName.BIND_FACT: {
        "operation": "BIND_FACT",
        "issue_id": "已有争点 ID",
        "fact_ids": ["已有事实 ID"],
    },
    ActionName.COMMIT: {
        "operation": "COMMIT",
        "issue_id": "已有争点 ID",
        "conclusion": "阶段性结论",
        "support": ["已有事实、知识或阶段性结论 ID"],
    },
    ActionName.RESOLVE: {
        "operation": "RESOLVE",
        "issue_id": "已有争点 ID",
    },
    ActionName.STOP: {
        "operation": "STOP",
    },
}


def _normalize_operations(
    allowed_operations: Iterable[ActionName | str],
) -> list[ActionName]:
    operations: list[ActionName] = []
    for operation in allowed_operations:
        try:
            normalized = ActionName(operation)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Unsupported action operation: {operation!r}") from error
        if normalized not in operations:
            operations.append(normalized)
    if not operations:
        raise ValueError("At least one allowed operation is required")
    return operations


def _format_operation_targets(
    state: LegalState, operation: ActionName, focus_issue_id: str | None = None
) -> str:
    if operation is ActionName.EXPAND_ISSUE:
        issue_ids = ", ".join(issue.id for issue in state.issues)
        parent_hint = "parent_issue 可为 null"
        if issue_ids:
            parent_hint += f" 或已有 issue ID: {issue_ids}"
        return f"- {operation.value}: 可新增争点；{parent_hint}"

    if operation is ActionName.BIND_FACT:
        target_ids = (
            [
                issue.id
                for issue in state.issues
                if issue.status in (IssueStatus.OPEN, IssueStatus.REASONING)
            ]
            if state.facts
            else []
        )
    elif operation is ActionName.COMMIT:
        target_ids = [
            issue.id for issue in state.issues if issue.status is IssueStatus.REASONING
        ]
    elif operation is ActionName.RESOLVE:
        concluded_issue_ids = {conclusion.issue_id for conclusion in state.conclusions}
        target_ids = [
            issue.id
            for issue in state.issues
            if issue.status is IssueStatus.REASONING and issue.id in concluded_issue_ids
        ]
    else:
        return f"- {operation.value}: 不需要 issue_id"

    if focus_issue_id is not None:
        target_ids = [issue_id for issue_id in target_ids if issue_id == focus_issue_id]
    targets = ", ".join(target_ids) or "无合法目标争点"
    return f"- {operation.value}: {targets}"


def build_action_prompt(
    case_text: str,
    question: str,
    state: LegalState,
    allowed_operations: Iterable[ActionName | str],
    *,
    focus_issue_id: str | None = None,
) -> str:
    """将案件、问题、当前状态和允许操作组装为单步行动生成 prompt。"""
    validated_state = LegalState.model_validate(state.model_dump())
    operations = _normalize_operations(allowed_operations)
    input_json = json.dumps(
        {
            "case_text": case_text,
            "question": question,
            "legal_state": validated_state.model_dump(mode="json"),
        },
        ensure_ascii=False,
        indent=2,
    )
    operation_names = ", ".join(operation.value for operation in operations)
    scopes = sorted(
        {
            fact.material.scope
            for fact in validated_state.facts
            if fact.material and fact.material.scope is not None
        }
    )
    material_instruction = ""
    if scopes:
        material_instruction = (
            "\n材料作用域规则：facts 中带 material 的对象是原文引文，"
            "不是已证事实或已经核实的法律规则。material.start/end 是原 question 的字符位置，"
            "end 不含在范围内；scope=null 是题干，其他 scope 是独立选项。"
            "先判断选项是在描述假设案情还是提出待验证的法律命题，不得直接把选项命题当成正确规则。"
            "分析某个选项时 EXPAND_ISSUE 指定其 scope；综合比较时 scope=null。"
            "有作用域的争点只能引用题干和本选项材料，以及不含其他选项依赖的结论。"
            "子争点继承父争点作用域。综合争点可以比较不同选项，"
            "但不同选项的假设案情不能合并成同一案件事实。"
            "以下按争点的白名单优先于全局列表；未列出的引用会被拒绝。\n"
            + json.dumps(
                {
                    issue.id: {
                        "BIND_FACT.fact_ids": [
                            fact.id
                            for fact in validated_state.facts
                            if fact.id in validated_state.allowed_support_ids(issue.id)
                        ],
                        "COMMIT.support": validated_state.allowed_support_ids(issue.id),
                    }
                    for issue in validated_state.issues
                    if issue.scope is not None
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    focus_instruction = ""
    if focus_issue_id is not None:
        if focus_issue_id not in {issue.id for issue in validated_state.issues}:
            raise ValueError("Focus issue is not in the state")
        focus_instruction = (
            f"\n本轮采用逐争点流程：当前只处理 {focus_issue_id}。"
            "先绑定必要材料，再提交含判断和理由的结论，然后解决该争点。"
            "只有当前争点解决后才考虑下一个争点；已完成的动作不会再次提供。"
            "仅输出当前允许的操作，issue_id 必须使用该目标。"
        )
    stop_instruction = ""
    if ActionName.STOP in operations:
        stop_instruction = (
            "\n7. 当你认为当前 LegalState 已经无需继续更新时，"
            "且必要争点均为 resolved、已有足以回答原问题的结论，应选择 STOP；"
            "不要在空状态、尚无结论或仍有必要争点未解决时提前停止。"
            "否则应选择其他允许的操作继续更新状态。STOP 仅表示结束状态构建，"
            "不生成也不等同于最终答案。"
        )
    action_formats = {
        operation: dict(_ACTION_FORMATS[operation]) for operation in operations
    }
    if scopes and ActionName.EXPAND_ISSUE in operations:
        action_formats[ActionName.EXPAND_ISSUE]["scope"] = (
            "已有选项作用域；综合比较用 null"
        )
    formats = "\n".join(
        "- "
        + json.dumps(
            action_formats[operation], ensure_ascii=False, separators=(",", ":")
        )
        for operation in operations
    )
    operation_targets = "\n".join(
        _format_operation_targets(validated_state, operation, focus_issue_id)
        for operation in operations
    )
    references = {
        "EXPAND_ISSUE.parent_issue": [
            None,
            *(issue.id for issue in validated_state.issues),
        ],
        "BIND_FACT.fact_ids": [fact.id for fact in validated_state.facts],
        "COMMIT.support": [
            *(fact.id for fact in validated_state.facts),
            *(item.id for item in validated_state.knowledge),
            *(conclusion.id for conclusion in validated_state.conclusions),
        ],
    }
    if scopes:
        references["EXPAND_ISSUE.scope"] = [None, *scopes]
    reference_ids = json.dumps(
        references,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return f"""你是法律推理状态的单步行动生成器。
根据输入数据和当前 LegalState，选择并生成恰好一个允许的行动。

规则：
1. 只能使用以下 operation：{operation_names}
2. 只输出一个合法 JSON 对象，不要输出 Markdown、解释或思维链。
3. JSON 必须严格匹配对应格式，不得增加额外字段，也不得省略必填字段。文字保持简短，保留必要判断及理由，确保写完完整 JSON，不一次输出长篇分析。
4. issue_id 必须使用当前操作允许的争点 ID；其他引用只能从下方对应字段的白名单选择。EXPAND_ISSUE 的 parent_issue 可以为 null。不要编造新 ID；模型不负责分配对象 ID。
5. BIND_FACT 的 fact_ids 至少包含一个事实 ID；COMMIT 的 support 可以引用当前 LegalState 中已有的事实、知识或阶段性结论 ID，也可以为空数组。不得引用争点或关系 ID；关系只是绑定记录，不是结论依据。如果当前阶段性结论确实依赖已有阶段性结论，应在 support 中引用对应的 Conclusion ID；不要强行创建结论依赖，只有真实依赖时才引用。回答单选题时，支持选择的结论应明确选项字母及简短理由。
6. 按当前状态推进：尚无必要争点时 EXPAND_ISSUE；open 争点先 BIND_FACT；reasoning 争点形成实质判断后 COMMIT；该争点已有足够结论时下一步 RESOLVE。COMMIT 不会自动解决争点。仅在确有新事实或新判断时再次 BIND_FACT/COMMIT；不要重复绑定同一事实、重复结论或重复创建同一争点。每次只能选择一个允许的行动。{stop_instruction}{material_instruction}{focus_instruction}

当前引用白名单（空列表表示没有可引用对象；support 可为 []）：
{reference_ids}

当前操作可使用的争点目标：
{operation_targets}

允许的 JSON 格式：
{formats}

输入数据：
{input_json}
"""
