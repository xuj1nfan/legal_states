import json
from collections.abc import Iterable

from legal_state.actions import ActionName
from legal_state.schemas import LegalState

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
        "support": ["已有事实或知识 ID"],
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


def build_action_prompt(
    case_text: str,
    question: str,
    state: LegalState,
    allowed_operations: Iterable[ActionName | str],
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
    stop_instruction = ""
    if ActionName.STOP in operations:
        stop_instruction = (
            "\n7.当你认为当前 LegalState 已经无需继续更新时，可以选择 STOP；"
            "STOP 仅表示结束状态构建，不生成也不等同于最终答案。"
        )
    formats = "\n".join(
        "- "
        + json.dumps(
            _ACTION_FORMATS[operation], ensure_ascii=False, separators=(",", ":")
        )
        for operation in operations
    )

    return f"""你是法律推理状态的单步行动生成器。
根据输入数据和当前 LegalState，选择并生成恰好一个允许的行动。

规则：
1. 只能使用以下 operation：{operation_names}
2. 只输出一个合法 JSON 对象，不要输出 Markdown、解释或思维链。
3. JSON 必须严格匹配对应格式，不得增加额外字段，也不得省略必填字段。
4. issue_id、parent_issue、fact_ids 和 support 中的引用必须使用当前 LegalState 中已有的 ID；EXPAND_ISSUE 的 parent_issue 可以为 null。
5. BIND_FACT 的 fact_ids 至少包含一个事实 ID；COMMIT 的 support 可以为空数组。
6. 选择符合当前争点状态及已有结论的行动。{stop_instruction}

允许的 JSON 格式：
{formats}

输入数据：
{input_json}
"""
