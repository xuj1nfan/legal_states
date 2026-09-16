"""运行 `uv run python examples/demo.py`，查看完整状态操作轨迹。

示例中的事实、规则和结论均为手工输入。
"""

from legal_state.operations import bind_fact, commit, expand_issue, resolve
from legal_state.schemas import Fact, Knowledge, LegalState, StateTransition


def show_state(label: str, state: LegalState) -> None:
    print(f"\n{label}")
    print(state.model_dump_json(indent=2))


def main() -> None:
    print("案例：甲借给乙10万元，约定一年后偿还；到期后乙未偿还，甲请求返还本金。")
    base = LegalState(
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
    )
    r0 = expand_issue(base, "乙是否负有返还借款本金的义务")
    issue_id = r0.issues[-1].id
    show_state("R0：通过 EXPAND_ISSUE 创建初始争点", r0)

    print(f"\n→ BIND_FACT({issue_id}, [F1, F2, F3])")
    r1 = bind_fact(r0, issue_id, ["F1", "F2", "F3"])
    show_state("R1：保存事实绑定关系，争点进入 reasoning", r1)

    print(f"\n→ COMMIT({issue_id}, 结论, [F1, F2, F3, K1])")
    r2 = commit(
        r1,
        issue_id,
        "乙负有返还借款本金的义务",
        ["F1", "F2", "F3", "K1"],
    )
    show_state("R2：保存结论，争点仍为 reasoning", r2)

    print(f"\n→ RESOLVE({issue_id})")
    r3 = resolve(r2, issue_id)
    show_state("R3：确认已有结论，将争点标记为 resolved", r3)

    print("\nR0 → BIND_FACT → R1 → COMMIT → R2 → RESOLVE → R3")
    print(
        "保留的争点状态："
        + " → ".join(state.issues[0].status.value for state in [r0, r1, r2, r3])
    )

    trajectory: list[StateTransition] = [
        StateTransition(operation="BIND_FACT", before=r0, after=r1),
        StateTransition(operation="COMMIT", before=r1, after=r2),
        StateTransition(operation="RESOLVE", before=r2, after=r3),
    ]
    print("\n状态转移记录（trajectory）：")
    for index, transition in enumerate(trajectory):
        print(
            f"{index}: {transition.operation} | "
            f"{transition.before.issues[0].status.value} → "
            f"{transition.after.issues[0].status.value}"
        )


if __name__ == "__main__":
    main()
