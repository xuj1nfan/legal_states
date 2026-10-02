"""覆盖四个选项、映射规则条件并在选择前复核的闭卷流程。"""

import json
import re
from itertools import combinations

from legal_state.actions import AssessOptionAction, AuditOptionsAction, StopAction
from legal_state.operations import (
    InvalidTransitionError,
    bind_fact,
    commit,
    expand_issue,
    resolve,
)
from legal_state.schemas import LegalState, OptionAssessment, OptionDecision


def material_date_intervals(materials: list[dict]) -> list[dict]:
    """Calculate calendar-month distances between quoted dates, without legal rules.

    Month precision is preserved: this is not an elapsed-day calculation or a
    finding about a right's duration. Every endpoint retains its material ID.
    """
    dates: dict[tuple[int, int], dict] = {}
    for material in materials:
        for match in re.finditer(r"(\d{4})年(\d{1,2})月(?:\d{1,2}日)?", material["content"]):
            year, month = map(int, match.group(1, 2))
            if not 1 <= month <= 12:
                continue
            key = year, month
            if key not in dates:
                dates[key] = {"text": match.group(), "evidence": []}
            if material["id"] not in dates[key]["evidence"]:
                dates[key]["evidence"].append(material["id"])
    intervals = []
    for start, end in combinations(sorted(dates), 2):
        intervals.append({
            "from": dates[start], "to": dates[end],
            "calendar_month_difference": (end[0] - start[0]) * 12 + end[1] - start[1],
        })
    return intervals


def next_option(state: LegalState) -> str | None:
    assessed = {item.option for item in state.option_assessments}
    return next((option for option in "ABCD" if option not in assessed), None)


def assessment_response_format(state: LegalState) -> dict:
    """约束本次动作的 JSON 语法和字段；引用仍由执行器校验。"""
    if state.option_decision:
        action_type = StopAction
    elif next_option(state) is not None:
        action_type = AssessOptionAction
    else:
        action_type = AuditOptionsAction
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "legal_state_action",
            "schema": action_type.model_json_schema(),
        },
    }


def assess_option(state: LegalState, assessment: OptionAssessment) -> LegalState:
    candidate = LegalState.model_validate(state.model_dump())
    assessment = OptionAssessment.model_validate(assessment.model_dump())
    if candidate.option_decision or assessment.option != next_option(candidate):
        raise InvalidTransitionError("Assess the next uncovered option before auditing")
    scopes = {
        fact.material.scope for fact in candidate.facts if fact.material is not None
    }
    if not set("ABCD") <= scopes:
        raise InvalidTransitionError("Assessment workflow requires quoted A-D materials")
    candidate.option_assessments.append(assessment)
    # Validate scope/evidence before creating any dependent objects.
    candidate = LegalState.model_validate(candidate.model_dump())
    candidate = expand_issue(
        candidate, f"选项 {assessment.option} 是否符合原题问法？", scope=assessment.option
    )
    issue_id = candidate.issues[-1].id
    evidence = list(
        dict.fromkeys(
            reference
            for condition in assessment.conditions
            for reference in condition.evidence
        )
    )
    candidate = bind_fact(candidate, issue_id, evidence)
    return commit(
        candidate,
        issue_id,
        "待复核初判：" + assessment.model_dump_json(),
        evidence,
    )


def audit_options(state: LegalState, decision: OptionDecision) -> LegalState:
    candidate = LegalState.model_validate(state.model_dump())
    decision = OptionDecision.model_validate(decision.model_dump())
    if next_option(candidate) is not None or candidate.option_decision:
        raise InvalidTransitionError("Audit requires four assessments and no prior decision")
    candidate.option_decision = decision
    for issue in list(candidate.issues):
        if issue.scope in set("ABCD"):
            candidate = resolve(candidate, issue.id)
    candidate = expand_issue(candidate, "复核四个选项并确定最终选择")
    issue_id = candidate.issues[-1].id
    evidence = [fact.id for fact in candidate.facts]
    candidate = bind_fact(candidate, issue_id, evidence)
    candidate = commit(
        candidate,
        issue_id,
        f"最终选择：{decision.answer}。{decision.rationale}。"
        f"竞争选项复核：{decision.counterargument}。"
        "前面的选项评估均为初判，最终判断以本次复核为准。",
        [conclusion.id for conclusion in candidate.conclusions],
    )
    return resolve(candidate, issue_id)


def build_assessment_prompt(question: str, state: LegalState) -> str:
    """只给当前阶段必要的材料，完整快照仍由 runner 保存。"""
    option = next_option(state)
    common = (
        "你是法律推理状态的单步行动生成器。仅依据原题和已有知识闭卷分析，"
        "不访问外部知识源。只输出一个严格 JSON，无 Markdown。"
        "注意原题问的是正确、错误、合法、不合法或例外；"
        "meets_question 表示符合题目要求的答案，does_not_meet 表示不符合，"
        "uncertain 表示依据不足。不要把选项自身的法律断言当成已证规则。"
        "规则记忆不确定时明确说明，禁止虚构条号。遵循题中时间与程序阶段。"
        "先检查选项真正问的条件：日期、期限、身份、专门用途、程序名称。"
        "时间经过可能改变权利和义务；名称相似的程序不可混同。"
        "涉及时间时先定位起算点，计算法定期限与合同期限，检查判断时权利是否仍存续；"
        "日期间隔只是题面计算，不是给定法条，不能单凭间隔推定权利效力。"
        "逐字核对选项与规则，尤其主语、指定主体、仅限、应当、可以等限定，"
        "不能以近似正确的表述代替原选项。"
        "使用紧凑 JSON，每项 reason 不超过40字，其余文字字段不超过60字，"
        "条件最多三项，不重述题面，保证在输出预算内写完 JSON。\n"
    )
    if state.option_decision:
        return common + '四项评估及竞争复核已完成，只输出 {"operation":"STOP"}。'
    if option is not None:
        materials = [
            {"id": fact.id, "content": fact.content, "scope": fact.material.scope}
            for fact in state.facts
            if fact.material and fact.material.scope in (None, option)
        ]
        # Other initial verdicts are intentionally omitted to reduce anchoring.
        stem = "\n".join(m["content"] for m in materials if m["scope"] is None)
        payload = {"question": stem, "current_option": option, "materials": materials,
                   "knowledge": [item.model_dump() for item in state.knowledge]}
        intervals = material_date_intervals(materials)
        if intervals:
            payload["quoted_date_intervals"] = intervals
        template = {
            "operation": "ASSESS_OPTION",
            "assessment": {
                "option": option, "rule": "适用规则及必要条件",
                "rule_source": "model_recall",
                "conditions": [{"condition": "决定性条件", "evidence": [materials[-1]["id"]],
                                "finding": "将本选项原文对应到条件，说明成立与否及推理"}],
                "exception": "核查例外、反例、复合表述是否有部分错误",
                "verdict": "uncertain",
            },
        }
        return common + (
            f"独立评估选项 {option}，暂不作最终选择。识别决定性法律问题，"
            "写出规则，再把具体案情逐项对应到条件；明确一般规则是否被例外排除。"
            "同一事实可能产生多个法律后果，应检查竞合、并列责任等，不能从一个后果成立推断其他后果不存在。"
            "独立从原文识别本选项的决定性事实，不沿用其他选项的判断。"
            "不要只概括一般规则，须检查题中是否存在改变规则分支的当事人态度、权利存续或程序行为。"
            "evidence 只能引用下面的材料 ID，不得跨选项引用。"
            "条件没有题面依据时 evidence=[] 并在 finding 明确缺失或依据规则记忆判断，"
            "不能将材料未提及当作条件已经证明；整项评估至少引用一条材料。"
            "provided_knowledge 仅在输入确有给定法律知识时使用，否则用 model_recall。\n"
            "输出格式（文字是占位说明）：" + json.dumps(template, ensure_ascii=False)
            + "\n输入数据：\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )
    payload = {
        "question": question,
        "initial_assessments": [
            item.model_dump(exclude={"verdict"}) for item in state.option_assessments
        ],
    }
    if state.draft_reasoning is not None:
        payload["tentative_draft"] = state.draft_reasoning
    template = {
        "operation": "AUDIT_OPTIONS",
        "decision": {
            "reviews": [{"option": letter, "reason": "法律判断及是否符合题目问法，或更正原因"}
                        for letter in "ABCD"],
            "answer": "UNKNOWN", "rationale": "比较后最终选择及关键规则",
            "counterargument": "指出最容易混淆的其他选项并说明为何不选；若无竞争项，说明排除依据",
        },
    }
    return common + (
        "四个初判均可能出错。重新核查原题问法、决定性规则、例外和适用阶段，"
        "对每个选项用文字给出复核判断及原因；发现初判错误应明确更正，不盲从初判。"
        "仅排除其他选项不能证明所选项成立，还须说明所选项符合决定性规则。"
        "若有暂定分析，应比较它和条件核查的实质理由；"
        "更改暂定选择须指出具体被忽略的事实、规则或例外，不仅因为表达格式不同。"
        "暂定分析不是事实或给定知识；也不能仅因为有草稿就沿用它的规则记忆。"
        "所有复核理由、rationale 和 answer 必须相互一致；不得选择一个又在理由中否定它。"
        "若多个选项初判正确，必须找到其他选项具体的错误命题，"
        "不得凭‘更直接、更全面、更严谨’或原文未要求的条件排除实质正确的选项。"
        "诉讼主体须核查各方赞成或否认权利主张所触发的条件分支，不机械套用一般规则。"
        "比较全部选项后给出唯一最佳答案，同时说明另一个最强竞争选项为何不如所选项。"
        "不能把局部成立扩展成整个复合选项正确。确实无法区分时才用 UNKNOWN。\n"
        "输出格式（文字是占位说明）：" + json.dumps(template, ensure_ascii=False)
        + "\n输入数据：\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
