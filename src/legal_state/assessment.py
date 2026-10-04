"""覆盖四个选项、映射规则条件并在选择前复核的闭卷流程。"""

import json
import re
from copy import deepcopy
from itertools import combinations

from legal_state.actions import (
    AssessOptionAction, AuditOptionsAction, FrameQuestionAction, StopAction,
)
from legal_state.operations import (
    InvalidTransitionError,
    bind_fact,
    commit,
    expand_issue,
    resolve,
)
from legal_state.schemas import LegalState, OptionAssessment, OptionDecision, QuestionFrame


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


def material_quote_candidates(materials: list[dict]) -> dict[str, list[str]]:
    """原文短句白名单，只做字符切分，不补充事实或法律判断。"""
    candidates = {}
    for material in materials:
        quotes = []
        for clause in re.split(r'[，,。.;；:：?？!！“”"\n]', material["content"]):
            clause = clause.strip()
            if not clause:
                continue
            if len(clause) <= 32:
                quotes.append(clause)
            else:
                quotes.extend(clause[start:start + 32] for start in range(0, len(clause), 32))
                quotes.append(clause[-32:])
        candidates[material["id"]] = list(dict.fromkeys(quotes or [material["content"][:32]]))
    return candidates


def scoped_material_date_intervals(materials: list[dict]) -> list[dict]:
    """Calculate within a stem/option, without combining separate hypothetical cases."""
    result = []
    seen = set()
    scopes = {material["id"]: material.get("scope") for material in materials}
    for scope in (None, *"ABCD"):
        visible = [material for material in materials if material.get("scope") in (None, scope)]
        for interval in material_date_intervals(visible):
            refs = interval["from"]["evidence"] + interval["to"]["evidence"]
            scoped = {"scope": scope if any(scopes[ref] is not None for ref in refs) else None,
                      **interval}
            key = json.dumps(scoped, ensure_ascii=False, sort_keys=True)
            if key not in seen:
                result.append(scoped)
                seen.add(key)
    return result


def assessment_response_format(state: LegalState, *, require_question_frame: bool = False) -> dict:
    """约束本次动作的 JSON 语法和字段；引用仍由执行器校验。"""
    if state.option_decision:
        action_type = StopAction
    elif require_question_frame and state.question_frame is None:
        action_type = FrameQuestionAction
    elif next_option(state) is not None:
        action_type = AssessOptionAction
    else:
        action_type = AuditOptionsAction
    schema = action_type.model_json_schema()
    if action_type is AssessOptionAction:
        option = next_option(state)
        allowed = [fact.id for fact in state.facts
                   if fact.material and fact.material.scope in (None, option)]
        assessment_schema = schema["$defs"]["OptionAssessment"]
        assessment_schema["properties"]["option"] = {"type": "string", "const": option}
        condition_schema = schema["$defs"]["RuleCondition"]
        condition_schema["properties"]["evidence"]["items"]["enum"] = allowed
        # Start with a grounded condition. Later missing conditions may retain [],
        # without forcing fabricated evidence or repairing a generated response.
        first_condition = deepcopy(condition_schema)
        first_condition["properties"]["evidence"]["minItems"] = 1
        assessment_schema["properties"]["conditions"]["prefixItems"] = [first_condition]
    if action_type is FrameQuestionAction:
        # Scope/reference pairs are known from original quote metadata. Constrain
        # them before generation; never substitute references after a response.
        check_schema = schema["$defs"]["FramedCheck"]
        quotes = material_quote_candidates([
            {"id": fact.id, "content": fact.content} for fact in state.facts if fact.material
        ])
        branches = []
        for scope in (None, *"ABCD"):
            allowed = [fact.id for fact in state.facts
                       if fact.material and fact.material.scope in (None, scope)]
            for owner in allowed:
                branch = deepcopy(check_schema)
                branch["required"] = ["trigger_quote", "question", "evidence", "scope"]
                branch["properties"]["trigger_quote"] = {"type": "string", "enum": quotes[owner]}
                branch["properties"]["scope"] = {
                    "type": "null" if scope is None else "string", "const": scope,
                }
                branch["properties"]["evidence"]["items"] = {"type": "string", "enum": allowed}
                branch["properties"]["evidence"]["enum"] = [
                    list(refs) for size in range(1, min(3, len(allowed)) + 1)
                    for refs in combinations(allowed, size) if owner in refs
                ]
                branches.append(branch)
        schema["$defs"]["FramedCheck"] = {"anyOf": branches}
        schema["$defs"]["QuestionFrame"]["properties"]["checks"]["maxItems"] = 3
    if action_type is AuditOptionsAction:
        # Keep old stored decisions readable, but require complete verdicts from
        # the current constrained model. Semantic consistency is checked locally.
        review_schema = schema["$defs"]["OptionReview"]
        review_schema["required"].append("verdict")
        review_schema["properties"]["verdict"] = {
            "type": "string",
            "enum": ["meets_question", "does_not_meet", "uncertain"],
        }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "legal_state_action",
            "schema": schema,
        },
    }


def frame_question(state: LegalState, frame: QuestionFrame) -> LegalState:
    """保存有引用的检查计划，不创建事实或把模型记忆标为给定规则。"""
    candidate = LegalState.model_validate(state.model_dump())
    if candidate.question_frame or candidate.option_assessments or candidate.option_decision:
        raise InvalidTransitionError("Frame the question once, before assessments")
    scopes = {fact.material.scope for fact in candidate.facts if fact.material}
    if not set("ABCD") <= scopes:
        raise InvalidTransitionError("Question framing requires quoted A-D materials")
    if any(check.trigger_quote is None for check in frame.checks):
        raise InvalidTransitionError("New framed checks require a verbatim trigger quote")
    candidate.question_frame = QuestionFrame.model_validate(frame.model_dump())
    return LegalState.model_validate(candidate.model_dump())


def build_question_frame_prompt(question: str, state: LegalState) -> str:
    materials = [
        {"id": fact.id, "content": fact.content, "scope": fact.material.scope}
        for fact in state.facts if fact.material
    ]
    payload = {"question": question, "materials": materials,
               "quote_candidates": material_quote_candidates(materials)}
    intervals = scoped_material_date_intervals(materials)
    if intervals:
        payload["quoted_date_intervals"] = intervals
    template = {
        "operation": "FRAME_QUESTION",
        "frame": {"question_type": "correct", "checks": [
            {"trigger_quote": "原文决定性条件的短引文", "question": "这一条件可能改变哪条规则分支？",
             "evidence": ["F1"], "scope": None},
        ]},
    }
    return (
        "你是法律推理状态的单步行动生成器。先从原文定位必要争点，只输出严格 JSON。"
        "本步只提出待核查问题，不选择答案、不写法律结论、不虚构规则。"
        "question_type 为 correct（选正确）、incorrect（选错误）或 other。"
        "先找题干中可能改变结论的具体事实，逐字摘录为 trigger_quote，"
        "再围绕该事实提问。引用必须完整保留限定词，不是概括或改写。"
        "优先检查题干的具体日期、当事人的赞成或否认、具体行为，"
        "不要仅把选项改写成问题。题干无具体案情时，再检查各选项的原文细节。"
        "检查容易遗漏的决定性细节：基础权利与合同各自的期限及起点、"
        "当事人身份与具体立场、程序类型与所处阶段、复合选项的各个命题。"
        "每项 question 应点明具体原文细节及需要核查的问题，不写笼统的‘是否合法’。"
        "最多三项，每个问题不超过30字。trigger_quote 必须从 quote_candidates "
        "中逐字选取，不能自行缩写、改写或插入空格。"
        "trigger_quote 必须逐字出现在 evidence 引用的某条材料中。"
        "evidence 只引用原文材料 ID；"
        "题干问题用 scope=null 并只引用题干；选项专属问题用对应字母，"
        "仅引用题干及该选项，不把不同选项的假设拼成同一案情。"
        "月份计算只是原文算术，不代表法定期限。\n输出格式："
        + json.dumps(template, ensure_ascii=False)
        + "\n输入数据：\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


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


def build_assessment_prompt(
    question: str, state: LegalState, *, require_question_frame: bool = False,
    compare_options: bool = False,
) -> str:
    """只给当前阶段必要的材料，完整快照仍由 runner 保存。"""
    if require_question_frame and state.question_frame is None and state.option_decision is None:
        return build_question_frame_prompt(question, state)
    option = next_option(state)
    framing_instruction = (
        "question_frame 是待核查计划；先逐项回答当前可见的具体问题，"
        "再形成规则和结论。遗漏的问题先回到原文核查；"
        "计划不是事实、法律知识或可引用的 evidence。"
    ) if state.question_frame is not None else ""
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
        "区分选项提出的假设与题干已发生的事实：选项若说‘满足某条件后可以’，"
        "应判断这一条件命题是否合法，不能因题干尚未发生该条件就判选项错误。"
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
        # Other assessments are omitted; a draft is a hypothesis, never evidence.
        stem = "\n".join(m["content"] for m in materials if m["scope"] is None)
        payload = {"question": stem, "current_option": option, "materials": materials,
                   "knowledge": [item.model_dump() for item in state.knowledge]}
        if compare_options:
            payload["comparison_options"] = {
                fact.material.scope: fact.content for fact in state.facts
                if fact.material and fact.material.scope is not None
            }
        if state.question_frame is not None:
            payload["question_frame"] = {
                "question_type": state.question_frame.question_type,
                "checks": [check.model_dump() for check in state.question_frame.checks
                           if check.scope in (None, option)],
            }
        if state.draft_reasoning is not None:
            payload["tentative_draft"] = state.draft_reasoning
        intervals = material_date_intervals(materials)
        if intervals:
            payload["quoted_date_intervals"] = intervals
        template = {
            "operation": "ASSESS_OPTION",
            "assessment": {
                "option": option,
                "conditions": [{"condition": "决定性条件", "evidence": [materials[-1]["id"]],
                                "finding": "原文事实或选项假设如何满足该条件"}],
                "rule": "适用规则及必要条件", "rule_source": "model_recall",
                "exception": "核查例外、反例、复合表述是否有部分错误",
                "verdict": "uncertain",
            },
        }
        return common + framing_instruction + (
            ("comparison_options 仅帮助比较原题候选命题，不是已证事实或引用依据。"
             "各选项假设独立，不得把其他选项的事实、时间或条件并入当前案情。"
             "判断当前选项是否完整回答原题要求，不能把部分法律后果存在当成处断规则完整。"
             if compare_options else "")
            +
            f"独立评估选项 {option}，暂不作最终选择。识别决定性法律问题，"
            "先把具体案情逐项对应到 conditions，再回忆规则；明确一般规则是否被例外排除。"
            "同一事实可能产生多个法律后果，应检查竞合、并列责任等，不能从一个后果成立推断其他后果不存在。"
            "独立从原文识别本选项的决定性事实，不沿用其他选项的判断。"
            "tentative_draft 若存在，仅是待验证的模型假说；"
            "核对其决定性规则与事实，正确的理由可保留，有具体反例才更正。"
            "草稿中其他选项的案情不能作为本选项事实，草稿不能作为 evidence。"
            "不要只概括一般规则，须检查题中是否存在改变规则分支的当事人态度、权利存续或程序行为。"
            "evidence 只能引用下面的材料 ID，不得跨选项引用。"
            "条件没有题面依据时 evidence=[] 并在 finding 明确缺失或依据规则记忆判断，"
            "不能将材料未提及当作条件已经证明；第一项条件必须引用至少一条材料，其他缺失条件可用 []。"
            "provided_knowledge 仅在输入确有给定法律知识时使用，否则用 model_recall。\n"
            "输出格式（文字是占位说明）：" + json.dumps(template, ensure_ascii=False)
            + "\n输入数据：\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )
    payload = {
        "question": question,
        "materials": [
            {"id": fact.id, "content": fact.content, "scope": fact.material.scope}
            for fact in state.facts if fact.material
        ],
        "knowledge": [item.model_dump() for item in state.knowledge],
        "initial_assessments": [
            item.model_dump() for item in state.option_assessments
        ],
    }
    if state.draft_reasoning is not None:
        payload["tentative_draft"] = state.draft_reasoning
    if state.question_frame is not None:
        payload["question_frame"] = state.question_frame.model_dump()
    intervals = scoped_material_date_intervals(payload["materials"])
    if intervals:
        payload["quoted_date_intervals"] = intervals
    template = {
        "operation": "AUDIT_OPTIONS",
        "decision": {
            "reviews": [{"option": letter, "reason": "法律判断及是否符合题目问法，或更正原因",
                         "verdict": "uncertain"}
                        for letter in "ABCD"],
            "rationale": "比较后最终选择及关键规则",
            "counterargument": "指出最容易混淆的其他选项并说明为何不选；若无竞争项，说明排除依据",
            "answer": "UNKNOWN",
        },
    }
    return common + framing_instruction + (
        "复核四项初判。初判是待验证假说，不是事实；"
        "核查原题问法、决定性规则、例外和适用阶段，"
        "从 materials 回到原文逐项核对，初判中没有原文支持的事实不得沿用。"
        "针对同一法律问题使用同一套规则及条件分支，不为不同选项随意更换规则。"
        "对每个选项先写 reason，再写 verdict。发现具体错误时更正；"
        "未发现错误时保留有依据的初判，不为更改答案制造新的事实或条件。"
        "仅排除其他选项不能证明所选项成立，还须说明所选项符合决定性规则。"
        "若有暂定分析，应比较它和条件核查的实质理由；"
        "更改暂定选择须指出具体被忽略的事实、规则或例外，不仅因为表达格式不同。"
        "暂定分析不是事实或给定知识；也不能仅因为有草稿就沿用它的规则记忆。"
        "所有复核理由、rationale 和 answer 必须相互一致；不得选择一个又在理由中否定它。"
        "复核中恰有一项 meets_question 时，answer 必须选该项；"
        "零项或多项符合且仍无法解决时，answer 必须 UNKNOWN。"
        "先完成四项复核、选择依据和竞争理由，最后才写 answer；"
        "写 answer 前回看所选项的复核理由是否支持该选择。"
        "若多个选项初判正确，必须找到其他选项具体的错误命题，"
        "不得凭‘更直接、更全面、更严谨’或原文未要求的条件排除实质正确的选项。"
        "比较法律程序时，区别启动条件、审查对象、处理阶段和文书类型；"
        "不能仅凭‘条件不符’的一般直觉推导具体裁判形式。"
        "诉讼主体须核查各方赞成或否认权利主张所触发的条件分支，不机械套用一般规则。"
        "比较全部选项后给出唯一最佳答案，同时说明另一个最强竞争选项为何不如所选项。"
        "不能把局部成立扩展成整个复合选项正确。确实无法区分时才用 UNKNOWN。\n"
        "输出格式（文字是占位说明）：" + json.dumps(template, ensure_ascii=False)
        + "\n输入数据：\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
