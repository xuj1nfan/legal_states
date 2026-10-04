"""Coverage, evidence integrity and revision in the verified workflow."""

import json

import pytest
from pydantic import ValidationError

from legal_state.actions import ActionName, parse_action_json
from legal_state.assessment import (
    assess_option, assessment_response_format, audit_options,
    build_assessment_prompt, material_date_intervals,
)
from legal_state.executor import apply_action
from legal_state.experiment.config import ExperimentConfig, ReasoningSettings
from legal_state.experiment.data import CaseInput
from legal_state.experiment.engine import JournalClient, baseline_prompt, run_case
from legal_state.model import ModelCallResult
from legal_state.operations import InvalidTransitionError
from legal_state.runner import RunnerStepError, determine_allowed_operations, run_legal_state
from legal_state.schemas import Fact, LegalState, MaterialSpan, OptionAssessment, OptionDecision


@pytest.fixture
def state():
    return LegalState(facts=[
        Fact(id="F1", content="题干", source="question_material",
             material=MaterialSpan(start=0, end=2)),
        *[Fact(id=f"F{i}", content=letter, source="question_material",
               material=MaterialSpan(start=i, end=i+1, scope=letter))
          for i, letter in enumerate("ABCD", start=2)],
    ])


def assessment(letter, **overrides):
    return OptionAssessment.model_validate({
        "option": letter, "rule": "适用条件", "rule_source": "model_recall",
        "conditions": [{"condition": "条件", "evidence": [f"F{'ABCD'.index(letter)+2}"],
                        "finding": "原文满足条件"}],
        "exception": "无例外", "verdict": "meets_question", **overrides,
    })


def decision(**overrides):
    return OptionDecision.model_validate({
        "reviews": [{"option": letter, "reason": "复核更正初判"} for letter in "ABCD"],
        "answer": "D", "rationale": "例外导致其余选项不成立",
        "counterargument": "A 不满足特殊条件",
        **overrides,
    })


def covered(state):
    for letter in "ABCD":
        state = assess_option(state, assessment(letter))
    return state


def test_coverage_and_audit_gates_and_immutable_snapshots(state):
    original = state.model_dump()
    with pytest.raises(InvalidTransitionError):
        audit_options(state, decision())
    with pytest.raises(InvalidTransitionError):
        assess_option(state, assessment("B"))
    assert determine_allowed_operations(state, workflow="verified") == (ActionName.ASSESS_OPTION,)
    all_options = covered(state)
    assert determine_allowed_operations(all_options, workflow="verified") == (ActionName.AUDIT_OPTIONS,)
    revised = audit_options(all_options, decision())
    assert all_options.option_decision is None
    assert all(r.verdict == "meets_question" for r in revised.option_assessments)
    assert revised.option_decision.reviews[0].reason == "复核更正初判"
    assert all(issue.status.value == "resolved" for issue in revised.issues)
    assert "最终选择：D" in revised.conclusions[-1].content
    assert determine_allowed_operations(revised, workflow="verified") == (ActionName.STOP,)
    assert state.model_dump() == original
    with pytest.raises(InvalidTransitionError):
        audit_options(revised, decision())


@pytest.mark.parametrize("reference", ["missing", "F3", "I1"])
def test_condition_references_reject_unknown_or_other_option_material(state, reference):
    item = assessment("A", conditions=[{"condition": "条件", "evidence": [reference], "finding": "对应"}])
    with pytest.raises(ValidationError):
        assess_option(state, item)
    assert state.option_assessments == []


def test_rules_cannot_be_labelled_provided_without_knowledge(state):
    with pytest.raises(ValidationError, match="No provided knowledge"):
        assess_option(state, assessment("A", rule_source="provided_knowledge"))


@pytest.mark.parametrize("overrides", [
    {"answer": "E"}, {"counterargument": ""},
    {"reviews": [{"option": "A", "reason": "未判定"}]*4},
])
def test_audit_rejects_missing_coverage_or_invalid_decision(overrides):
    with pytest.raises(ValidationError):
        decision(**overrides)


def test_assessment_prompt_omits_previous_verdicts_and_other_materials(state):
    state = assess_option(state, assessment("A", rule="SECRET_PRIOR_VERDICT"))
    prompt = build_assessment_prompt("原题", state)
    data = json.loads(prompt.split("输入数据：\n")[1])
    assert data["current_option"] == "B"
    assert {m["id"] for m in data["materials"]} == {"F1", "F3"}
    assert "SECRET_PRIOR_VERDICT" not in prompt


class Client:
    max_output_tokens = 512

    def __init__(self, actions):
        self.actions = iter(actions)
        self.requests = []
        self.formats = []

    def generate(self, prompt, *, max_output_tokens=None, response_format=None):
        self.requests.append((prompt, max_output_tokens))
        self.formats.append(response_format)
        value = next(self.actions)
        text = value if isinstance(value, str) else json.dumps(value)
        return ModelCallResult(raw_text=text, model="scripted",
                               input_tokens=100, output_tokens=10, latency_seconds=0,
                               finish_reason="stop")


def actions():
    audited = decision().model_dump()
    for review in audited["reviews"]:
        review["verdict"] = "meets_question" if review["option"] == "D" else "does_not_meet"
    return [*[{"operation": "ASSESS_OPTION", "assessment": assessment(l).model_dump()}
              for l in "ABCD"],
            {"operation": "AUDIT_OPTIONS", "decision": audited},
            {"operation": "STOP"}]


def test_verified_runner_preserves_revision_and_budget(state):
    client = Client(actions())
    result = run_legal_state("材料", "原题", state, client, 24,
                             workflow="verified", max_reasoning_output_tokens=8192)
    assert result.termination_reason.value == "stop"
    assert len(result.steps) == 6
    assert result.steps[4].before_state.option_decision is None
    assert result.steps[4].after_state.option_decision.answer == "D"
    limited = run_legal_state("材料", "原题", state, Client(actions()), 24,
                              workflow="verified", max_reasoning_output_tokens=20)
    assert limited.termination_reason.value == "token_budget"
    assert limited.final_state.option_decision is None
    assert len(limited.steps) == 2


def test_verified_runner_rejects_early_stop_and_keeps_failure(state):
    client = Client([{"operation": "STOP"}])
    with pytest.raises(RunnerStepError) as caught:
        run_legal_state("材料", "原题", state, client, 24, workflow="verified")
    assert caught.value.failed_step.stage == "check_allowed_operation"
    assert len(client.requests) == 1


def test_new_actions_use_strict_parser_and_existing_executor(state):
    action = parse_action_json(json.dumps(actions()[0]))
    assert apply_action(state, action).option_assessments[0].option == "A"
    payload = actions()[0]
    payload["assessment"]["conditions"][0]["evidence"] = [1]
    with pytest.raises(ValidationError):
        parse_action_json(json.dumps(payload))


def test_verified_configuration_requires_scoped_materials():
    with pytest.raises(ValidationError, match="requires scoped_options"):
        ExperimentConfig(legal_state_workflow="verified")


def test_experiment_transcribes_only_audited_decision_and_replays_without_calls(tmp_path):
    config = ExperimentConfig(legal_state_workflow="verified",
                              legal_state_materials="scoped_options",
                              legal_state_constrained_json=True)
    case = CaseInput(case_id="lawbench-3-6-000000", source_index=0,
                     stem="原题", question="原题A:甲B:乙C:丙D:丁",
                     options=dict(zip("ABCD", "甲乙丙丁", strict=True)))
    base = Client([*actions(), {"answer": "D"}])
    journal_path = tmp_path / "calls.jsonl"
    first = run_case(case, "legal_state", config,
                     JournalClient(base, config, journal_path, "scripted"))
    assert first["status"] == "completed"
    artifact = json.loads(first["reasoning_artifact"])
    assert artifact["answer"] == "D"
    assert "待复核初判" not in first["reasoning_artifact"]
    assert first["final_state"]["option_assessments"][0]["verdict"] == "meets_question"
    assert artifact["reviews"][0]["reason"] == "复核更正初判"
    second = run_case(case, "legal_state", config,
                      JournalClient(base, config, journal_path, "scripted"))
    assert second["status"] == "completed"
    assert second["final_answer"] == first["final_answer"] == "D"
    assert len(base.requests) == 7
    assert all(f["type"] == "json_schema" for f in base.formats[:6])
    assert base.formats[-1] is None
    replay = JournalClient(base, config, journal_path, "scripted")
    with pytest.raises(ValueError, match="Replay prompt/budget changed"):
        replay.generate(base.requests[0][0], max_output_tokens=512,
                        response_format={"type": "json_object"})


def test_truncated_assessment_remains_failed_without_final_answer_or_retry(state):
    class TruncatedClient(Client):
        def generate(self, prompt, *, max_output_tokens=None):
            result = super().generate(prompt, max_output_tokens=max_output_tokens)
            return result.model_copy(update={"finish_reason": "length"})

    client = TruncatedClient(actions())
    with pytest.raises(RunnerStepError) as caught:
        run_legal_state("材料", "原题", state, client, 24, workflow="verified")
    assert caught.value.failed_step.finish_reason == "length"
    assert caught.value.failed_step.stage == "model_call"
    assert caught.value.completed_steps == ()
    assert len(client.requests) == 1


def test_missing_condition_evidence_stays_explicit_without_invented_references(state):
    item = assessment("A")
    payload = item.model_dump()
    payload["conditions"].append({"condition": "例外是否有证据", "evidence": [],
                                  "finding": "题面没有提供，不能断定"})
    item = OptionAssessment.model_validate(payload)
    updated = assess_option(state, item)
    assert updated.option_assessments[0].conditions[-1].evidence == []
    assert updated.relations[0].fact_ids == ["F2"]
    payload["conditions"][0]["evidence"] = []
    with pytest.raises(ValidationError, match="at least one material"):
        OptionAssessment.model_validate(payload)


def test_constrained_json_requires_verified_workflow():
    with pytest.raises(ValidationError, match="verified workflow"):
        ExperimentConfig(legal_state_constrained_json=True)


def test_audit_prompt_preserves_initial_reasoning_and_declared_verdicts(state):
    prompt = build_assessment_prompt("原题", covered(state))
    data = json.loads(prompt.split("输入数据：\n")[1])
    assert len(data["initial_assessments"]) == 4
    assert all(item["verdict"] == "meets_question" for item in data["initial_assessments"])


@pytest.mark.parametrize("candidates,answer", [
    (["C"], "C"), ([], "UNKNOWN"), (["A", "D"], "UNKNOWN"),
])
def test_audit_selection_matches_explicit_reviews_and_keeps_legacy_decisions(candidates, answer):
    reviews = [{"option": letter, "reason": "复核依据",
                "verdict": "meets_question" if letter in candidates else "does_not_meet"}
               for letter in "ABCD"]
    assert decision(reviews=reviews, answer=answer).answer == answer
    wrong = "A" if answer != "A" else "B"
    with pytest.raises(ValidationError, match="unique reviewed candidate"):
        decision(reviews=reviews, answer=wrong)
    assert decision().answer == "D"  # Decisions stored before verdict fields remain readable.


def test_audit_rejects_incomplete_verdicts_and_preserves_input_state(state):
    reviews = decision().model_dump()["reviews"]
    reviews[0]["verdict"] = "does_not_meet"
    original = state.model_dump()
    with pytest.raises(ValidationError, match="cover all four"):
        decision(reviews=reviews)
    assert state.model_dump() == original


def test_constrained_audit_requires_nonnull_verdicts_without_changing_legacy_schema(state):
    schema = assessment_response_format(covered(state))["json_schema"]["schema"]
    review = schema["$defs"]["OptionReview"]
    assert "verdict" in review["required"]
    assert review["properties"]["verdict"]["enum"] == [
        "meets_question", "does_not_meet", "uncertain",
    ]
    assert "verdict" not in OptionDecision.model_json_schema()["$defs"]["OptionReview"]["required"]


def test_inconsistent_audit_fails_with_original_response_and_no_repair_or_retry(state):
    original = state.model_dump()
    bad = decision().model_dump()
    for review in bad["reviews"]:
        review["verdict"] = (
            "meets_question" if review["option"] == "D" else "does_not_meet"
        )
    bad["answer"] = "A"
    client = Client([*actions()[:4], {"operation": "AUDIT_OPTIONS", "decision": bad}])
    with pytest.raises(RunnerStepError) as caught:
        run_legal_state("材料", "原题", state, client, 24, workflow="verified",
                        constrained_json=True, max_reasoning_output_tokens=8192)
    assert caught.value.failed_step.stage == "parse_action"
    assert len(caught.value.completed_steps) == 4
    assert caught.value.failed_step.before_state.option_decision is None
    assert json.loads(caught.value.failed_step.raw_model_output)["decision"]["answer"] == "A"
    assert len(client.requests) == 5
    assert state.model_dump() == original


def test_independent_assessment_only_exposes_stem_and_current_option(state):
    prompt = build_assessment_prompt("原题A:其他选项不应进入此阶段", state)
    data = json.loads(prompt.split("输入数据：\n")[1])
    assert data["question"] == "题干"
    assert data["current_option"] == "A"
    assert "其他选项不应进入此阶段" not in prompt


def hybrid_case():
    return CaseInput(case_id="lawbench-3-6-000000", source_index=0,
                     stem="原题", question="原题A:甲B:乙C:丙D:丁",
                     options=dict(zip("ABCD", "甲乙丙丁", strict=True)))


def hybrid_config(**overrides):
    return ExperimentConfig(legal_state_workflow="verified",
                            legal_state_materials="scoped_options",
                            legal_state_constrained_json=True,
                            legal_state_initial_analysis="cot", **overrides)


def test_hybrid_draft_uses_baseline_input_and_keeps_tentative_provenance(tmp_path):
    case = hybrid_case()
    config = hybrid_config()
    client = Client(["暂定分析选择 A [STOP]", *actions(), {"answer": "D"}])
    record = run_case(case, "legal_state", config,
                      JournalClient(client, config, tmp_path / "calls.jsonl", "scripted"))
    assert record["status"] == "completed"
    assert client.requests[0][0] == baseline_prompt(case, "cot", "")
    assert client.formats[0] is None and client.formats[-1] is None
    assert all(f["type"] == "json_schema" for f in client.formats[1:-1])
    assert record["initial_state"]["draft_reasoning"] == "暂定分析选择 A"
    assert len(record["steps"]) == 6
    assert record["usage"]["call_count"] == 8
    assert record["usage"]["reasoning_output_tokens"] == 70
    assert record["final_answer"] == "D"
    assert not record["final_state"]["knowledge"]
    assert all(f["source"] == "question_material" for f in record["final_state"]["facts"])


@pytest.mark.parametrize("limit,reason,reply_actions", [
    ({"max_steps": 1}, "max_steps", []),
    ({"max_output_tokens_total": 20}, "token_budget", actions()[:1]),
])
def test_hybrid_draft_consumes_shared_step_and_output_budgets(tmp_path, limit, reason, reply_actions):
    config = hybrid_config(reasoning=ReasoningSettings(**limit))
    client = Client(["暂定分析 A [STOP]", *reply_actions, {"answer": "A"}])
    record = run_case(hybrid_case(), "legal_state", config,
                      JournalClient(client, config, tmp_path / "calls.jsonl", "scripted"))
    assert record["status"] == "completed"
    assert record["termination_reason"] == reason
    assert len(record["steps"]) + 1 <= config.reasoning.max_steps
    assert record["usage"]["reasoning_output_tokens"] <= config.reasoning.max_output_tokens_total
    assert record["final_state"].get("option_decision") is None
    assert len(client.requests) == len(reply_actions) + 2


def test_hybrid_requires_verified_workflow():
    with pytest.raises(ValidationError, match="Initial CoT analysis"):
        ExperimentConfig(legal_state_initial_analysis="cot")


def test_rule_recall_is_tentative_and_consumes_the_shared_budget(tmp_path):
    config = hybrid_config().model_copy(update={"legal_state_initial_analysis": "rules"})
    rules = "规则假设：不同程序适用不同条件；例外记忆不确定。"
    client = Client([rules, *actions(), {"answer": "D"}])
    record = run_case(hybrid_case(), "legal_state", config,
                      JournalClient(client, config, tmp_path / "rules.jsonl", "scripted"))
    assert record["status"] == "completed"
    assert "不选择答案" in client.requests[0][0]
    assert client.formats[0] is None
    assert record["draft_step"]["kind"] == "model_rule_recall"
    assert record["initial_state"]["draft_reasoning"] == rules
    assert record["initial_state"]["knowledge"] == []
    assert rules not in [fact["content"] for fact in record["initial_state"]["facts"]]
    assert record["usage"]["call_count"] == 8
    assert record["usage"]["reasoning_output_tokens"] == 70


def test_rule_recall_requires_verified_workflow():
    with pytest.raises(ValidationError, match="rule recall"):
        ExperimentConfig(legal_state_initial_analysis="rules")


def test_draft_is_tentative_context_without_changing_material_evidence(state):
    state = LegalState.model_validate({**state.model_dump(), "draft_reasoning": "PRIVATE_DRAFT"})
    original = state.model_dump()
    data = json.loads(build_assessment_prompt("原题", state).split("输入数据：\n")[1])
    assert data["tentative_draft"] == "PRIVATE_DRAFT"
    assert {m["id"] for m in data["materials"]} == {"F1", "F2"}
    assert data["knowledge"] == []
    with pytest.raises(ValidationError, match="quoted facts"):
        assess_option(state, assessment("A", conditions=[{
            "condition": "条件", "evidence": ["tentative_draft"], "finding": "草稿推断",
        }]))
    prompt = build_assessment_prompt("原题", covered(state))
    data = json.loads(prompt.split("输入数据：\n")[1])
    assert data["tentative_draft"] == "PRIVATE_DRAFT"
    assert state.model_dump() == original


def test_audit_retains_original_scoped_materials_and_date_arithmetic(state):
    original = state.model_dump()
    prompt = build_assessment_prompt("原题", covered(state))
    data = json.loads(prompt.split("输入数据：\n")[1])
    assert data["materials"] == [
        {"id": fact.id, "content": fact.content, "scope": fact.material.scope}
        for fact in state.facts
    ]
    assert data["knowledge"] == []
    assert "quoted_date_intervals" not in data
    dated = state.model_dump()
    dated["facts"][0]["content"] = "2001年2月到2003年2月"
    dated["facts"][0]["material"]["end"] = len(dated["facts"][0]["content"])
    dated_state = LegalState.model_validate(dated)
    prompt = build_assessment_prompt("原题", covered(dated_state))
    data = json.loads(prompt.split("输入数据：\n")[1])
    assert data["quoted_date_intervals"][0]["calendar_month_difference"] == 24
    assert data["quoted_date_intervals"][0]["from"]["evidence"] == ["F1"]
    assert state.model_dump() == original


def test_quoted_date_intervals_preserve_month_precision_and_provenance():
    result = material_date_intervals([
        {"id": "F1", "content": "2003年3月1日申请，2005年3月签约；2003年3月。"},
        {"id": "F2", "content": "2013年3月之后，另有2005年3月、2014年13月。"},
    ])
    assert len(result) == 3
    ten_years = result[1]
    assert ten_years["calendar_month_difference"] == 120
    assert ten_years["from"] == {"text": "2003年3月1日", "evidence": ["F1"]}
    assert ten_years["to"] == {"text": "2013年3月", "evidence": ["F2"]}
    assert result[0]["to"]["evidence"] == ["F1", "F2"]
    assert material_date_intervals([{"id": "F1", "content": "无日期"}]) == []


def test_assessment_grammar_requires_grounded_first_condition_and_scoped_ids(state):
    schema = assessment_response_format(state)["json_schema"]["schema"]
    fields = schema["$defs"]["OptionAssessment"]["properties"]
    assert fields["option"] == {"type": "string", "const": "A"}
    first = fields["conditions"]["prefixItems"][0]["properties"]["evidence"]
    assert first["minItems"] == 1
    assert first["items"]["enum"] == ["F1", "F2"]
    other = schema["$defs"]["RuleCondition"]["properties"]["evidence"]
    assert "minItems" not in other  # Missing later conditions can still use [].
    assert other["items"]["enum"] == ["F1", "F2"]
    updated = assess_option(state, assessment("A"))
    next_fields = assessment_response_format(updated)["json_schema"]["schema"]["$defs"]["OptionAssessment"]["properties"]
    assert next_fields["option"]["const"] == "B"
    assert next_fields["conditions"]["prefixItems"][0]["properties"]["evidence"]["items"]["enum"] == ["F1", "F3"]


def test_comparison_context_keeps_claims_separate_from_usable_evidence(state):
    original = state.model_dump()
    data = json.loads(build_assessment_prompt("原题", state, compare_options=True).split("输入数据：\n")[1])
    assert data["comparison_options"] == dict(zip("ABCD", "ABCD", strict=True))
    assert {material["id"] for material in data["materials"]} == {"F1", "F2"}
    with pytest.raises(ValidationError, match="crosses option scopes"):
        assess_option(state, assessment("A", conditions=[{
            "condition": "其他选项假设", "evidence": ["F3"], "finding": "不能作为本选项事实",
        }]))
    assert state.model_dump() == original


def test_option_comparison_requires_verified_workflow():
    with pytest.raises(ValidationError, match="Option comparison"):
        ExperimentConfig(legal_state_compare_options=True)


def test_calendar_intervals_never_join_different_hypothetical_cases():
    from legal_state.assessment import scoped_material_date_intervals

    result = scoped_material_date_intervals([
        {"id": "F1", "content": "判断四个独立案情", "scope": None},
        {"id": "F2", "content": "2001年3月到2002年3月", "scope": "A"},
        {"id": "F3", "content": "2011年3月到2012年3月", "scope": "B"},
    ])
    assert len(result) == 2
    assert {item["scope"] for item in result} == {"A", "B"}
    assert {item["calendar_month_difference"] for item in result} == {12}
    assert result[0]["from"]["evidence"] == result[0]["to"]["evidence"] == ["F2"]
    assert result[1]["from"]["evidence"] == result[1]["to"]["evidence"] == ["F3"]
