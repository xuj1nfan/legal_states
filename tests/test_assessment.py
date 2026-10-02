"""Coverage, evidence integrity and revision in the verified workflow."""

import json

import pytest
from pydantic import ValidationError

from legal_state.actions import ActionName, parse_action_json
from legal_state.assessment import (
    assess_option, audit_options, build_assessment_prompt, material_date_intervals,
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
    return [*[{"operation": "ASSESS_OPTION", "assessment": assessment(l).model_dump()}
              for l in "ABCD"],
            {"operation": "AUDIT_OPTIONS", "decision": decision().model_dump()},
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


def test_audit_prompt_keeps_reasoning_without_initial_verdict_labels(state):
    prompt = build_assessment_prompt("原题", covered(state))
    data = json.loads(prompt.split("输入数据：\n")[1])
    assert len(data["initial_assessments"]) == 4
    assert all("verdict" not in item for item in data["initial_assessments"])


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


def test_draft_is_reserved_for_audit_after_independent_assessments(state):
    state = LegalState.model_validate({**state.model_dump(), "draft_reasoning": "PRIVATE_DRAFT"})
    assert "PRIVATE_DRAFT" not in build_assessment_prompt("原题", state)
    prompt = build_assessment_prompt("原题", covered(state))
    data = json.loads(prompt.split("输入数据：\n")[1])
    assert data["tentative_draft"] == "PRIVATE_DRAFT"


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
