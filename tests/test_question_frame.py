"""Source scopes, transition gates and shared budgets in issue planning."""

import json

import pytest
from pydantic import ValidationError

from legal_state.actions import ActionName, parse_action_json
from legal_state.assessment import (
    assessment_response_format, build_assessment_prompt, frame_question, material_quote_candidates,
)
from legal_state.executor import apply_action
from legal_state.experiment.config import ExperimentConfig, ModelSettings, ReasoningSettings
from legal_state.experiment.data import CaseInput
from legal_state.experiment.engine import JournalClient, run_case
from legal_state.experiment.materials import scoped_material_state
from legal_state.experiment.storage import checked_preflight, preflight
from legal_state.operations import InvalidTransitionError
from legal_state.runner import determine_allowed_operations
from legal_state.schemas import QuestionFrame
from test_assessment import Client, actions, assessment


@pytest.fixture
def case():
    return CaseInput(case_id="lawbench-3-6-000000", source_index=0, stem="题干",
                     question="题干A:甲B:乙C:丙D:丁",
                     options=dict(zip("ABCD", "甲乙丙丁", strict=True)))


def frame():
    return QuestionFrame(question_type="correct", checks=[
        {"trigger_quote": "题干", "question": "题干中的条件是否改变规则分支？",
         "evidence": ["F1"], "scope": None},
        {"trigger_quote": "甲", "question": "甲的特定条件是否满足？",
         "evidence": ["F1", "F2"], "scope": "A"},
        {"trigger_quote": "乙", "question": "乙的特定条件是否满足？", "evidence": ["F3"], "scope": "B"},
    ])


def test_frame_is_immutable_planning_metadata_with_stage_gates(case):
    state = scoped_material_state(case)
    original = state.model_dump()
    assert determine_allowed_operations(state, workflow="verified", require_question_frame=True) == (
        ActionName.FRAME_QUESTION,
    )
    action = parse_action_json(json.dumps({"operation": "FRAME_QUESTION", "frame": frame().model_dump()}))
    updated = apply_action(state, action)
    assert updated.question_frame == frame()
    assert updated.facts == state.facts and updated.knowledge == []
    assert updated.issues == [] and updated.conclusions == []
    assert state.model_dump() == original
    assert determine_allowed_operations(updated, workflow="verified", require_question_frame=True) == (
        ActionName.ASSESS_OPTION,
    )
    with pytest.raises(InvalidTransitionError, match="once"):
        frame_question(updated, frame())


@pytest.mark.parametrize("scope,evidence", [(None, ["F2"]), ("A", ["F3"]), ("A", ["draft_reasoning"])])
def test_frame_rejects_unknown_or_cross_scope_references_without_mutation(case, scope, evidence):
    state = scoped_material_state(case)
    original = state.model_dump()
    item = QuestionFrame(question_type="correct", checks=[
        {"trigger_quote": "题干", "question": "待核查", "scope": scope, "evidence": evidence},
    ])
    with pytest.raises(ValidationError):
        frame_question(state, item)
    assert state.model_dump() == original


def test_planner_does_not_see_draft_and_assessment_only_sees_relevant_checks(case):
    state = scoped_material_state(case)
    state.draft_reasoning = "PRIVATE_DRAFT"
    prompt = build_assessment_prompt(case.question, state, require_question_frame=True)
    assert "PRIVATE_DRAFT" not in prompt
    updated = frame_question(state, frame())
    prompt = build_assessment_prompt(case.question, updated, require_question_frame=True)
    data = json.loads(prompt.split("输入数据：\n")[1])
    assert {check["scope"] for check in data["question_frame"]["checks"]} == {None, "A"}
    assert {material["id"] for material in data["materials"]} == {"F1", "F2"}
    assert data["knowledge"] == []


def test_framed_engine_records_plan_replays_and_debits_shared_budget(case, tmp_path):
    config = ExperimentConfig(legal_state_workflow="verified", legal_state_materials="scoped_options",
                              legal_state_constrained_json=True, legal_state_question_frame=True)
    replies = [{"operation": "FRAME_QUESTION", "frame": frame().model_dump()},
               *actions(), {"answer": "D"}]
    base = Client(replies)
    path = tmp_path / "calls.jsonl"
    result = run_case(case, "legal_state", config, JournalClient(base, config, path, "scripted"))
    assert result["status"] == "completed"
    assert result["steps"][0]["parsed_action"]["operation"] == "FRAME_QUESTION"
    assert result["final_state"]["question_frame"] == frame().model_dump()
    assert result["usage"]["reasoning_output_tokens"] == 70
    assert result["usage"]["call_count"] == 8
    assert result["final_answer"] == "D"
    replay = run_case(case, "legal_state", config, JournalClient(Client([]), config, path, "scripted"))
    assert replay["final_state"] == result["final_state"]
    limited = config.model_copy(update={"reasoning": ReasoningSettings(max_output_tokens_total=10)})
    base = Client([replies[0], {"answer": "UNKNOWN"}])
    result = run_case(case, "legal_state", limited,
                      JournalClient(base, limited, tmp_path / "limited.jsonl", "scripted"))
    assert result["termination_reason"] == "token_budget"
    assert len(result["steps"]) == 1
    assert result["final_state"].get("option_decision") is None


def test_question_frame_requires_verified_configuration():
    with pytest.raises(ValidationError, match="verified"):
        ExperimentConfig(legal_state_question_frame=True)


def test_frame_generation_contract_limits_each_scope_to_its_quote_ids(case):
    state = scoped_material_state(case)
    schema = assessment_response_format(state, require_question_frame=True)["json_schema"]["schema"]
    branches = schema["$defs"]["FramedCheck"]["anyOf"]
    refs = {branch["properties"]["scope"]["const"]:
            branch["properties"]["evidence"]["items"]["enum"] for branch in branches}
    assert refs == {None: ["F1"], "A": ["F1", "F2"], "B": ["F1", "F3"],
                    "C": ["F1", "F4"], "D": ["F1", "F5"]}
    assert all("scope" in branch["required"] for branch in branches)
    assert all("trigger_quote" in branch["required"] for branch in branches)
    for branch in branches:
        if "甲" in branch["properties"]["trigger_quote"]["enum"]:
            assert all("F2" in refs for refs in branch["properties"]["evidence"]["enum"])
    # The unconstrained state schema keeps validating stored snapshots normally.
    assert "anyOf" not in state.model_json_schema()["$defs"]["FramedCheck"]


def test_quote_candidates_are_verbatim_and_preserve_numeric_whitespace():
    content = "支付5000元，乙公司否认将房产抵押给丙银行。" + "长原文" * 30
    quotes = material_quote_candidates([{"id": "F1", "content": content}])["F1"]
    assert "支付5000元" in quotes
    assert "乙公司否认将房产抵押给丙银行" in quotes
    assert all(quote in content and len(quote) <= 32 for quote in quotes)
    assert len(quotes) == len(set(quotes))


def test_frame_rejects_invented_quote_but_retains_legacy_snapshots(case):
    state = scoped_material_state(case)
    original = state.model_dump()
    bad = frame().model_dump()
    bad["checks"][0]["trigger_quote"] = "原文没有的事实"
    with pytest.raises(ValidationError, match="verbatim"):
        frame_question(state, QuestionFrame.model_validate(bad))
    assert state.model_dump() == original
    legacy = frame().model_dump()
    for check in legacy["checks"]:
        del check["trigger_quote"]
    snapshot = type(state).model_validate({**original, "question_frame": legacy})
    assert snapshot.question_frame.checks[0].trigger_quote is None
    with pytest.raises(InvalidTransitionError, match="verbatim"):
        frame_question(state, QuestionFrame.model_validate(legacy))


def test_preflight_probes_actual_framing_grammar_and_rejects_invented_quotes(monkeypatch, tmp_path):
    from pathlib import Path

    monkeypatch.setenv("LSP_MODEL", "scripted")
    monkeypatch.setenv("LSP_ENDPOINT", "http://local.test/v1/chat/completions")
    config = ExperimentConfig(legal_state_workflow="verified", legal_state_materials="scoped_options",
                              legal_state_constrained_json=True, legal_state_question_frame=True,
                              runs_dir=str(tmp_path), model=ModelSettings(hidden_reasoning_disabled=True))
    action = {"operation": "FRAME_QUESTION", "frame": {"question_type": "other", "checks": [
        {"trigger_quote": "测试题干", "question": "需要核查何种条件？", "evidence": ["F1"], "scope": None},
    ]}}
    assess = {"operation": "ASSESS_OPTION", "assessment": assessment("A").model_dump()}
    client = Client([{"answer": "A"}, action, assess])
    result = preflight(config, Path.cwd(), client)
    assert result["status"] == "passed" and len(client.requests) == 3
    assert result["result"]["raw_text"] == '{"answer": "A"}'
    assert result["framing_probe"]["result"]["raw_text"] == json.dumps(action)
    assert client.formats[1]["json_schema"]["schema"]["$defs"]["FramedCheck"]["anyOf"]
    unconstrained = config.model_copy(update={"legal_state_constrained_json": False})
    client = Client([{"answer": "A"}, action, assess])
    result = preflight(unconstrained, Path.cwd(), client)
    assert result["status"] == "passed" and client.formats == [None, None, None]
    assert result["framing_probe"]["response_format"] is None
    action["frame"]["checks"][0]["trigger_quote"] = "不在原文的内容"
    with pytest.raises(ValueError, match="verbatim"):
        preflight(config, Path.cwd(), Client([{"answer": "A"}, action]))
    with pytest.raises(ValueError, match="latest preflight failed"):
        checked_preflight(config, Path.cwd())


@pytest.mark.parametrize("bad_case", ["empty", "other_option", "wrong_letter"])
def test_preflight_validates_the_assessment_probe_without_retry(monkeypatch, tmp_path, bad_case):
    from pathlib import Path

    monkeypatch.setenv("LSP_MODEL", "scripted")
    monkeypatch.setenv("LSP_ENDPOINT", "http://local.test/v1/chat/completions")
    config = ExperimentConfig(legal_state_workflow="verified", legal_state_materials="scoped_options",
                              legal_state_constrained_json=True, legal_state_question_frame=True,
                              runs_dir=str(tmp_path), model=ModelSettings(hidden_reasoning_disabled=True))
    framing = {"operation": "FRAME_QUESTION", "frame": {"question_type": "other", "checks": [
        {"trigger_quote": "测试题干", "question": "需要核查何种条件？", "evidence": ["F1"], "scope": None},
    ]}}
    assessed = assessment("A").model_dump()
    if bad_case == "wrong_letter":
        assessed["option"] = "B"
    else:
        assessed["conditions"][0]["evidence"] = [] if bad_case == "empty" else ["F3"]
    client = Client([{"answer": "A"}, framing, {"operation": "ASSESS_OPTION", "assessment": assessed}])
    with pytest.raises(ValueError, match="Preflight failed"):
        preflight(config, Path.cwd(), client)
    assert len(client.requests) == 3
    with pytest.raises(ValueError, match="latest preflight failed"):
        checked_preflight(config, Path.cwd())


def test_compact_json_requires_schema_decoding():
    with pytest.raises(ValueError, match="Compact JSON requires constrained JSON"):
        ExperimentConfig(legal_state_compact_json=True)
    with pytest.raises(ValueError, match="Compact audit scope requires compact JSON"):
        ExperimentConfig(legal_state_compact_json_scope="audit")


def test_compact_preflight_exercises_the_nested_audit_schema(monkeypatch, tmp_path):
    from pathlib import Path

    monkeypatch.setenv("LSP_MODEL", "scripted")
    monkeypatch.setenv("LSP_ENDPOINT", "http://local.test/v1/chat/completions")
    config = ExperimentConfig(legal_state_workflow="verified", legal_state_materials="scoped_options",
                              legal_state_constrained_json=True, legal_state_question_frame=True,
                              legal_state_compact_json=True, runs_dir=str(tmp_path),
                              model=ModelSettings(hidden_reasoning_disabled=True))
    framing = {"operation": "FRAME_QUESTION", "frame": {"question_type": "other", "checks": [
        {"trigger_quote": "测试题干", "question": "需要核查何种条件？", "evidence": ["F1"], "scope": None},
    ]}}
    client = Client([json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                     for value in [{"answer": "A"}, framing, actions()[0], actions()[4]]])
    result = preflight(config, Path.cwd(), client)
    assert result["status"] == "passed" and len(client.requests) == 4
    assert json.loads(result["audit_probe"]["result"]["raw_text"])["operation"] == "AUDIT_OPTIONS"
    assert client.formats[3]["json_schema"]["schema"]["properties"]["operation"]["const"] == "AUDIT_OPTIONS"
