"""Quoted material provenance and isolation of hypothetical option contexts."""

import json

import pytest
from pydantic import ValidationError

from legal_state.actions import parse_action_json
from legal_state.executor import apply_action
from legal_state.experiment.config import ExperimentConfig
from legal_state.experiment.data import CaseInput
from legal_state.experiment.materials import scoped_material_state
from legal_state.operations import bind_fact, commit, expand_issue
from legal_state.prompts import build_action_prompt
from legal_state.schemas import Fact, LegalState


@pytest.fixture
def case() -> CaseInput:
    stem = " 以下哪项成立？\n"
    options = {
        "A": "甲的假设🙂。 ",
        "B": "乙的假设🙂。 ",
        "C": "规则待验证。",
        "D": "甲的假设🙂。 ",
    }
    return CaseInput(
        case_id="lawbench-3-6-000000",
        source_index=0,
        stem=stem,
        question=stem + "".join(f"{letter}:{options[letter]}" for letter in "ABCD"),
        options=options,
    )


def test_materials_preserve_exact_unicode_source_spans_and_option_identity(case):
    snapshot = case.model_dump()
    state = scoped_material_state(case)
    assert [fact.id for fact in state.facts] == ["F1", "F2", "F3", "F4", "F5"]
    assert [fact.material.scope for fact in state.facts] == [None, "A", "B", "C", "D"]
    for fact in state.facts:
        span = fact.material
        assert span.kind == "quoted_material"
        assert case.question[span.start : span.end] == fact.content
    assert state.facts[1].content == state.facts[4].content
    assert state.facts[1].material.start != state.facts[4].material.start
    assert not state.knowledge  # No inferred or retrieved law is added.
    assert case.model_dump() == snapshot
    case.options["A"] = "corrupted input"
    with pytest.raises(ValidationError):
        scoped_material_state(case)


@pytest.mark.parametrize("start,end,content", [(2, 1, "x"), (0, 2, "x"), (-1, 1, "xx")])
def test_invalid_material_spans_are_rejected(start, end, content):
    with pytest.raises(ValidationError):
        Fact(
            id="F1",
            content=content,
            source="input",
            material={"start": start, "end": end},
        )


def test_legacy_payloads_and_config_remain_unchanged():
    assert Fact(id="F1", content="材料", source="case").model_dump() == {
        "id": "F1",
        "content": "材料",
        "source": "case",
    }
    action = parse_action_json('{"operation":"EXPAND_ISSUE","question":"争点"}')
    state = apply_action(LegalState(), action)
    assert "scope" not in state.issues[0].model_dump()
    assert "legal_state_materials" not in ExperimentConfig().model_dump()
    assert (
        ExperimentConfig(legal_state_materials="scoped_options").model_dump()[
            "legal_state_materials"
        ]
        == "scoped_options"
    )


def test_scoped_actions_allow_shared_material_and_preserve_snapshots(case):
    state = scoped_material_state(case)
    original = state.model_dump()
    expanded = apply_action(
        state,
        parse_action_json(
            '{"operation":"EXPAND_ISSUE","question":"判断 A","scope":"A"}'
        ),
    )
    bound = bind_fact(expanded, "I1", ["F1", "F2"])
    concluded = commit(bound, "I1", "A 的判断及理由", ["F1", "F2"])
    assert concluded.support_scopes(["C1"]) == {"A"}
    assert concluded.allowed_support_ids("I1") == ["F1", "F2", "C1"]
    assert state.model_dump() == original
    assert expanded.issues[0].status.value == "open"
    assert not bound.conclusions


@pytest.mark.parametrize(
    "operation", ["bind", "commit", "load_relation", "load_conclusion"]
)
def test_other_option_references_fail_in_operations_and_loaded_states(case, operation):
    state = bind_fact(
        expand_issue(scoped_material_state(case), "判断 A", scope="A"), "I1", ["F2"]
    )
    snapshot = state.model_dump()
    with pytest.raises(ValidationError, match="crosses option scopes"):
        if operation == "bind":
            bind_fact(state, "I1", ["F3"])
        elif operation == "commit":
            commit(state, "I1", "错误地引用 B", ["F3"])
        else:
            data = state.model_dump()
            if operation == "load_relation":
                data["relations"][0]["fact_ids"] = ["F3"]
            else:
                data["conclusions"] = [
                    {"id": "C1", "issue_id": "I1", "content": "判断", "support": ["F3"]}
                ]
            LegalState.model_validate(data)
    assert state.model_dump() == snapshot


def test_scope_cannot_be_laundered_through_aggregate_conclusions(case):
    state = expand_issue(scoped_material_state(case), "比较各选项")
    state = commit(bind_fact(state, "I1", ["F3"]), "I1", "B 的判断", ["F3"])
    state = commit(state, "I1", "依赖 B 的后续判断", ["C1"])
    state = bind_fact(expand_issue(state, "判断 A", scope="A"), "I2", ["F2"])
    assert "C2" not in state.allowed_support_ids("I2")
    with pytest.raises(ValidationError, match="crosses option scopes"):
        commit(state, "I2", "不应引用 B 的间接结论", ["C2"])
    # An aggregate comparison can cite multiple options with their provenance intact.
    aggregate = commit(state, "I1", "比较 A 和 B", ["F2", "F3"])
    assert aggregate.support_scopes(["C3"]) == {"A", "B"}


def test_issue_scope_is_preserved_even_when_conclusion_has_empty_support(case):
    state = expand_issue(scoped_material_state(case), "判断 B", scope="B")
    state = commit(bind_fact(state, "I1", ["F3"]), "I1", "未提供依据", [])
    state = bind_fact(expand_issue(state, "判断 A", scope="A"), "I2", ["F2"])
    with pytest.raises(ValidationError, match="crosses option scopes"):
        commit(state, "I2", "引用另一选项结论", ["C1"])


def test_children_inherit_scope_and_cannot_escape_it(case):
    state = expand_issue(scoped_material_state(case), "判断 A", scope="A")
    child = expand_issue(state, "A 的子争点", parent_issue="I1")
    assert child.issues[1].scope == "A"
    with pytest.raises(ValidationError, match="parent's scope"):
        expand_issue(state, "错误子争点", parent_issue="I1", scope="B")
    with pytest.raises(ValidationError, match="unknown material scope"):
        expand_issue(state, "不存在的选项", scope="invented")
    data = child.model_dump()
    data["issues"][1].pop("scope")
    with pytest.raises(ValidationError, match="parent's scope"):
        LegalState.model_validate(data)


def test_prompt_exposes_scoped_whitelist_without_changing_legacy_formats(case):
    state = expand_issue(scoped_material_state(case), "判断 A", scope="A")
    prompt = build_action_prompt(
        case.stem, case.question, state, ["EXPAND_ISSUE", "BIND_FACT"]
    )
    section = (
        prompt.split("以下按争点的白名单", 1)[1].split("\n", 1)[1].split("\n\n", 1)[0]
    )
    assert json.loads(section)["I1"] == {
        "BIND_FACT.fact_ids": ["F1", "F2"],
        "COMMIT.support": ["F1", "F2"],
    }
    assert "不是已证事实" in prompt
    input_data = json.loads(prompt.split("输入数据：\n", 1)[1])
    assert input_data["question"] == case.question
    assert input_data["legal_state"] == state.model_dump(mode="json")
    legacy = build_action_prompt(
        case.stem, case.question, LegalState(), ["EXPAND_ISSUE"]
    )
    assert '"scope"' not in legacy
