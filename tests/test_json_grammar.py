"""Check action languages with an independent parser, without model credentials."""

import json
import re

import pytest
from lark import Lark, UnexpectedInput

from legal_state.assessment import assessment_response_format
from legal_state.json_grammar import compact_json_grammar, has_formatting_whitespace
from test_assessment import assessment, state


def parser(schema):
    grammar = compact_json_grammar(schema)
    lines = [line.replace("::=", ":").replace("json-char", "json_char")
             for line in grammar.splitlines() if not line.startswith("json-char ::=")]
    grammar = "\n".join(lines)
    grammar = re.sub(r"json_char\{(\d+),(\d+)\}", r"json_char~\1..\2", grammar)
    # Lark uses regex terminals for character classes; the production compiler
    # emits the corresponding EBNF character classes for vLLM/xgrammar.
    grammar += '\n' + r'json_char: /[^"\\\x00-\x1f]/ | "\\" (/["\\\/bfnrt]/ | "u" /[0-9a-fA-F]/~4)'
    return Lark(grammar, start="root", parser="earley")


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def test_strings_keep_content_but_disallow_json_formatting_whitespace():
    grammar = parser({"type": "object", "properties": {"text": {"type": "string", "minLength": 1}},
                      "required": ["text"], "additionalProperties": False})
    for text in ('汉字 含空格', '引号"和斜杠\\和换行\n', '😀'):
        grammar.parse(compact({"text": text}))
    for text in ('{"text":""}', '{ "text":"内容"}', '{"text":"内容"}\n',
                 '{"text":"内容"', '{"text":"内容","extra":true}'):
        with pytest.raises(UnexpectedInput):
            grammar.parse(text)


def test_generated_assessment_language_enforces_scope_and_grounding(state):
    schema = assessment_response_format(state)["json_schema"]["schema"]
    grammar = parser(schema)
    good = {"operation": "ASSESS_OPTION", "assessment": assessment("A").model_dump()}
    grammar.parse(compact(good))
    later = json.loads(compact(good))
    later["assessment"]["conditions"].append({"condition": "缺失条件", "evidence": [], "finding": "原文未给出"})
    grammar.parse(compact(later))
    for field, value in (("option", "B"), ("evidence", []), ("evidence", ["F3"]),
                         ("evidence", ["unknown"])):
        bad = json.loads(compact(good))
        target = bad["assessment"] if field == "option" else bad["assessment"]["conditions"][0]
        target[field] = value
        with pytest.raises(UnexpectedInput):
            grammar.parse(compact(bad))


def test_framing_language_requires_verbatim_quote_owner_and_scope(state):
    schema = assessment_response_format(state, require_question_frame=True)["json_schema"]["schema"]
    grammar = parser(schema)
    good = {"operation": "FRAME_QUESTION", "frame": {"question_type": "correct", "checks": [
        {"trigger_quote": "A", "question": "检查该选项？", "evidence": ["F2"], "scope": "A"},
    ]}}
    grammar.parse(compact(good))
    for field, value in (("trigger_quote", "原文没有"), ("evidence", ["F1"]),
                         ("scope", "B"), ("evidence", ["F3"])):
        bad = json.loads(compact(good))
        bad["frame"]["checks"][0][field] = value
        with pytest.raises(UnexpectedInput):
            grammar.parse(compact(bad))
    bad = json.loads(compact(good))
    bad["frame"]["checks"] *= 4
    with pytest.raises(UnexpectedInput):
        grammar.parse(compact(bad))


@pytest.mark.parametrize("schema", [
    {"type": "number"}, {"type": "string", "pattern": "x"},
    {"type": "object", "additionalProperties": True},
])
def test_unsupported_constraints_fail_before_generation(schema):
    with pytest.raises(ValueError, match="Unsupported|requires closed"):
        compact_json_grammar(schema)


def test_whitespace_probe_distinguishes_strings_from_json_formatting():
    assert not has_formatting_whitespace(compact({"text": '引号" 后的 空格\\\n'}))
    assert has_formatting_whitespace('{"answer":"D"}\n}')
    assert has_formatting_whitespace('{ "answer":"D"}')
