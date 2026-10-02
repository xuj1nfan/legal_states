import csv
import json
from pathlib import Path

import pytest

from legal_state.experiment.__main__ import main
from legal_state.experiment.config import ExperimentConfig, ModelSettings
from legal_state.experiment.data import dataset_fingerprint, prepare, split
from legal_state.experiment.engine import JournalClient, run_case
from legal_state.experiment.evaluation import choice_judge, paired_comparison, score
from legal_state.experiment.io import (
    append_event,
    digest,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from legal_state.experiment.report import report
from legal_state.experiment.review import export_review, import_review
from legal_state.experiment.storage import freeze, preflight, run_batch
from legal_state.model import ModelCallResult


def call(
    text: str, *, tokens: int = 10, finish: str = "stop", model: str = "fixture-model"
) -> ModelCallResult:
    return ModelCallResult(
        raw_text=text,
        model=model,
        input_tokens=100,
        output_tokens=tokens,
        latency_seconds=0.01,
        finish_reason=finish,
        reasoning_tokens=0,
    )


class ScriptedClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def generate(self, prompt, *, max_output_tokens=None):
        self.requests.append((prompt, max_output_tokens))
        response = next(self.replies)
        if isinstance(response, BaseException):
            raise response
        return response


class FullClient:
    """Exercise the actual prompts, parsing and state transitions without a network."""

    def __init__(self):
        self.requests = []

    def generate(self, prompt, *, max_output_tokens=None):
        self.requests.append((prompt, max_output_tokens))
        assert "SECRET_GOLD_CRITERION" not in prompt
        assert "正确答案:A。" not in prompt
        if "统一的最终答案生成器" in prompt or prompt.startswith("只输出严格 JSON"):
            return call('{"answer":"A"}')
        if "法律推理状态的单步行动生成器" in prompt:
            state = json.loads(prompt.split("输入数据：\n", 1)[1])["legal_state"]
            if not state["issues"]:
                value = {
                    "operation": "EXPAND_ISSUE",
                    "question": "是否符合条件",
                    "parent_issue": None,
                }
            elif state["issues"][0]["status"] == "open":
                value = {"operation": "BIND_FACT", "issue_id": "I1", "fact_ids": ["F1"]}
            elif not state["conclusions"]:
                value = {
                    "operation": "COMMIT",
                    "issue_id": "I1",
                    "conclusion": "正确选项为 A",
                    "support": ["F1"],
                }
            elif state["issues"][0]["status"] == "reasoning":
                value = {"operation": "RESOLVE", "issue_id": "I1"}
            else:
                value = {"operation": "STOP"}
            return call(json.dumps(value, ensure_ascii=False))
        if "更新通用状态" in prompt:
            return call(
                '{"observations":["已判断条件"],"plan":[],"intermediate_answer":"A","stop":true}'
            )
        return call("条件成立，因此选 A。\n[STOP]")


@pytest.mark.parametrize(
    "variant,expected_facts", [("stem_only", 1), ("scoped_options", 5)]
)
def test_material_variant_runs_through_engine_without_gold(
    experiment, variant, expected_facts
):
    from legal_state.experiment.data import load_cases

    root, config, _ = experiment
    config.legal_state_materials = variant
    base = FullClient()
    client = JournalClient(base, config, root / f"{variant}.jsonl", "fixture-model")
    record = run_case(
        next(iter(load_cases(root / "data").values())), "legal_state", config, client
    )
    assert record["status"] == "completed"
    assert len(record["initial_state"]["facts"]) == expected_facts
    assert record["final_answer"] == "A"
    assert record["final_state"]["facts"] == record["initial_state"]["facts"]


def test_scoped_reference_failure_is_recorded_without_repair_or_retry(experiment):
    from legal_state.experiment.data import load_cases

    root, config, _ = experiment
    config.legal_state_materials = "scoped_options"
    base = ScriptedClient(
        [
            call('{"operation":"EXPAND_ISSUE","question":"判断 A","scope":"A"}'),
            call('{"operation":"BIND_FACT","issue_id":"I1","fact_ids":["F3"]}'),
        ]
    )
    client = JournalClient(base, config, root / "scope_failure.jsonl", "fixture-model")
    record = run_case(
        next(iter(load_cases(root / "data").values())), "legal_state", config, client
    )
    assert record["status"] == "failed"
    assert record["failure"]["stage"] == "apply_action"
    assert "crosses option scopes" in record["failure"]["error_message"]
    assert len(record["initial_state"]["facts"]) == 5
    assert not record["failed_step"]["before_state"]["relations"]
    assert len(base.requests) == 2
    assert record["final_answer"] is None


def test_sequential_workflow_runs_through_experiment_engine(experiment):
    from legal_state.experiment.data import load_cases

    root, config, _ = experiment
    config.legal_state_workflow = "sequential"
    client = JournalClient(
        FullClient(), config, root / "sequential.jsonl", "fixture-model"
    )
    record = run_case(
        next(iter(load_cases(root / "data").values())), "legal_state", config, client
    )
    assert record["status"] == "completed"
    assert record["termination_reason"] == "stop"
    assert len(record["steps"]) == 5
    assert len(record["initial_state"]["facts"]) == 1


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    write_json(
        source,
        [
            {
                "instruction": "choose",
                "question": f"案例 {index}：甲已经交付材料。A:成立;B:不成立;C:无效;D:其他",
                "answer": "正确答案:A。",
            }
            for index in range(8)
        ],
    )
    data = tmp_path / "data"
    prepare(source, data)
    with (data / "review.csv").open(encoding="utf-8", newline="") as handle:
        review = list(csv.DictReader(handle))
    for row in review:
        row["eligible"] = "yes"
    _csv(data / "review.csv", review)
    subsets = split(data, data / "review.csv", 6, 2, 2, 20260930)
    annotations = [
        {
            "case_id": cid,
            "required_issues": [
                {
                    "issue_id": "G1",
                    "description": "必要判断",
                    "coverage_criterion": "SECRET_GOLD_CRITERION",
                }
            ],
            "review_notes": "fixture",
            "reviewed": True,
        }
        for cid in subsets["process"]
    ]
    write_jsonl(data / "annotations/issues.jsonl", annotations)
    config = ExperimentConfig(
        data_dir="data", model=ModelSettings(hidden_reasoning_disabled=True)
    )
    monkeypatch.setenv("LSP_MODEL", "fixture-model")
    monkeypatch.setenv("LSP_ENDPOINT", "https://example.test/chat/completions")
    monkeypatch.setenv("LSP_API_KEY", "fixture-secret")
    return tmp_path, config, subsets


def _csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_preparation_is_label_separated_and_deduplicated(tmp_path):
    row = {
        "instruction": "old formatting",
        "question": "案件材料A:甲B:乙C:丙D:丁",
        "answer": "正确答案:C。",
    }
    source = tmp_path / "source.json"
    write_json(
        source,
        [
            row,
            row,
            {**row, "question": "bad"},
            {**row, "question": "另一材料A:甲B:乙C:丙D:丁", "answer": "正确答案:AB。"},
        ],
    )
    result = prepare(source, tmp_path / "data")
    assert result["valid_count"] == 1
    assert result["duplicates"] == [1]
    assert len(result["errors"]) == 2
    case = read_jsonl(tmp_path / "data/cases.jsonl")[0]
    assert "answer" not in case and "instruction" not in case
    assert case["stem"] == "案件材料"
    assert read_jsonl(tmp_path / "data/targets.jsonl")[0]["gold_answer"] == "C"


def test_split_requires_completed_review_and_preserves_split(experiment):
    root, _, subsets = experiment
    assert not set(subsets["dev"]) & set(subsets["test"])
    assert set(subsets["process"]) <= set(subsets["test"])
    with pytest.raises(ValueError, match="already exist"):
        split(root / "data", root / "data/review.csv", 6, 2, 2, 1)
    with (root / "data/review.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["eligible"] = "no"
    _csv(root / "data/review.csv", rows)
    with pytest.raises(ValueError, match="changed"):
        dataset_fingerprint(root / "data")


def test_three_methods_freeze_score_review_and_resume(experiment):
    root, config, subsets = experiment
    client = FullClient()
    preflight(config, root, client)
    run_batch(config, root, "dev", "dev", client=client)
    assert score(config, root, "dev")["summary"][0]["accuracy"] == 1
    freeze(config, root, "test", "dev")
    batch = run_batch(config, root, "test", "test", client=client)
    assert batch["executed"] == 12
    calls = len(client.requests)
    resumed = run_batch(config, root, "test", "test", resume=True, client=client)
    assert resumed == {"run_id": "test", "executed": 0, "skipped": 12}
    assert len(client.requests) == calls
    result = score(config, root, "test")
    assert all(row["accuracy"] == 1 for row in result["summary"])
    assert result["comparisons"]["legal_state_minus_generic"]["ci95"] == [0, 0]
    exported = read_json(root / "runs/test/lawbench/cot/3-6.json")
    assert list(exported) == ["0", "1", "2", "3"]
    review = export_review(config, root, "test")
    assert review["materials"] == len(subsets["process"]) * 3
    with (root / "runs/test/human_review/materials/ratings.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.update(identified="yes", analyzed="yes", contradiction="no")
    _csv(root / "ratings.csv", rows)
    human = import_review(config, root, "test", root / "ratings.csv")
    assert all(row["issue_coverage_rate"] == 1 for row in human["summary"])
    artifact = report(config, root, "test")
    assert artifact["process_review_complete"] is True
    assert "100.00%" in Path(artifact["report"]).read_text()


def test_failure_is_scored_and_next_case_runs(experiment):
    root, config, _ = experiment
    preflight(config, root, FullClient())
    client = ScriptedClient(
        [
            RuntimeError("transport fixture-secret"),
            *[call("A [STOP]"), call('{"answer":"A"}')],
        ]
    )
    # A bad method output should not kill the other methods or cases.
    run_batch(config, root, "dev", "dev", client=client)
    result = score(config, root, "dev")
    assert len(result["summary"]) == 3
    assert sum(row["failure_rate"] for row in result["summary"]) > 0
    first = read_json(root / "runs/dev/records" / result_id(root) / "cot.json")
    assert "fixture-secret" not in first["failure"]["error_message"]


def result_id(root):
    return read_json(root / "data/splits.json")["dev"][0]


@pytest.mark.parametrize("method", ["cot", "generic", "legal_state"])
def test_shared_remaining_budget_and_final_reserve(experiment, method):
    from legal_state.experiment.data import load_cases

    root, config, _ = experiment
    config.reasoning.max_output_tokens_per_call = 3
    config.reasoning.max_output_tokens_total = 5
    replies = {
        "cot": [call("先判断条件", tokens=3), call("结论 A", tokens=2)],
        "generic": [
            call(
                '{"observations":[],"plan":["继续"],"intermediate_answer":"","stop":false}',
                tokens=3,
            ),
            call(
                '{"observations":["结论 A"],"plan":[],"intermediate_answer":"A","stop":false}',
                tokens=2,
            ),
        ],
        "legal_state": [
            call(
                '{"operation":"EXPAND_ISSUE","question":"条件","parent_issue":null}',
                tokens=3,
            ),
            call(
                '{"operation":"BIND_FACT","issue_id":"I1","fact_ids":["F1"]}', tokens=2
            ),
        ],
    }[method]
    base = ScriptedClient([*replies, call('{"answer":"A"}', tokens=2)])
    journal = JournalClient(base, config, root / f"{method}.jsonl", "fixture-model")
    result = run_case(
        next(iter(load_cases(root / "data").values())), method, config, journal
    )
    assert result["termination_reason"] == "token_budget"
    assert result["final_answer"] == "A"
    assert [limit for _, limit in base.requests] == [3, 2, 128]
    assert result["usage"]["output_tokens"] == 7


@pytest.mark.parametrize(
    "finish,answer", [("length", "A"), ("stop", "AB"), ("stop", "a")]
)
def test_invalid_or_truncated_answers_fail(experiment, finish, answer):
    from legal_state.experiment.data import load_cases

    root, config, _ = experiment
    base = ScriptedClient(
        [call("A [STOP]"), call(json.dumps({"answer": answer}), finish=finish)]
    )
    client = JournalClient(base, config, root / "journal.jsonl", "fixture-model")
    result = run_case(
        next(iter(load_cases(root / "data").values())), "cot", config, client
    )
    assert result["status"] == "failed"
    assert result["failure"]["stage"] == "parse_final_answer"
    assert result["usage"]["call_count"] == 2


def test_replay_completed_call_does_not_call_model(experiment):
    root, config, _ = experiment
    base = ScriptedClient([call("A [STOP]")])
    first = JournalClient(base, config, root / "journal.jsonl", "fixture-model")
    first.generate("prompt", max_output_tokens=512)
    replay = JournalClient(
        ScriptedClient([]), config, root / "journal.jsonl", "fixture-model"
    )
    assert replay.generate("prompt", max_output_tokens=512).raw_text == "A [STOP]"
    assert replay.usage()["call_count"] == 1
    with pytest.raises(ValueError, match="changed"):
        JournalClient(
            ScriptedClient([]), config, root / "journal.jsonl", "fixture-model"
        ).generate("different")


def test_pending_request_is_never_retried(experiment):
    root, config, _ = experiment
    path = root / "journal.jsonl"
    append_event(
        path,
        {
            "event": "request",
            "call_index": 0,
            "phase": "reasoning",
            "prompt": "prompt",
            "max_output_tokens": 512,
            "signature": digest(
                {"prompt": "prompt", "limit": 512, "phase": "reasoning"}
            ),
        },
    )
    base = ScriptedClient([])
    client = JournalClient(base, config, path, "fixture-model")
    with pytest.raises(RuntimeError, match="interrupted_call"):
        client.generate("prompt")
    assert base.requests == []
    assert client.usage()["unknown_usage_calls"] == 1


def test_changed_configuration_and_model_block_resume(experiment, monkeypatch):
    root, config, _ = experiment
    client = FullClient()
    preflight(config, root, client)
    run_batch(config, root, "dev", "dev", client=client)
    config.reasoning.max_steps += 1
    with pytest.raises(ValueError, match="changed"):
        run_batch(config, root, "dev", "dev", resume=True, client=client)
    config.reasoning.max_steps -= 1
    monkeypatch.setenv("LSP_MODEL", "another-model")
    with pytest.raises(ValueError, match="changed"):
        run_batch(config, root, "dev", "dev", resume=True, client=client)


@pytest.mark.parametrize(
    "prediction,correct,abstention",
    [
        ("A", 1, 0),
        ("AAA", 1, 0),
        ("B", 0, 0),
        ("A B", 0, 0),
        ("", 0, 1),
        ("UNKNOWN", 0, 1),
    ],
)
def test_official_choice_semantics(prediction, correct, abstention):
    assert choice_judge(prediction, "A") == {"score": correct, "abstention": abstention}


def test_paired_statistics_identity_and_symmetry():
    assert paired_comparison([1, 0], [1, 0], seed=1)["ci95"] == [0, 0]
    positive = paired_comparison([1, 1, 0], [0, 1, 0], seed=2)
    negative = paired_comparison([0, 1, 0], [1, 1, 0], seed=2)
    assert positive["difference"] == -negative["difference"]
    assert positive["ci95"] == [-negative["ci95"][1], -negative["ci95"][0]]


def test_dry_run_needs_no_model_credentials(experiment, monkeypatch, capsys):
    root, config, _ = experiment
    import yaml

    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config.model_dump()))
    monkeypatch.chdir(root)
    monkeypatch.delenv("LSP_MODEL")
    monkeypatch.delenv("LSP_ENDPOINT")
    monkeypatch.delenv("LSP_API_KEY")
    assert (
        main(
            [
                "run",
                "--config",
                str(config_path),
                "--run-id",
                "preview",
                "--split",
                "test",
                "--dry-run",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["combinations"] == 12


def test_interrupt_after_paid_calls_replays_before_continuing(experiment, monkeypatch):
    from legal_state.experiment import storage

    root, config, _ = experiment
    client = FullClient()
    preflight(config, root, client)
    original = storage.write_json

    def interrupted_write(path, value):
        if path.name == "cot.json":
            raise KeyboardInterrupt
        return original(path, value)

    monkeypatch.setattr(storage, "write_json", interrupted_write)
    with pytest.raises(KeyboardInterrupt):
        run_batch(config, root, "dev", "dev", client=client)
    assert len(client.requests) == 3  # preflight + reasoning + final answer
    monkeypatch.setattr(storage, "write_json", original)
    run_batch(config, root, "dev", "dev", resume=True, client=client)
    assert len(client.requests) == 21  # two cases x (2 + 2 + 6), plus preflight
    assert score(config, root, "dev")["summary"][0]["accuracy"] == 1


def test_corrupt_journal_is_preserved_without_network_retry(experiment):
    root, config, subsets = experiment
    client = FullClient()
    preflight(config, root, client)
    from legal_state.experiment.storage import write_manifest

    write_manifest(config, root, "dev", "dev")
    path = root / "runs/dev/journals" / subsets["dev"][0] / "cot.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"event": "request", "call_ind')
    run_batch(config, root, "dev", "dev", resume=True, client=client)
    result = read_json(root / "runs/dev/records" / subsets["dev"][0] / "cot.json")
    assert result["failure"]["stage"] == "checkpoint"
    assert result["usage"]["unknown_usage_calls"] == 1
    assert (
        len(client.requests) == 19
    )  # no reasoning/final calls for the corrupt combination
    assert path.read_text() == '{"event": "request", "call_ind'


def test_terminal_record_edits_are_rejected(experiment):
    root, config, subsets = experiment
    client = FullClient()
    preflight(config, root, client)
    run_batch(config, root, "dev", "dev", client=client)
    path = root / "runs/dev/records" / subsets["dev"][0] / "cot.json"
    record = read_json(path)
    record["final_answer"] = "B"
    write_json(path, record)
    with pytest.raises(ValueError, match="Terminal records changed"):
        score(config, root, "dev")


def test_failed_preflight_invalidates_earlier_pass(experiment):
    from legal_state.experiment.storage import checked_preflight

    root, config, _ = experiment
    preflight(config, root, FullClient())
    with pytest.raises(ValueError, match="Preflight failed"):
        preflight(
            config, root, ScriptedClient([call('{"answer":"A"}', finish="length")])
        )
    with pytest.raises(ValueError, match="latest preflight failed"):
        checked_preflight(config, root)


def test_concurrent_run_lock_rejects_second_process(experiment):
    import fcntl

    root, config, _ = experiment
    path = root / "runs/dev/.run.lock"
    path.parent.mkdir(parents=True)
    with path.open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="already active"):
            run_batch(config, root, "dev", "dev", client=FullClient())


@pytest.mark.parametrize("kind", ["model", "reasoning", "limit", "finish"])
def test_protocol_failure_keeps_response_usage(experiment, kind):
    root, config, _ = experiment
    result = call("A")
    overrides = {
        "model": {"model": "changed-model"},
        "reasoning": {"reasoning_tokens": 2},
        "limit": {"output_tokens": 513},
        "finish": {"finish_reason": None},
    }
    client = JournalClient(
        ScriptedClient([result.model_copy(update=overrides[kind])]),
        config,
        root / "journal.jsonl",
        "fixture-model",
    )
    with pytest.raises(ValueError):
        client.generate("prompt")
    assert client.usage()["successful_responses"] == 1
    assert client.usage()["output_tokens"] == (513 if kind == "limit" else 10)


def test_official_function_parity_without_optional_nlp_packages():
    from legal_state.experiment.evaluation import load_official_scorer

    upstream = Path(__file__).resolve().parents[1] / "third_party/LawBench"
    if not (upstream / "evaluation/evaluation_functions/jec_ac.py").exists():
        pytest.skip("Optional parity check needs the downloaded upstream checkout")
    compute = load_official_scorer(upstream)
    examples = [
        {"origin_prompt": [], "prediction": prediction, "refr": f"正确答案:{letter}。"}
        for prediction in (
            "",
            "A",
            "AB",
            "A A",
            "B",
            "C",
            "D",
            "UNKNOWN",
            "选项 C。",
            "ABCD",
        )
        for letter in "ABCD"
    ]
    actual = compute(examples)
    expected = [choice_judge(row["prediction"], row["refr"][5]) for row in examples]
    assert actual["score"] == sum(row["score"] for row in expected) / len(expected)
    assert actual["abstention_rate"] == sum(
        row["abstention"] for row in expected
    ) / len(expected)


def test_client_configuration_and_raw_error_response(experiment, monkeypatch):
    import httpx

    from legal_state.model import ModelClient

    root, config, _ = experiment
    payloads = []

    def post(url, **kwargs):
        payloads.append(kwargs["json"])
        return httpx.Response(
            401, request=httpx.Request("POST", url), json={"error": "fixture-secret"}
        )

    monkeypatch.setattr(httpx, "post", post)
    base = ModelClient(
        "fixture-secret",
        "fixture-model",
        "https://example.test/completions",
        provider_options={},
        max_output_tokens=512,
    )
    client = JournalClient(base, config, root / "journal.jsonl", "fixture-model")
    with pytest.raises(httpx.HTTPStatusError):
        client.generate("prompt", max_output_tokens=7)
    assert payloads[0]["max_tokens"] == 7
    assert payloads[0]["temperature"] == 0
    assert "thinking" not in payloads[0]
    assert "fixture-secret" not in (root / "journal.jsonl").read_text()
    assert "[REDACTED]" in read_jsonl(root / "journal.jsonl")[1]["raw_response"]


def test_report_preserves_official_check(experiment):
    root, config, _ = experiment
    upstream = Path(__file__).resolve().parents[1] / "third_party/LawBench"
    if not (upstream / "evaluation/evaluation_functions/jec_ac.py").exists():
        pytest.skip("Optional parity check needs the downloaded upstream checkout")
    config.upstream_dir = str(upstream)
    client = FullClient()
    preflight(config, root, client)
    run_batch(config, root, "dev", "dev", client=client)
    result = score(config, root, "dev", official_check=True)
    report(config, root, "dev")
    assert read_json(root / "runs/dev/scores.json")["official"] == result["official"]


def test_dependency_changes_block_resume(experiment, monkeypatch):
    from legal_state.experiment import storage

    root, config, _ = experiment
    client = FullClient()
    preflight(config, root, client)
    run_batch(config, root, "dev", "dev", client=client)
    original = storage.version
    monkeypatch.setattr(
        storage,
        "version",
        lambda name: "changed" if name == "pydantic" else original(name),
    )
    with pytest.raises(ValueError, match="changed"):
        run_batch(config, root, "dev", "dev", resume=True, client=client)
