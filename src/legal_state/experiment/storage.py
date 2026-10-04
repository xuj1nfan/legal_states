"""Preflight, experiment freezing and resumable batch execution."""

import fcntl
import re
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from legal_state.experiment.config import (
    ExperimentConfig,
    create_client,
    model_identity,
)
from legal_state.experiment.data import dataset_fingerprint, load_cases
from legal_state.experiment.engine import (
    GenericUpdate,
    JournalClient,
    JournalCorruptionError,
    generic_response_format,
    raw_response,
    run_case,
    scrub_error,
)
from legal_state.experiment.io import (
    digest,
    file_digest,
    read_json,
    source_fingerprint,
    write_json,
)
from legal_state.json_grammar import compact_json_grammar, has_formatting_whitespace


def run_path(config: ExperimentConfig, run_id: str, root: Path) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError(
            "run-id must contain only letters, digits, underscores or hyphens"
        )
    return root / config.runs_dir / run_id


def preflight_key(config: ExperimentConfig, root: Path) -> dict:
    return {
        "config": config.model_dump(),
        "model": model_identity(config),
        "source_sha256": source_fingerprint(root),
        "runtime": runtime_identity(),
    }


def runtime_identity() -> dict:
    return {
        "python_version": sys.version,
        "runtime_packages": {
            name: version(name)
            for name in ("httpx", "httpcore", "pydantic", "pydantic_core", "PyYAML")
        },
    }


def preflight(config: ExperimentConfig, root: Path, client=None) -> dict:
    key = preflight_key(config, root)
    compact_planning = config.legal_state_compact_json and config.legal_state_compact_json_scope == "all"
    client = create_client(config) if client is None else client
    report = {
        "identity": key,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "failed",
    }
    destination = root / config.runs_dir / "preflight" / f"{digest(key)}.json"
    try:
        format_options = {}
        if config.legal_state_constrained_json:
            format_options["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "preflight_answer",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string", "enum": ["A"]}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                },
            }
        result = client.generate(
            '只输出严格 JSON：{"answer":"A"}。',
            max_output_tokens=128,
            **format_options,
        )
        report.update(
            returned_model=result.model,
            result=result.model_dump(),
            reasoning_usage_reported=result.reasoning_tokens is not None,
        )
        if result.finish_reason != "stop" or result.reasoning_tokens not in (None, 0):
            raise ValueError(
                "Preflight requires finish_reason=stop and no reported hidden reasoning"
            )
        if result.output_tokens > 128 or read_answer(result.raw_text) != "A":
            raise ValueError("Preflight response does not satisfy output contract")
        if compact_planning and has_formatting_whitespace(result.raw_text):
            raise ValueError("Compact JSON preflight decoded formatting whitespace")
        report["raw_response"] = raw_response(client, config)
        if "generic" in config.methods:
            prompt = (
                "只输出严格 JSON。observations 和 plan 必须是字符串数组，"
                "intermediate_answer 必须是字符串，stop 必须是布尔值。"
                "记录已经识别材料并保留一个待办事项。"
            )
            response_format = generic_response_format()
            report["generic_probe"] = {
                "prompt": prompt,
                "response_format": response_format,
            }
            probe = client.generate(
                prompt,
                max_output_tokens=256,
                response_format=response_format,
            )
            report["generic_probe"].update(
                result=probe.model_dump(), raw_response=raw_response(client, config)
            )
            if (
                probe.model != result.model
                or probe.finish_reason != "stop"
                or probe.reasoning_tokens not in (None, 0)
                or probe.output_tokens > 256
            ):
                raise ValueError("Generic schema preflight failed the model/output contract")
            GenericUpdate.model_validate_json(probe.raw_text)
        if (
            config.legal_state_workflow == "sequential"
            and config.legal_state_constrained_json
        ):
            from legal_state.actions import BindFactAction, parse_action_json
            from legal_state.prompts import build_action_prompt
            from legal_state.runner import (
                action_response_format,
                determine_allowed_operations,
            )
            from legal_state.schemas import Fact, Issue, LegalState

            state = LegalState(
                issues=[Issue(id="I1", question="测试争点")],
                facts=[Fact(id="F1", content="测试材料", source="preflight")],
            )
            allowed = determine_allowed_operations(state, workflow="sequential")
            prompt = build_action_prompt(
                "测试材料", "测试问题", state, allowed, focus_issue_id="I1"
            )
            response_format = action_response_format(
                state, allowed, focus_issue_id="I1"
            )
            report["sequential_probe"] = {
                "prompt": prompt,
                "response_format": response_format,
            }
            probe = client.generate(
                prompt,
                max_output_tokens=256,
                response_format=response_format,
            )
            report["sequential_probe"].update(
                result=probe.model_dump(), raw_response=raw_response(client, config)
            )
            if (
                probe.model != result.model
                or probe.finish_reason != "stop"
                or probe.reasoning_tokens not in (None, 0)
                or probe.output_tokens > 256
            ):
                raise ValueError(
                    "Sequential schema preflight failed the model/output contract"
                )
            action = parse_action_json(probe.raw_text)
            if not isinstance(action, BindFactAction) or action.issue_id != "I1":
                raise ValueError("Sequential preflight requires BIND_FACT for I1")
            if compact_planning and has_formatting_whitespace(probe.raw_text):
                raise ValueError("Compact sequential preflight decoded whitespace")
        if config.legal_state_question_frame:
            from legal_state.actions import (
                AssessOptionAction, AuditOptionsAction, FrameQuestionAction, parse_action_json,
            )
            from legal_state.assessment import assessment_response_format, build_assessment_prompt
            from legal_state.executor import apply_action
            from legal_state.experiment.data import CaseInput
            from legal_state.experiment.materials import scoped_material_state

            # Probe the actual framing grammar on an independent synthetic item,
            # rather than discovering unsupported schema features on a benchmark.
            case = CaseInput(case_id="lawbench-3-6-000000", source_index=0,
                             stem="测试题干", question="测试题干A:甲B:乙C:丙D:丁",
                             options=dict(zip("ABCD", "甲乙丙丁", strict=True)))
            state = scoped_material_state(case)
            prompt = build_assessment_prompt(case.question, state, require_question_frame=True,
                                                   compare_options=config.legal_state_compare_options)
            response_format = (
                assessment_response_format(state, require_question_frame=True)
                if config.legal_state_constrained_json else None
            )
            report["framing_probe"] = {"prompt": prompt, "response_format": response_format}
            if compact_planning:
                report["framing_probe"]["grammar"] = compact_json_grammar(response_format["json_schema"]["schema"])
            format_options = {"response_format": response_format} if response_format else {}
            probe = client.generate(prompt, max_output_tokens=512, **format_options)
            report["framing_probe"].update(result=probe.model_dump(), raw_response=raw_response(client, config))
            if compact_planning and has_formatting_whitespace(probe.raw_text):
                raise ValueError("Compact framing preflight decoded formatting whitespace")
            if (probe.model != result.model or probe.finish_reason != "stop"
                    or probe.reasoning_tokens not in (None, 0) or probe.output_tokens > 512):
                raise ValueError("Framing preflight failed the model/output contract")
            action = parse_action_json(probe.raw_text)
            if not isinstance(action, FrameQuestionAction):
                raise ValueError("Framing preflight requires FRAME_QUESTION")
            state = apply_action(state, action)
            prompt = build_assessment_prompt(case.question, state, require_question_frame=True,
                                                   compare_options=config.legal_state_compare_options)
            response_format = (
                assessment_response_format(state, require_question_frame=True)
                if config.legal_state_constrained_json else None
            )
            report["assessment_probe"] = {"prompt": prompt, "response_format": response_format}
            if compact_planning:
                report["assessment_probe"]["grammar"] = compact_json_grammar(response_format["json_schema"]["schema"])
            format_options = {"response_format": response_format} if response_format else {}
            probe = client.generate(prompt, max_output_tokens=512, **format_options)
            report["assessment_probe"].update(result=probe.model_dump(), raw_response=raw_response(client, config))
            if compact_planning and has_formatting_whitespace(probe.raw_text):
                raise ValueError("Compact assessment preflight decoded formatting whitespace")
            if (probe.model != result.model or probe.finish_reason != "stop"
                    or probe.reasoning_tokens not in (None, 0) or probe.output_tokens > 512):
                raise ValueError("Assessment preflight failed the model/output contract")
            action = parse_action_json(probe.raw_text)
            if not isinstance(action, AssessOptionAction):
                raise ValueError("Assessment preflight requires ASSESS_OPTION")
            state = apply_action(state, action)
            if config.legal_state_compact_json:
                from legal_state.schemas import OptionAssessment

                # Populate the remaining synthetic hypotheses without spending
                # benchmark calls; probe the nested audit schema that used to loop.
                for letter in "BCD":
                    material_id = next(fact.id for fact in state.facts
                                       if fact.material and fact.material.scope == letter)
                    assessed = OptionAssessment(
                        option=letter, rule="合成预检题无案情，无法判断",
                        rule_source="model_recall", exception="无法确定",
                        conditions=[{"condition": "选项原文", "evidence": [material_id],
                                     "finding": "未给定案情"}], verdict="uncertain",
                    )
                    state = apply_action(state, AssessOptionAction(
                        operation="ASSESS_OPTION", assessment=assessed,
                    ))
                prompt = build_assessment_prompt(case.question, state, require_question_frame=True,
                                                   compare_options=config.legal_state_compare_options)
                response_format = assessment_response_format(state, require_question_frame=True)
                report["audit_probe"] = {"prompt": prompt, "response_format": response_format}
                report["audit_probe"]["grammar"] = compact_json_grammar(response_format["json_schema"]["schema"])
                probe = client.generate(prompt, max_output_tokens=512, response_format=response_format)
                report["audit_probe"].update(result=probe.model_dump(), raw_response=raw_response(client, config))
                if has_formatting_whitespace(probe.raw_text):
                    raise ValueError("Compact audit preflight decoded formatting whitespace")
                if (probe.model != result.model or probe.finish_reason != "stop"
                        or probe.reasoning_tokens not in (None, 0) or probe.output_tokens > 512):
                    raise ValueError("Audit preflight failed the model/output contract")
                action = parse_action_json(probe.raw_text)
                if not isinstance(action, AuditOptionsAction):
                    raise ValueError("Audit preflight requires AUDIT_OPTIONS")
                apply_action(state, action)
    except Exception as error:
        report["error"] = scrub_error(error, config)
        report["raw_response"] = raw_response(client, config)
        write_json(destination, report)
        raise ValueError(f"Preflight failed: {report['error']}") from error
    report["status"] = "passed"
    write_json(destination, report)
    return report


def read_answer(text: str) -> str:
    from legal_state.final_answer import parse_final_answer_json

    return parse_final_answer_json(text).answer


def checked_preflight(config: ExperimentConfig, root: Path) -> dict:
    key = preflight_key(config, root)
    path = root / config.runs_dir / "preflight" / f"{digest(key)}.json"
    if not path.exists():
        raise ValueError(
            "Run preflight with this exact code, model and configuration first"
        )
    report = read_json(path)
    if report.get("status") != "passed":
        raise ValueError(
            "The latest preflight failed; complete a successful preflight first"
        )
    return report


def signature(config: ExperimentConfig, root: Path, split_name: str) -> dict:
    data = root / config.data_dir
    cases = load_cases(data)
    splits = read_json(data / "splits.json")
    if splits["seed"] != config.seed:
        raise ValueError("Sample seed and experiment configuration seed must match")
    selected = splits[split_name]
    if len(selected) != len(set(selected)) or not set(selected).issubset(cases):
        raise ValueError("Malformed sample split")
    if set(splits["dev"]) & set(splits["test"]) or not set(splits["process"]).issubset(
        splits["test"]
    ):
        raise ValueError("Splits overlap or process subset is not held out")
    return {
        "config": config.model_dump(),
        "model": model_identity(config),
        "source_sha256": source_fingerprint(root),
        "runtime": runtime_identity(),
        "split": split_name,
        "case_ids": selected,
        "dataset": dataset_fingerprint(data, annotations=split_name == "test"),
    }


def write_manifest(
    config: ExperimentConfig, root: Path, run_id: str, split_name: str
) -> dict:
    identity = signature(config, root, split_name)
    ready = checked_preflight(config, root)
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    manifest = {
        "run_id": run_id,
        "identity": identity,
        "identity_sha256": digest(identity),
        "returned_model": ready["returned_model"],
        "preflight": ready,
        "created_at": datetime.now(UTC).isoformat(),
        "project_commit": revision.stdout.strip() if revision.returncode == 0 else None,
        **runtime_identity(),
        "root": str(root.resolve()),
    }
    write_json(run_path(config, run_id, root) / "manifest.json", manifest)
    return manifest


def check_manifest(
    config: ExperimentConfig, root: Path, manifest: dict, split_name: str
) -> None:
    if digest(signature(config, root, split_name)) != manifest["identity_sha256"]:
        raise ValueError(
            "Code, configuration, model or dataset changed; use a new run ID"
        )


def completed_records(path: Path, manifest: dict) -> list[dict]:
    rows = []
    hashes = {}
    for cid in manifest["identity"]["case_ids"]:
        for method in manifest["identity"]["config"]["methods"]:
            record_path = path / "records" / cid / f"{method}.json"
            if not record_path.exists():
                raise ValueError(f"Missing terminal record for {cid}/{method}")
            row = read_json(record_path)
            if (
                row["case_id"] != cid
                or row["method"] != method
                or row["status"] not in {"completed", "failed"}
            ):
                raise ValueError("Malformed terminal record")
            rows.append(row)
            hashes[f"{cid}/{method}"] = file_digest(record_path)
    seal = path / "records_manifest.json"
    if seal.exists() and read_json(seal) != hashes:
        raise ValueError("Terminal records changed after the batch completed")
    return rows


def freeze(config: ExperimentConfig, root: Path, run_id: str, dev_run_id: str) -> dict:
    destination = run_path(config, run_id, root)
    if destination.exists():
        raise ValueError("Run already exists; choose a new run ID")
    dev_path = run_path(config, dev_run_id, root)
    dev_manifest = read_json(dev_path / "manifest.json")
    check_manifest(config, root, dev_manifest, "dev")
    records = completed_records(dev_path, dev_manifest)
    def valid_answer(row: dict) -> bool:
        answer = row.get("final_answer")
        if config.benchmark == "mslr":
            return isinstance(answer, str) and bool(answer.strip())
        return answer in ("A", "B", "C", "D")

    for method in config.methods:
        if not any(
            row["method"] == method
            and row["status"] == "completed"
            and row["termination_reason"] == "stop"
            and valid_answer(row)
            and (
                method != "legal_state"
                or (
                    row.get("final_state", {}).get("conclusions")
                    and any(
                        issue["status"] == "resolved"
                        for issue in row["final_state"]["issues"]
                    )
                )
            )
            for row in records
        ):
            raise ValueError(
                f"Development needs a complete STOP trajectory for {method}"
            )
    manifest = write_manifest(config, root, run_id, "test")
    manifest["development_run"] = dev_run_id
    manifest["development_records_sha256"] = digest(records)
    write_json(destination / "manifest.json", manifest)
    return manifest


def run_batch(
    config: ExperimentConfig,
    root: Path,
    run_id: str,
    split_name: str,
    resume: bool = False,
    client=None,
) -> dict:
    """Hold an OS lock throughout a batch; process death releases it automatically."""
    path = run_path(config, run_id, root)
    if split_name == "test" and not (path / "manifest.json").exists():
        raise ValueError(
            "Freeze the test experiment after development before running it"
        )
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".run.lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("This run is already active in another process") from error
        try:
            return _run_batch(config, root, run_id, split_name, resume, client)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _run_batch(
    config: ExperimentConfig,
    root: Path,
    run_id: str,
    split_name: str,
    resume: bool = False,
    client=None,
) -> dict:
    path = run_path(config, run_id, root)
    manifest_path = path / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        check_manifest(config, root, manifest, split_name)
        existing = (path / "records").exists() or (path / "journals").exists()
        if existing and not resume:
            raise ValueError("Run has started; pass --resume or choose a new run ID")
    elif split_name == "test":
        raise ValueError(
            "Freeze the test experiment after development before running it"
        )
    else:
        if path.exists() and any(item.name != ".run.lock" for item in path.iterdir()):
            raise ValueError("Run directory exists without a manifest")
        manifest = write_manifest(config, root, run_id, split_name)
    # The model-facing loader never opens the target file.
    cases = load_cases(root / config.data_dir)
    sealed_path = path / "records_manifest.json"
    if sealed_path.exists():
        completed_records(path, manifest)
    base = create_client(config) if client is None else client
    executed, skipped = 0, 0
    for index, cid in enumerate(manifest["identity"]["case_ids"]):
        offset = index % len(config.methods)
        methods = config.methods[offset:] + config.methods[:offset]
        for method in methods:
            destination = path / "records" / cid / f"{method}.json"
            if destination.exists():
                row = read_json(destination)
                if (
                    row.get("status") not in {"completed", "failed"}
                    or row.get("case_id") != cid
                    or row.get("method") != method
                ):
                    raise ValueError("Existing record is not terminal")
                skipped += 1
                continue
            journal = path / "journals" / cid / f"{method}.jsonl"
            try:
                client_for_case = JournalClient(
                    base, config, journal, manifest["returned_model"]
                )
            except JournalCorruptionError as error:
                record = {
                    "case_id": cid,
                    "method": method,
                    "status": "failed",
                    "steps": [],
                    "reasoning_artifact": "",
                    "termination_reason": None,
                    "final_answer": None,
                    "final_answer_result": None,
                    "failure": {
                        "stage": "checkpoint",
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    },
                    "usage": error.usage,
                    "execution_seconds": 0,
                    "latency_seconds": error.usage["call_latency_seconds"],
                    "resumed": True,
                }
            else:
                record = run_case(cases[cid], method, config, client_for_case)
            record["run_id"] = run_id
            write_json(destination, record)
            executed += 1
            print(f"{cid}/{method}: {record['status']}", flush=True)
    summary = {"run_id": run_id, "executed": executed, "skipped": skipped}
    # A completed batch is sealed, so accidental result edits cannot silently rescore it.
    completed_records(path, manifest)
    if not sealed_path.exists():
        write_json(
            sealed_path,
            {
                f"{cid}/{method}": file_digest(
                    path / "records" / cid / f"{method}.json"
                )
                for cid in manifest["identity"]["case_ids"]
                for method in config.methods
            },
        )
    write_json(path / "batch_summary.json", summary)
    return summary
