"""Three reasoning methods with shared budgets and replayable call journals."""

import json
import os
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from legal_state.experiment.config import ExperimentConfig
from legal_state.experiment.data import CaseInput
from legal_state.experiment.io import append_event, digest
from legal_state.final_answer import generate_final_answer
from legal_state.model import ModelCallResult
from legal_state.runner import RunnerStepError, run_legal_state
from legal_state.schemas import Fact, LegalState

COMMON_INSTRUCTION = """依据原题材料分析四个候选选项，选择一个正确答案。
每次只推进一个小步骤。题干中的主张和假设按原文语境理解；候选选项不是已证实事实。
在终止推理前形成对原题正确选项的明确结论。不要访问外部知识源。
"""


class GenericUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    observations: list[str]
    plan: list[str]
    intermediate_answer: str
    stop: bool


class JournalCorruptionError(ValueError):
    def __init__(self, message: str, usage: dict):
        self.usage = usage
        super().__init__(message)


def scrub_error(error: Exception | str, config: ExperimentConfig) -> str:
    message = str(error)
    secret = os.getenv(config.model.api_key_env)
    return message.replace(secret, "[REDACTED]") if secret else message


def raw_response(client, config: ExperimentConfig) -> str | None:
    response = getattr(client, "last_response", None)
    return scrub_error(response, config) if response is not None else None


class JournalClient:
    """Persist requests before sending; replay responses without new paid calls.

    A request without a response is ambiguous after interruption. It becomes a
    terminal error on resume, rather than automatically sending the request again.
    """

    def __init__(
        self, base_client, config: ExperimentConfig, path: Path, expected_model: str
    ):
        self.base_client = base_client
        self.config = config
        self.path = path
        self.expected_model = expected_model
        self.phase = "reasoning"
        self.max_output_tokens = config.reasoning.max_output_tokens_per_call
        self.requests: list[dict] = []
        self.responses: dict[int, dict] = {}
        self.cursor = 0
        self.resumed = path.exists()
        if path.exists():
            # A trailing partial write is an ambiguous request, never silently replayed.
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    event = json.loads(line)
                    if event["event"] == "request":
                        if event["call_index"] != len(self.requests):
                            raise ValueError("Non-contiguous call journal")
                        expected = digest(
                            {
                                "prompt": event["prompt"],
                                "limit": event["max_output_tokens"],
                                "phase": event["phase"],
                            }
                        )
                        if event["signature"] != expected:
                            raise ValueError("Journal request signature does not match")
                        self.requests.append(event)
                    elif event["event"] in {"response", "error"}:
                        index = event["call_index"]
                        if (
                            type(index) is not int
                            or index < 0
                            or index >= len(self.requests)
                            or index in self.responses
                        ):
                            raise ValueError("Malformed/duplicate journal response")
                        if event["event"] == "response":
                            ModelCallResult.model_validate(event["result"])
                        elif not all(
                            isinstance(event.get(key), str)
                            for key in ("error_type", "error_message")
                        ):
                            raise ValueError("Malformed journal error")
                        self.responses[index] = event
                    else:
                        raise ValueError("Unknown call journal event")
            except (ValueError, KeyError, TypeError) as error:
                usage = self.usage()
                usage["unknown_usage_calls"] = max(1, usage["unknown_usage_calls"])
                usage["call_count"] = max(
                    usage["call_count"], usage["successful_responses"] + 1
                )
                raise JournalCorruptionError(
                    "corrupt_journal: preserve and inspect journal; no automatic retry",
                    usage,
                ) from error

    def _validate_result(self, result: ModelCallResult, limit: int) -> None:
        if result.model != self.expected_model:
            raise ValueError("Returned model identity changed after preflight")
        if result.finish_reason not in {"stop", "length"}:
            raise ValueError("Missing or unsupported finish_reason")
        if result.reasoning_tokens not in (None, 0):
            raise ValueError(
                "Hidden reasoning tokens reported; experiment requires disabled reasoning"
            )
        if result.output_tokens > limit:
            raise ValueError("Provider exceeded the requested output token limit")

    def generate(
        self, prompt: str, *, max_output_tokens: int | None = None
    ) -> ModelCallResult:
        default = (
            self.config.final_answer.max_output_tokens
            if self.phase == "final_answer"
            else self.config.reasoning.max_output_tokens_per_call
        )
        limit = default if max_output_tokens is None else max_output_tokens
        signature = digest({"prompt": prompt, "limit": limit, "phase": self.phase})
        index = self.cursor
        self.cursor += 1
        if index < len(self.requests):
            request = self.requests[index]
            if request["signature"] != signature:
                raise ValueError("Replay prompt/budget changed; use a new run ID")
            response = self.responses.get(index)
            if response is None:
                raise RuntimeError(
                    "interrupted_call: request outcome unknown; no automatic retry"
                )
            if response["event"] == "error":
                raise RuntimeError(
                    f"Recorded {response['error_type']}: {response['error_message']}"
                )
            result = ModelCallResult.model_validate(response["result"])
        else:
            request = {
                "event": "request",
                "call_index": index,
                "phase": self.phase,
                "prompt": prompt,
                "max_output_tokens": limit,
                "signature": signature,
            }
            append_event(self.path, request)
            self.requests.append(request)
            started = time.perf_counter()
            try:
                result = self.base_client.generate(prompt, max_output_tokens=limit)
            except Exception as error:
                event = {
                    "event": "error",
                    "call_index": index,
                    "error_type": type(error).__name__,
                    "error_message": scrub_error(error, self.config),
                    "latency_seconds": time.perf_counter() - started,
                    "raw_response": raw_response(self.base_client, self.config),
                }
                append_event(self.path, event)
                self.responses[index] = event
                raise
            event = {
                "event": "response",
                "call_index": index,
                "result": result.model_dump(),
                "raw_response": raw_response(self.base_client, self.config),
            }
            append_event(self.path, event)
            self.responses[index] = event
        self._validate_result(result, limit)
        return result

    def usage(self) -> dict:
        results = [
            row["result"]
            for row in self.responses.values()
            if row["event"] == "response"
        ]
        input_tokens = sum(row["input_tokens"] for row in results)
        output_tokens = sum(row["output_tokens"] for row in results)
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "call_count": len(self.requests),
            "successful_responses": len(results),
            "unknown_usage_calls": len(self.requests) - len(results),
            "call_latency_seconds": sum(
                row.get(
                    "latency_seconds", row.get("result", {}).get("latency_seconds", 0)
                )
                for row in self.responses.values()
            ),
            "reported_models": sorted({row["model"] for row in results}),
            "truncated_calls": sum(
                row.get("finish_reason") == "length" for row in results
            ),
            "reasoning_output_tokens": sum(
                response["result"]["output_tokens"]
                for index, response in self.responses.items()
                if response["event"] == "response"
                and self.requests[index]["phase"] == "reasoning"
            ),
            "final_output_tokens": sum(
                response["result"]["output_tokens"]
                for index, response in self.responses.items()
                if response["event"] == "response"
                and self.requests[index]["phase"] == "final_answer"
            ),
        }


def baseline_prompt(case: CaseInput, method: str, artifact: str) -> str:
    if method == "cot":
        instruction = (
            "用自由文本继续一步分析，不要输出 JSON。已完成时在末尾单独写 [STOP]。"
        )
    else:
        instruction = (
            "更新通用状态，只输出严格 JSON："
            '{"observations":["观察"],"plan":["待完成任务"],'
            '"intermediate_answer":"阶段性答案","stop":false}。'
            "仅包含这些字段，完成时 stop 为 true。"
        )
    return (
        COMMON_INSTRUCTION
        + instruction
        + "\n"
        + json.dumps(
            {
                "question": case.question,
                "source_material": case.stem,
                "current_reasoning": artifact,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def run_case(
    case: CaseInput, method: str, config: ExperimentConfig, client: JournalClient
) -> dict:
    started = time.perf_counter()
    record = {
        "case_id": case.case_id,
        "method": method,
        "status": "failed",
        "steps": [],
        "reasoning_artifact": "",
        "termination_reason": None,
        "final_answer": None,
        "final_answer_result": None,
        "failure": None,
    }
    artifact = ""
    stage = "reasoning"
    try:
        if method == "legal_state":
            initial = LegalState(
                facts=[Fact(id="F1", content=case.stem, source="question_material")]
            )
            result = run_legal_state(
                case_text=case.stem,
                question=COMMON_INSTRUCTION + case.question,
                initial_state=initial,
                model_client=client,
                max_steps=config.reasoning.max_steps,
                max_reasoning_output_tokens=config.reasoning.max_output_tokens_total,
            )
            record["steps"] = [step.model_dump(mode="json") for step in result.steps]
            record["initial_state"] = initial.model_dump(mode="json")
            record["final_state"] = result.final_state.model_dump(mode="json")
            artifact = result.final_state.model_dump_json(indent=2)
            record["termination_reason"] = result.termination_reason.value
        elif method in {"cot", "generic"}:
            output_tokens = 0
            state = {
                "observations": [],
                "plan": [],
                "intermediate_answer": "",
                "stop": False,
            }
            record["termination_reason"] = "max_steps"
            for index in range(config.reasoning.max_steps):
                remaining = config.reasoning.max_output_tokens_total - output_tokens
                if remaining <= 0:
                    record["termination_reason"] = "token_budget"
                    break
                current = (
                    artifact
                    if method == "cot"
                    else json.dumps(state, ensure_ascii=False)
                )
                prompt = baseline_prompt(case, method, current)
                stage = "model_call"
                call = client.generate(
                    prompt,
                    max_output_tokens=min(
                        config.reasoning.max_output_tokens_per_call, remaining
                    ),
                )
                step = {
                    "step_index": index,
                    "prompt": prompt,
                    "call": call.model_dump(),
                    "before": current,
                }
                record["steps"].append(step)
                output_tokens += call.output_tokens
                if call.finish_reason == "length":
                    raise ValueError(
                        "Model output was truncated (finish_reason=length)"
                    )
                stage = "parse_reasoning"
                if method == "cot":
                    text = call.raw_text.rstrip()
                    stopped = text.endswith("[STOP]")
                    if stopped:
                        text = text[: -len("[STOP]")].rstrip()
                    artifact = "\n".join(part for part in (artifact, text) if part)
                else:
                    state = GenericUpdate.model_validate_json(
                        call.raw_text
                    ).model_dump()
                    stopped = state["stop"]
                    artifact = json.dumps(state, ensure_ascii=False, sort_keys=True)
                step["after"] = artifact
                if stopped:
                    record["termination_reason"] = "stop"
                    break
        else:
            raise ValueError(f"Unknown method {method}")
        record["reasoning_artifact"] = artifact
        client.phase = "final_answer"
        stage = "final_answer"
        answer = generate_final_answer(
            case_text=case.stem,
            question=case.question,
            provided_knowledge=(),
            reasoning_artifact=artifact,
            model_client=client,
            answer_format="single_choice",
        )
        record["final_answer_result"] = answer.model_dump(mode="json")
        record["final_answer"] = answer.parsed_answer.answer
        record["status"] = "completed"
    except Exception as error:  # noqa: BLE001 -- every method failure is an experimental outcome
        if isinstance(error, RunnerStepError):
            record["steps"] = [
                step.model_dump(mode="json") for step in error.completed_steps
            ]
            record["failed_step"] = error.failed_step.model_dump(mode="json")
            artifact = error.failed_step.before_state.model_dump_json(indent=2)
            stage = error.failed_step.stage
        record["reasoning_artifact"] = artifact
        record["failure"] = {
            "stage": getattr(error, "stage", stage),
            "error_type": type(error).__name__,
            "error_message": scrub_error(error, config),
        }
    record["usage"] = client.usage()
    record["execution_seconds"] = time.perf_counter() - started
    record["latency_seconds"] = record["usage"]["call_latency_seconds"]
    record["resumed"] = client.resumed
    return record
