"""Three reasoning methods with shared budgets and replayable call journals."""

import json
import os
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from legal_state.experiment.config import ExperimentConfig
from legal_state.experiment.data import CaseInput
from legal_state.experiment.io import append_event, digest
from legal_state.experiment.materials import scoped_material_state
from legal_state.final_answer import generate_final_answer
from legal_state.model import ModelCallResult
from legal_state.runner import RunResult, RunnerStepError, TerminationReason, run_legal_state
from legal_state.schemas import Fact, Issue, LegalState

COMMON_INSTRUCTION = """依据原题材料分析四个候选选项，选择一个正确答案。
每次只推进一个小步骤，用几句简短文字保留关键判断及必要依据。长分析分步完成，不重述原题或已经记录的分析。
题干中的主张和假设按原文语境理解；候选选项不是已证实事实。
每步开始前检查已有记录。必要判断已完成并足以支持明确选项时，立即按本方法的格式结束；不要反复验证相同结论。
在终止推理前记录正确选项字母及简短理由。不要访问外部知识源。
"""

MSLR_INSTRUCTION = """依据案件材料完成内幕交易案件分析与裁判预测。
沿“内幕信息形成 → 信息知悉 → 交易行为 → 违法所得 → 法律适用与判决类型 → 处罚结果”逐步推进。
每次只推进一个尚未完成的判断，用简洁文字保留事实、规则、适用过程和阶段结论，不重述原始材料。
每步开始前检查已有记录；六个环节及最终裁判结论均已完成时立即按本方法的格式结束。
仅依据题目与模型已有知识分析，不访问外部知识源；不虚构材料中不存在的事实。
"""

MSLR_STAGE_QUESTIONS = (
    "内幕信息形成：本案哪些事项构成内幕信息，其形成及公开时间如何认定？",
    "信息知悉：当事人是否以及如何知悉相关内幕信息？",
    "交易行为：敏感期内实施了哪些与内幕信息相关的证券交易？",
    "违法所得：涉案交易的违法所得如何认定？",
    "法律适用与判决类型：应适用何种法律规则并作出何种裁判或处理？",
    "处罚结果：对当事人应作出什么具体处罚？",
)

MSLR_STAGE_MARKERS = (
    ("内幕信息",),
    ("知悉", "获知"),
    ("交易", "买入", "卖出"),
    ("违法所得", "盈利"),
    ("法律适用", "判决类型", "裁判", "处理"),
    ("处罚", "罚款", "没收"),
)


def common_instruction(case: CaseInput) -> str:
    return MSLR_INSTRUCTION if case.benchmark == "mslr" else COMMON_INSTRUCTION


def mslr_artifact_complete(artifact: str) -> bool:
    """Check protocol coverage before accepting a model's stop signal."""
    return all(any(marker in artifact for marker in group) for group in MSLR_STAGE_MARKERS)


def call_signature(prompt: str, limit: int, phase: str, response_format=None) -> str:
    payload = {"prompt": prompt, "limit": limit, "phase": phase}
    if response_format is not None:
        payload["response_format"] = response_format
    return digest(payload)


class GenericUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    observations: list[str]
    plan: list[str]
    intermediate_answer: str
    stop: bool


def generic_response_format() -> dict[str, object]:
    """Constrain the generic baseline to its declared state protocol."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "generic_state_update",
            "schema": GenericUpdate.model_json_schema(),
        },
    }


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
                        expected = call_signature(
                            event["prompt"], event["max_output_tokens"], event["phase"],
                            event.get("response_format"),
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
        self,
        prompt: str,
        *,
        max_output_tokens: int | None = None,
        response_format: dict[str, object] | None = None,
    ) -> ModelCallResult:
        default = (
            self.config.final_answer.max_output_tokens
            if self.phase == "final_answer"
            else self.config.reasoning.max_output_tokens_per_call
        )
        limit = default if max_output_tokens is None else max_output_tokens
        signature = call_signature(prompt, limit, self.phase, response_format)
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
            if response_format is not None:
                request["response_format"] = response_format
            append_event(self.path, request)
            self.requests.append(request)
            started = time.perf_counter()
            try:
                format_options = (
                    {"response_format": response_format}
                    if response_format is not None
                    else {}
                )
                result = self.base_client.generate(
                    prompt, max_output_tokens=limit, **format_options
                )
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
            "用自由文本继续一步分析，不要输出 JSON。"
            "这一小步只处理一个尚未完成的判断。"
            + (
                "六个推理环节和完整裁判结论均已形成时，在末尾单独写 [STOP]。"
                if case.benchmark == "mslr"
                else "已有分析足以确定选项时，简短写明选项及理由，在末尾单独写 [STOP]。"
            )
            + "尚未完成则保留本步结果，下一次继续；不要为写完长分析耗尽单次输出额度。"
        )
    else:
        completion = (
            "六个推理环节和完整裁判结论均已形成时，plan 必须为空，"
            "intermediate_answer 写完整分析结论，stop 必须为 true。"
            if case.benchmark == "mslr"
            else "必要判断已完成且已确定选项时，plan 必须为空，intermediate_answer 写选项字母及简短理由，stop 必须为 true。"
        )
        instruction = (
            "更新通用状态，只输出严格 JSON，字段仅为 observations、plan、intermediate_answer、stop。"
            "observations 必须是字符串数组，用简短摘要保留必要判断和依据，合并重复内容；"
            "plan 必须是字符串数组，只列尚未完成的具体判断，完成一项就移除，不重新加入已经完成的任务。"
            "未完成时 stop 为 false；intermediate_answer 保留已有阶段性结论。"
            + completion
            + "若当前状态已经满足这些条件，本步直接输出完成状态，不重复上一状态。"
            "完成状态的格式示例（内容为占位说明，不是本题答案）："
            '{"observations":["必要判断及依据"],"plan":[],'
            '"intermediate_answer":"完整结论或选项及理由","stop":true}。'
            "控制摘要长度，确保在单次输出额度内写完所有字段和 JSON 结束括号。"
        )
    return (
        common_instruction(case)
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


def rule_recall_prompt(case: CaseInput) -> str:
    """Elicit tentative conditional rules, without supplying knowledge or choosing."""
    return (
        "本步只闭卷回忆分析原题需要的法律规则，不生成最终答案。"
        "仅使用模型已有知识，不访问外部知识源。"
        "先辨认题干中的法律关系，再分别列出适用规则、成立条件和例外分支。"
        "同一行为是否产生并列或竞合后果，要完整回忆，不能只记其中一种。"
        "基础权利与合同义务分别核查法定期限和起算点，不能混用不同起点。"
        "主体与程序规则分别核对身份、具体立场、程序阶段和处理行为，勿混同近似名称。"
        "遵循原题时间适用的规则；记忆不确定时明示不确定，禁止虚构条号。"
        "只列最多六条简短规则假设，每条不超过60字；不重述题干。\n"
        + json.dumps({"question": case.question, "source_material": case.stem}, ensure_ascii=False)
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
                issues=(
                    [
                        Issue(id=f"I{index}", question=question)
                        for index, question in enumerate(
                            MSLR_STAGE_QUESTIONS, start=1
                        )
                    ]
                    if case.benchmark == "mslr"
                    else []
                ),
                facts=[Fact(id="F1", content=case.stem, source="question_material")]
            )
            if config.legal_state_materials == "scoped_options":
                initial = scoped_material_state(case)
            record["initial_state"] = initial.model_dump(mode="json")
            remaining_steps = config.reasoning.max_steps
            remaining_tokens = config.reasoning.max_output_tokens_total
            if config.legal_state_initial_analysis != "none":
                prompt = (
                    baseline_prompt(case, "cot", "")
                    if config.legal_state_initial_analysis == "cot" else rule_recall_prompt(case)
                )
                record["draft_step"] = {"prompt": prompt, "call": None}
                if config.legal_state_initial_analysis == "rules":
                    record["draft_step"]["kind"] = "model_rule_recall"
                stage = "model_call"
                call = client.generate(
                    prompt, max_output_tokens=min(
                        config.reasoning.max_output_tokens_per_call, remaining_tokens
                    ),
                )
                record["draft_step"]["call"] = call.model_dump()
                if call.finish_reason == "length":
                    raise ValueError("Initial analysis was truncated (finish_reason=length)")
                stage = "parse_reasoning"
                draft = call.raw_text.rstrip()
                if draft.endswith("[STOP]"):
                    draft = draft[:-len("[STOP]")].rstrip()
                initial = LegalState.model_validate({
                    **initial.model_dump(), "draft_reasoning": draft,
                })
                remaining_steps -= 1
                remaining_tokens -= call.output_tokens
            if remaining_steps <= 0 or remaining_tokens <= 0:
                result = RunResult(
                    case_text=case.stem, question=case.question,
                    initial_state=initial, final_state=initial, steps=(),
                    termination_reason=(
                        TerminationReason.TOKEN_BUDGET if remaining_tokens <= 0
                        else TerminationReason.MAX_STEPS
                    ), max_steps=config.reasoning.max_steps,
                )
            else:
                result = run_legal_state(
                    case_text=case.stem,
                    question=(
                        case.question if config.legal_state_workflow == "verified"
                        else common_instruction(case) + case.question
                    ),
                    initial_state=initial,
                    model_client=client,
                    max_steps=remaining_steps,
                    max_reasoning_output_tokens=remaining_tokens,
                    workflow=config.legal_state_workflow,
                    constrained_json=config.legal_state_constrained_json,
                    require_question_frame=config.legal_state_question_frame,
                    compare_options=config.legal_state_compare_options,
                )
            record["steps"] = [step.model_dump(mode="json") for step in result.steps]
            record["initial_state"] = initial.model_dump(mode="json")
            record["final_state"] = result.final_state.model_dump(mode="json")
            artifact = result.final_state.model_dump_json(indent=2)
            if result.final_state.option_decision is not None:
                # Only the audited decision is authoritative; initial verdicts remain in steps.
                artifact = result.final_state.option_decision.model_dump_json()
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
                format_options = (
                    {"response_format": generic_response_format()}
                    if method == "generic"
                    else {}
                )
                call = client.generate(
                    prompt,
                    max_output_tokens=min(
                        config.reasoning.max_output_tokens_per_call, remaining
                    ),
                    **format_options,
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
                    if (
                        stopped
                        and case.benchmark == "mslr"
                        and not mslr_artifact_complete(artifact)
                    ):
                        stopped = False
                else:
                    state = GenericUpdate.model_validate_json(
                        call.raw_text
                    ).model_dump()
                    stopped = state["stop"]
                    if (
                        stopped
                        and case.benchmark == "mslr"
                        and not mslr_artifact_complete(
                            json.dumps(state, ensure_ascii=False)
                        )
                    ):
                        state["stop"] = False
                        stopped = False
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
            answer_format=(
                "free_text" if case.benchmark == "mslr" else "single_choice"
            ),
            constrained_json=case.benchmark == "mslr",
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
