from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from legal_state.actions import (
    Action,
    ActionName,
    StopAction,
    parse_action_json,
)
from legal_state.assessment import (
    assessment_response_format,
    build_assessment_prompt,
    next_option,
)
from legal_state.executor import apply_action
from legal_state.model import ModelCallResult, ModelClient
from legal_state.prompts import build_action_prompt
from legal_state.schemas import IssueStatus, LegalState

__all__ = [
    "FailedStepRecord",
    "RunResult",
    "RunnerStepError",
    "StepRecord",
    "TerminationReason",
    "determine_allowed_operations",
    "run_legal_state",
]


class TerminationReason(StrEnum):
    STOP = "stop"
    MAX_STEPS = "max_steps"
    TOKEN_BUDGET = "token_budget"


class _RunnerModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class StepRecord(_RunnerModel):
    """一次成功的模型调用及其完整状态转移。"""

    step_index: int = Field(ge=0)
    before_state: LegalState
    allowed_operations: tuple[ActionName, ...]
    prompt: str
    raw_model_output: str
    parsed_action: Action
    after_state: LegalState
    model: str = Field(min_length=1)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_seconds: float = Field(ge=0)
    finish_reason: str | None = None


class RunResult(_RunnerModel):
    """一次 Legal State loop 的自包含结果。"""

    case_text: str
    question: str
    initial_state: LegalState
    final_state: LegalState
    steps: tuple[StepRecord, ...]
    termination_reason: TerminationReason
    max_steps: int = Field(gt=0)


FailureStage = Literal[
    "model_call",
    "parse_action",
    "check_allowed_operation",
    "apply_action",
]


class FailedStepRecord(_RunnerModel):
    """失败步骤中已经能够取得的实验信息。"""

    step_index: int = Field(ge=0)
    stage: FailureStage
    before_state: LegalState
    allowed_operations: tuple[ActionName, ...]
    prompt: str
    raw_model_output: str | None = None
    parsed_action: Action | None = None
    model: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_seconds: float | None = Field(default=None, ge=0)
    error_type: str = Field(min_length=1)
    error_message: str
    finish_reason: str | None = None


class RunnerStepError(RuntimeError):
    """保留既有轨迹和失败现场的单步执行错误。"""

    def __init__(
        self,
        completed_steps: tuple[StepRecord, ...],
        failed_step: FailedStepRecord,
    ) -> None:
        self.completed_steps = completed_steps
        self.failed_step = failed_step
        super().__init__(
            f"Runner step {failed_step.step_index} failed during "
            f"{failed_step.stage}: {failed_step.error_type}: "
            f"{failed_step.error_message}"
        )


def _snapshot(state: LegalState) -> LegalState:
    return LegalState.model_validate(state.model_dump())


def determine_allowed_operations(
    state: LegalState, *, workflow: Literal["free", "sequential", "verified"] = "free",
    require_question_frame: bool = False,
) -> tuple[ActionName, ...]:
    """返回结构上可执行的操作，不判断法律上的合理性。"""
    validated_state = _snapshot(state)
    if workflow not in ("free", "sequential", "verified"):
        raise ValueError("Unknown Legal State workflow")
    if require_question_frame and workflow != "verified":
        raise ValueError("Question framing requires the verified workflow")
    if workflow == "verified":
        if validated_state.option_decision:
            return (ActionName.STOP,)
        if require_question_frame and validated_state.question_frame is None:
            return (ActionName.FRAME_QUESTION,)
        if next_option(validated_state) is not None:
            return (ActionName.ASSESS_OPTION,)
        return (ActionName.AUDIT_OPTIONS,)
    if workflow == "sequential":
        focus = next(
            (
                issue
                for issue in validated_state.issues
                if issue.status is not IssueStatus.RESOLVED
            ),
            None,
        )
        if focus is None:
            return (
                (ActionName.EXPAND_ISSUE, ActionName.STOP)
                if validated_state.conclusions
                else (ActionName.EXPAND_ISSUE,)
            )
        if focus.status is IssueStatus.OPEN:
            if not validated_state.facts:
                raise ValueError("Sequential workflow needs material for an open issue")
            return (ActionName.BIND_FACT,)
        if any(c.issue_id == focus.id for c in validated_state.conclusions):
            return (ActionName.RESOLVE,)
        return (ActionName.COMMIT,)
    operations = [ActionName.EXPAND_ISSUE]

    actionable_issues = [
        issue
        for issue in validated_state.issues
        if issue.status in (IssueStatus.OPEN, IssueStatus.REASONING)
    ]
    reasoning_issues = [
        issue
        for issue in validated_state.issues
        if issue.status is IssueStatus.REASONING
    ]

    if validated_state.facts and actionable_issues:
        operations.append(ActionName.BIND_FACT)
    if reasoning_issues:
        operations.append(ActionName.COMMIT)

    concluded_issue_ids = {
        conclusion.issue_id for conclusion in validated_state.conclusions
    }
    if any(issue.id in concluded_issue_ids for issue in reasoning_issues):
        operations.append(ActionName.RESOLVE)

    operations.append(ActionName.STOP)
    return tuple(operations)


def action_response_format(
    state: LegalState,
    allowed_operations: tuple[ActionName, ...],
    *,
    focus_issue_id: str | None = None,
) -> dict[str, object]:
    """Constrain free/sequential actions to valid fields and current references."""
    branches: list[dict[str, object]] = []
    schema_operations = (
        (ActionName.STOP,)
        if ActionName.STOP in allowed_operations
        else allowed_operations
    )
    for operation in schema_operations:
        properties: dict[str, object] = {
            "operation": {"type": "string", "const": operation.value}
        }
        required = ["operation"]
        if operation is ActionName.EXPAND_ISSUE:
            properties["question"] = {"type": "string", "minLength": 1}
            required.append("question")
        elif operation is ActionName.BIND_FACT:
            properties.update(
                issue_id={"type": "string", "const": focus_issue_id},
                fact_ids={
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [fact.id for fact in state.facts],
                    },
                    "minItems": 1,
                },
            )
            required.extend(("issue_id", "fact_ids"))
        elif operation is ActionName.COMMIT:
            properties.update(
                issue_id={"type": "string", "const": focus_issue_id},
                conclusion={"type": "string", "minLength": 1},
                support={
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": state.allowed_support_ids(focus_issue_id),
                    },
                },
            )
            required.extend(("issue_id", "conclusion", "support"))
        elif operation is ActionName.RESOLVE:
            properties["issue_id"] = {
                "type": "string",
                "const": focus_issue_id,
            }
            required.append("issue_id")
        elif operation is not ActionName.STOP:
            raise ValueError(f"Unsupported action schema for {operation.value}")
        branches.append(
            {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            }
        )
    schema = branches[0] if len(branches) == 1 else {"anyOf": branches}
    return {
        "type": "json_schema",
        "json_schema": {"name": "legal_state_action", "schema": schema},
    }


def _failed_step_record(
    *,
    step_index: int,
    stage: FailureStage,
    before_state: LegalState,
    allowed_operations: tuple[ActionName, ...],
    prompt: str,
    error: Exception,
    model_call: ModelCallResult | None = None,
    parsed_action: Action | None = None,
) -> FailedStepRecord:
    return FailedStepRecord(
        step_index=step_index,
        stage=stage,
        before_state=_snapshot(before_state),
        allowed_operations=allowed_operations,
        prompt=prompt,
        raw_model_output=model_call.raw_text if model_call is not None else None,
        parsed_action=parsed_action,
        model=model_call.model if model_call is not None else None,
        input_tokens=model_call.input_tokens if model_call is not None else None,
        output_tokens=model_call.output_tokens if model_call is not None else None,
        latency_seconds=(
            model_call.latency_seconds if model_call is not None else None
        ),
        error_type=type(error).__name__,
        error_message=str(error),
        finish_reason=model_call.finish_reason if model_call is not None else None,
    )


def run_legal_state(
    case_text: str,
    question: str,
    initial_state: LegalState,
    model_client: ModelClient,
    max_steps: int,
    *,
    max_reasoning_output_tokens: int | None = None,
    workflow: Literal["free", "sequential", "verified"] = "free",
    constrained_json: bool = False,
    require_question_frame: bool = False,
    compare_options: bool = False,
) -> RunResult:
    """同步执行一个有步数上限的 Legal State loop。"""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if compare_options and workflow != "verified":
        raise ValueError("Option comparison requires the verified workflow")
    if max_reasoning_output_tokens is not None and max_reasoning_output_tokens <= 0:
        raise ValueError("max_reasoning_output_tokens must be positive")

    initial_snapshot = _snapshot(initial_state)
    current_state = _snapshot(initial_snapshot)
    completed_steps: list[StepRecord] = []
    output_tokens = 0
    termination_reason = TerminationReason.MAX_STEPS

    for step_index in range(max_steps):
        if (
            max_reasoning_output_tokens is not None
            and output_tokens >= max_reasoning_output_tokens
        ):
            termination_reason = TerminationReason.TOKEN_BUDGET
            break
        before_state = _snapshot(current_state)
        allowed_operations = determine_allowed_operations(
            before_state, workflow=workflow, require_question_frame=require_question_frame,
        )
        focus_issue_id = None
        if workflow == "sequential":
            focus_issue_id = next(
                (
                    issue.id
                    for issue in before_state.issues
                    if issue.status is not IssueStatus.RESOLVED
                ),
                None,
            )
        prompt = (
            build_assessment_prompt(question, before_state, require_question_frame=require_question_frame,
                                    compare_options=compare_options)
            if workflow == "verified"
            else build_action_prompt(
                case_text,
                question,
                before_state,
                allowed_operations,
                focus_issue_id=focus_issue_id,
            )
        )

        if workflow == "verified" and constrained_json:
            response_format = assessment_response_format(
                before_state, require_question_frame=require_question_frame,
            )
        elif constrained_json:
            response_format = action_response_format(
                before_state,
                allowed_operations,
                focus_issue_id=focus_issue_id,
            )
        else:
            response_format = None
        format_options = (
            {"response_format": response_format} if response_format else {}
        )

        try:
            if max_reasoning_output_tokens is None:
                model_call = model_client.generate(prompt, **format_options)
            else:
                model_call = model_client.generate(
                    prompt,
                    max_output_tokens=min(
                        model_client.max_output_tokens,
                        max_reasoning_output_tokens - output_tokens,
                    ),
                    **format_options,
                )
        except Exception as error:
            failed_step = _failed_step_record(
                step_index=step_index,
                stage="model_call",
                before_state=before_state,
                allowed_operations=allowed_operations,
                prompt=prompt,
                error=error,
            )
            raise RunnerStepError(tuple(completed_steps), failed_step) from error

        output_tokens += model_call.output_tokens
        if model_call.finish_reason == "length":
            error = ValueError("Model output was truncated (finish_reason=length)")
            raise RunnerStepError(
                tuple(completed_steps),
                _failed_step_record(
                    step_index=step_index,
                    stage="model_call",
                    before_state=before_state,
                    allowed_operations=allowed_operations,
                    prompt=prompt,
                    error=error,
                    model_call=model_call,
                ),
            ) from error

        try:
            parsed_action = parse_action_json(model_call.raw_text)
        except Exception as error:
            failed_step = _failed_step_record(
                step_index=step_index,
                stage="parse_action",
                before_state=before_state,
                allowed_operations=allowed_operations,
                prompt=prompt,
                error=error,
                model_call=model_call,
            )
            raise RunnerStepError(tuple(completed_steps), failed_step) from error

        wrong_target = (
            focus_issue_id is not None
            and getattr(parsed_action, "issue_id", None) != focus_issue_id
        )
        if parsed_action.operation not in allowed_operations or wrong_target:
            message = f"Action {parsed_action.operation.value!r} is not allowed for the current state"
            if wrong_target:
                message += f"; expected issue target {focus_issue_id!r}"
            error = ValueError(message)
            failed_step = _failed_step_record(
                step_index=step_index,
                stage="check_allowed_operation",
                before_state=before_state,
                allowed_operations=allowed_operations,
                prompt=prompt,
                error=error,
                model_call=model_call,
                parsed_action=parsed_action,
            )
            raise RunnerStepError(tuple(completed_steps), failed_step) from error

        try:
            after_state = apply_action(before_state, parsed_action)
        except Exception as error:
            failed_step = _failed_step_record(
                step_index=step_index,
                stage="apply_action",
                before_state=before_state,
                allowed_operations=allowed_operations,
                prompt=prompt,
                error=error,
                model_call=model_call,
                parsed_action=parsed_action,
            )
            raise RunnerStepError(tuple(completed_steps), failed_step) from error

        step = StepRecord(
            step_index=step_index,
            before_state=_snapshot(before_state),
            allowed_operations=allowed_operations,
            prompt=prompt,
            raw_model_output=model_call.raw_text,
            parsed_action=parsed_action,
            after_state=_snapshot(after_state),
            model=model_call.model,
            input_tokens=model_call.input_tokens,
            output_tokens=model_call.output_tokens,
            latency_seconds=model_call.latency_seconds,
            finish_reason=model_call.finish_reason,
        )
        completed_steps.append(step)
        current_state = _snapshot(after_state)

        if isinstance(parsed_action, StopAction):
            return RunResult(
                case_text=case_text,
                question=question,
                initial_state=_snapshot(initial_snapshot),
                final_state=_snapshot(current_state),
                steps=tuple(completed_steps),
                termination_reason=TerminationReason.STOP,
                max_steps=max_steps,
            )

    return RunResult(
        case_text=case_text,
        question=question,
        initial_state=_snapshot(initial_snapshot),
        final_state=_snapshot(current_state),
        steps=tuple(completed_steps),
        termination_reason=termination_reason,
        max_steps=max_steps,
    )
