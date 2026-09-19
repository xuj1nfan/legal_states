from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from legal_state.actions import (
    Action,
    ActionName,
    StopAction,
    parse_action_json,
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


def determine_allowed_operations(state: LegalState) -> tuple[ActionName, ...]:
    """返回结构上可执行的操作，不判断法律上的合理性。"""
    validated_state = _snapshot(state)
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
    )


def run_legal_state(
    case_text: str,
    question: str,
    initial_state: LegalState,
    model_client: ModelClient,
    max_steps: int,
) -> RunResult:
    """同步执行一个有步数上限的 Legal State loop。"""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    initial_snapshot = _snapshot(initial_state)
    current_state = _snapshot(initial_snapshot)
    completed_steps: list[StepRecord] = []

    for step_index in range(max_steps):
        before_state = _snapshot(current_state)
        allowed_operations = determine_allowed_operations(before_state)
        prompt = build_action_prompt(
            case_text,
            question,
            before_state,
            allowed_operations,
        )

        try:
            model_call = model_client.generate(prompt)
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

        if parsed_action.operation not in allowed_operations:
            error = ValueError(
                f"Action {parsed_action.operation.value!r} is not allowed "
                "for the current state"
            )
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
        termination_reason=TerminationReason.MAX_STEPS,
        max_steps=max_steps,
    )
