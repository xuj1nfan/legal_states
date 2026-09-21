import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from legal_state.model import ModelCallResult, ModelClient

__all__ = [
    "FinalAnswer",
    "FinalAnswerError",
    "FinalAnswerResult",
    "build_final_answer_prompt",
    "generate_final_answer",
    "parse_final_answer_json",
]


class FinalAnswer(BaseModel):
    """统一最终答案的严格输出结构。"""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    answer: str = Field(min_length=1)


class FinalAnswerResult(BaseModel):
    """一次成功最终答案生成的完整结果及调用元数据。"""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    prompt: str
    raw_model_output: str
    parsed_answer: FinalAnswer
    model: str = Field(min_length=1)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_seconds: float = Field(ge=0)


FinalAnswerFailureStage = Literal["model_call", "parse_final_answer"]


class FinalAnswerError(RuntimeError):
    """保留最终答案生成失败阶段、提示词和已取得的模型调用信息。"""

    def __init__(
        self,
        *,
        stage: FinalAnswerFailureStage,
        prompt: str,
        error: Exception,
        model_call: ModelCallResult | None = None,
    ) -> None:
        self.stage = stage
        self.prompt = prompt
        self.model_call = model_call
        self.error_type = type(error).__name__
        self.error_message = str(error)
        super().__init__(
            f"Final answer generation failed during {stage}: "
            f"{self.error_type}: {self.error_message}"
        )


def build_final_answer_prompt(
    case_text: str,
    question: str,
    provided_knowledge: tuple[str, ...],
    reasoning_artifact: str,
) -> str:
    """将统一最终答案生成器的输入组装为稳定的 prompt。"""
    input_json = json.dumps(
        {
            "case_text": case_text,
            "question": question,
            "provided_knowledge": list(provided_knowledge),
            "reasoning_artifact": reasoning_artifact,
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )

    return f"""你是统一的最终答案生成器。

输入包含原始案件、原始问题、给定法律知识，以及上游产生的推理记录。
你的任务是依据推理记录，生成对原始问题的简洁、直接回答。

规则：
1. 只将推理记录转写为最终答案，不得独立重新分析案件或创建新的争点。
2. 不得补充推理记录中没有出现的法律依据或结论。
3. 不得为了获得更好的答案而修复上游推理错误。
4. 如果推理记录存在多个结论，应围绕原始问题进行答案转写。
5. 如果推理记录矛盾或信息不足，应输出它最能支持的答案，而不是重新完成案件推理。
6. answer 必须是非空字符串。
7. 只输出严格 JSON，格式必须为：
   {{"answer":"最终答案"}}
8. 不得输出 Markdown、解释、思维链或额外字段。

输入数据：
{input_json}
"""


def parse_final_answer_json(
    data: str | bytes | bytearray,
) -> FinalAnswer:
    """解析并严格校验模型生成的最终答案 JSON。"""
    return FinalAnswer.model_validate_json(data, strict=True)


def generate_final_answer(
    *,
    case_text: str,
    question: str,
    provided_knowledge: tuple[str, ...],
    reasoning_artifact: str,
    model_client: ModelClient,
) -> FinalAnswerResult:
    """调用一次模型并严格解析统一最终答案。"""
    prompt = build_final_answer_prompt(
        case_text,
        question,
        provided_knowledge,
        reasoning_artifact,
    )

    try:
        model_call = model_client.generate(prompt)
    except Exception as error:
        raise FinalAnswerError(
            stage="model_call",
            prompt=prompt,
            error=error,
        ) from error

    try:
        parsed_answer = parse_final_answer_json(model_call.raw_text)
    except Exception as error:
        raise FinalAnswerError(
            stage="parse_final_answer",
            prompt=prompt,
            model_call=model_call,
            error=error,
        ) from error

    return FinalAnswerResult(
        prompt=prompt,
        raw_model_output=model_call.raw_text,
        parsed_answer=parsed_answer,
        model=model_call.model,
        input_tokens=model_call.input_tokens,
        output_tokens=model_call.output_tokens,
        latency_seconds=model_call.latency_seconds,
    )
