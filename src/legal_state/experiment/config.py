"""Validated experiment settings; credentials are environment-only."""

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from legal_state.model import ModelClient


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelSettings(Settings):
    name_env: str = "LSP_MODEL"
    endpoint_env: str = "LSP_ENDPOINT"
    api_key_env: str = "LSP_API_KEY"
    temperature: float = Field(default=0, ge=0, le=0)
    timeout_seconds: float = Field(default=60, gt=0)
    provider_options: dict[str, JsonValue] = Field(default_factory=dict)
    # Operator attestation is needed where the service does not report reasoning usage.
    hidden_reasoning_disabled: bool = False

    @model_validator(mode="after")
    def validate_options(self):
        controlled = {
            "model",
            "messages",
            "temperature",
            "max_tokens",
            "n",
            "api_key",
            "headers",
        }
        if controlled.intersection(self.provider_options):
            raise ValueError(
                "provider_options contains controlled or credential fields"
            )
        return self


class ReasoningSettings(Settings):
    max_steps: int = Field(default=24, gt=0)
    max_output_tokens_per_call: int = Field(default=512, gt=0)
    max_output_tokens_total: int = Field(default=8192, gt=0)


class AnswerSettings(Settings):
    max_output_tokens: int = Field(default=128, gt=0)


class ExecutionSettings(Settings):
    concurrency: Literal[1] = 1
    automatic_retries: Literal[0] = 0
    repetitions: Literal[1] = 1


class ExperimentConfig(Settings):
    experiment_id: str = Field(default="lawbench_3_6_v1", pattern=r"^[A-Za-z0-9_-]+$")
    task_id: Literal["3-6"] = "3-6"
    seed: int = 20260930
    methods: list[Literal["cot", "generic", "legal_state"]] = Field(
        default_factory=lambda: ["cot", "generic", "legal_state"]
    )
    knowledge_mode: Literal["closed_book"] = "closed_book"
    data_dir: str = "data/processed/lawbench_3_6"
    runs_dir: str = "runs"
    upstream_dir: str = "third_party/LawBench"
    model: ModelSettings = Field(default_factory=ModelSettings)
    reasoning: ReasoningSettings = Field(default_factory=ReasoningSettings)
    final_answer: AnswerSettings = Field(default_factory=AnswerSettings)
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)

    @model_validator(mode="after")
    def validate_methods(self):
        if len(self.methods) != 3 or set(self.methods) != {
            "cot",
            "generic",
            "legal_state",
        }:
            raise ValueError(
                "The experiment requires exactly cot, generic, legal_state"
            )
        return self


def load_config(path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    )


def model_identity(config: ExperimentConfig, *, require_key: bool = False) -> dict:
    values = {
        "name": os.getenv(config.model.name_env),
        "endpoint": os.getenv(config.model.endpoint_env),
    }
    if not all(values.values()):
        raise ValueError(f"Set {config.model.name_env} and {config.model.endpoint_env}")
    if not str(values["endpoint"]).startswith(("http://", "https://")):
        raise ValueError("Endpoint must be a complete HTTP(S) URL")
    if require_key and not os.getenv(config.model.api_key_env):
        raise ValueError(f"Set {config.model.api_key_env}")
    if not config.model.hidden_reasoning_disabled:
        raise ValueError(
            "Confirm hidden_reasoning_disabled after configuring the provider"
        )
    return values


def create_client(config: ExperimentConfig) -> ModelClient:
    identity = model_identity(config, require_key=True)
    return ModelClient(
        api_key=os.environ[config.model.api_key_env],
        model=identity["name"],
        endpoint=identity["endpoint"],
        timeout_seconds=config.model.timeout_seconds,
        temperature=config.model.temperature,
        max_output_tokens=config.reasoning.max_output_tokens_per_call,
        provider_options=config.model.provider_options,
    )
