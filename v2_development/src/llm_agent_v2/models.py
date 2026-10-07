"""Model adapter contracts and CPU-safe deferred implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from .state import AgentState
from .tools import ToolSpec


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "call_id": self.call_id,
        }


@dataclass(frozen=True)
class GenerationRequest:
    messages: tuple[dict[str, Any], ...]
    state: AgentState
    tool_specs: tuple[ToolSpec, ...]
    max_new_tokens: int = 512


@dataclass(frozen=True)
class ModelResponse:
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)


class GpuRequiredError(RuntimeError):
    """Raised when a deferred local model needs the currently unavailable GPU."""


class AdapterNotReadyError(RuntimeError):
    """Raised by an adapter whose external runtime is not configured yet."""


class ModelAdapter(ABC):
    """Runtime-facing model interface; AgentRuntime never depends on a model vendor."""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> ModelResponse:
        raise NotImplementedError

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        state: AgentState,
        tool_specs: Sequence[ToolSpec],
    ) -> ModelResponse:
        return self.generate(
            GenerationRequest(tuple(messages), state, tuple(tool_specs))
        )

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        response = self.generate(request)
        if response.content:
            yield response.content

    def tool_generate(self, request: GenerationRequest) -> ModelResponse:
        return self.generate(request)

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        raise NotImplementedError


class MockModelAdapter(ModelAdapter):
    """Deterministic adapter for CPU runtime tests and smoke evidence."""

    def __init__(self, responses: Sequence[ModelResponse], model_name: str = "cpu-mock") -> None:
        self._responses = list(responses)
        self._cursor = 0
        self.model_name = model_name

    def generate(self, request: GenerationRequest) -> ModelResponse:
        del request
        if self._cursor >= len(self._responses):
            return ModelResponse(
                content="mock response sequence exhausted",
                finish_reason="stop",
                usage={"total_tokens": 4},
            )
        response = self._responses[self._cursor]
        self._cursor += 1
        return response

    def get_model_info(self) -> dict[str, Any]:
        return {
            "adapter": "mock",
            "model_name": self.model_name,
            "device": "cpu",
            "capability_evidence": "runtime_smoke_only",
        }


class HuggingFaceModelAdapter(ModelAdapter):
    """Configuration-only local adapter until a GPU and V1 manifest are available."""

    def __init__(self, model_id: str, device: str = "cuda") -> None:
        self.model_id = model_id
        self.device = device

    def generate(self, request: GenerationRequest) -> ModelResponse:
        del request
        raise GpuRequiredError(
            "HuggingFaceModelAdapter is deferred: verify GPU and V1 model manifest first"
        )

    def get_model_info(self) -> dict[str, Any]:
        return {
            "adapter": "huggingface",
            "model_id": self.model_id,
            "device": self.device,
            "status": "deferred_until_gpu",
        }


class PEFTModelAdapter(HuggingFaceModelAdapter):
    """Configuration-only PEFT adapter; it does not load or mutate parent weights."""

    def __init__(self, base_model_id: str, adapter_path: str, device: str = "cuda") -> None:
        super().__init__(base_model_id, device=device)
        self.base_model_id = base_model_id
        self.adapter_path = adapter_path

    def get_model_info(self) -> dict[str, Any]:
        info = super().get_model_info()
        info.update({"adapter": "peft", "adapter_path": self.adapter_path})
        return info


class OpenAICompatibleAdapter(ModelAdapter):
    """Future API adapter boundary; network serving is intentionally not enabled here."""

    def __init__(self, endpoint: str, model_name: str) -> None:
        self.endpoint = endpoint
        self.model_name = model_name

    def generate(self, request: GenerationRequest) -> ModelResponse:
        del request
        raise AdapterNotReadyError(
            "OpenAI-compatible serving adapter is not configured in the CPU foundation"
        )

    def get_model_info(self) -> dict[str, Any]:
        return {
            "adapter": "openai_compatible",
            "endpoint": self.endpoint,
            "model_name": self.model_name,
            "status": "contract_only",
        }
