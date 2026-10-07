"""CPU-first V2 Agent Runtime foundation."""

from .checkpoint import Checkpoint, JsonCheckpointStore
from .models import (
    AdapterNotReadyError,
    GpuRequiredError,
    GenerationRequest,
    HuggingFaceModelAdapter,
    MockModelAdapter,
    ModelAdapter,
    ModelResponse,
    OpenAICompatibleAdapter,
    PEFTModelAdapter,
    ToolCall,
)
from .runtime import AgentRunResult, AgentRuntime, RuntimeConfig
from .state import AgentState, StateTransition, compute_state_diff
from .tools import ToolRegistry, ToolResult, ToolSpec, build_cpu_registry

__all__ = [
    "AdapterNotReadyError",
    "AgentRunResult",
    "AgentRuntime",
    "AgentState",
    "Checkpoint",
    "GenerationRequest",
    "GpuRequiredError",
    "HuggingFaceModelAdapter",
    "JsonCheckpointStore",
    "MockModelAdapter",
    "ModelAdapter",
    "ModelResponse",
    "OpenAICompatibleAdapter",
    "PEFTModelAdapter",
    "RuntimeConfig",
    "StateTransition",
    "ToolCall",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "build_cpu_registry",
    "compute_state_diff",
]
