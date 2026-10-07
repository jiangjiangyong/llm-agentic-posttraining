"""Standardized failure taxonomy for long-horizon tool agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .state import AgentState


class FailureType(str, Enum):
    PLANNING_ERROR = "F1_PLANNING_ERROR"
    TOOL_SELECTION_ERROR = "F2_TOOL_SELECTION_ERROR"
    ARGUMENT_ERROR = "F3_ARGUMENT_ERROR"
    TOOL_EXECUTION_ERROR = "F4_TOOL_EXECUTION_ERROR"
    OBSERVATION_ERROR = "F5_OBSERVATION_ERROR"
    STATE_DRIFT = "F6_STATE_DRIFT"
    MEMORY_ERROR = "F7_MEMORY_ERROR"
    LOOP_REPEATED_ACTION = "F8_LOOP_REPEATED_ACTION"
    PREMATURE_FINAL = "F9_PREMATURE_FINAL"
    BUDGET_EXHAUSTION = "F10_BUDGET_EXHAUSTION"
    ENVIRONMENT_ERROR = "F11_ENVIRONMENT_ERROR"
    PROTOCOL_ERROR = "F12_PROTOCOL_ERROR"


@dataclass(frozen=True)
class FailureEvent:
    failure_type: FailureType
    message: str
    step: int
    tool_name: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_type": self.failure_type.value,
            "message": self.message,
            "step": self.step,
            "tool_name": self.tool_name,
            "details": self.details,
        }


class FailureClassifier:
    _ERROR_MAP = {
        "unknown_tool": FailureType.TOOL_SELECTION_ERROR,
        "invalid_argument": FailureType.ARGUMENT_ERROR,
        "tool_execution": FailureType.TOOL_EXECUTION_ERROR,
        "observation_error": FailureType.OBSERVATION_ERROR,
        "environment_error": FailureType.ENVIRONMENT_ERROR,
        "protocol_error": FailureType.PROTOCOL_ERROR,
    }

    def classify_tool_result(
        self,
        state: AgentState,
        tool_name: str,
        error_type: str | None,
        message: str,
        repeated: bool = False,
    ) -> FailureEvent | None:
        if repeated:
            return FailureEvent(
                FailureType.LOOP_REPEATED_ACTION,
                "repeated tool action detected",
                state.steps_used,
                tool_name,
            )
        if error_type is None:
            return None
        failure_type = self._ERROR_MAP.get(error_type, FailureType.PROTOCOL_ERROR)
        return FailureEvent(failure_type, message, state.steps_used, tool_name)

    def state_drift(
        self,
        state: AgentState,
        expected: Mapping[str, Any],
        actual: Mapping[str, Any],
    ) -> FailureEvent | None:
        if dict(expected) == dict(actual):
            return None
        return FailureEvent(
            FailureType.STATE_DRIFT,
            "expected and observed state differ",
            state.steps_used,
            details={"expected": dict(expected), "actual": dict(actual)},
        )

    def budget_exhausted(self, state: AgentState) -> FailureEvent:
        return FailureEvent(
            FailureType.BUDGET_EXHAUSTION,
            "step or token budget exhausted before a final answer",
            state.steps_used,
            details={
                "steps_used": state.steps_used,
                "budget": state.budget,
                "tokens_used": state.tokens_used,
                "token_budget": state.token_budget,
            },
        )
