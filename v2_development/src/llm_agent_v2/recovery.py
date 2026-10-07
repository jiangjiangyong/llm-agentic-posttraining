"""Failure-aware recovery decisions with explicit evidence boundaries."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .failure import FailureEvent, FailureType
from .state import AgentState


class RecoveryAction(str, Enum):
    RETRY = "retry"
    REPAIR_ARGUMENTS = "repair_arguments"
    RESTORE_CHECKPOINT = "restore_checkpoint"
    REPLAN = "replan"
    BREAK_LOOP = "break_loop"
    REFRESH_OBSERVATION = "refresh_observation"
    ABSTAIN = "abstain"
    HUMAN_CONFIRMATION = "human_confirmation"
    NORMALIZE_PROTOCOL = "normalize_protocol"
    NONE = "none"


@dataclass(frozen=True)
class RecoveryDecision:
    failure_type: FailureType
    action: RecoveryAction
    allowed: bool
    reason: str
    max_additional_steps: int = 0


@dataclass(frozen=True)
class RecoveryRecord:
    failure_type: str
    recovery_action: str
    before_state: dict[str, Any]
    after_state: dict[str, Any]
    recovery_success: bool
    additional_steps: int
    additional_tokens: int
    latency_ms: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_type": self.failure_type,
            "recovery_action": self.recovery_action,
            "before_state": copy.deepcopy(self.before_state),
            "after_state": copy.deepcopy(self.after_state),
            "recovery_success": self.recovery_success,
            "additional_steps": self.additional_steps,
            "additional_tokens": self.additional_tokens,
            "latency_ms": self.latency_ms,
            "reason": self.reason,
        }


class RecoveryPolicy:
    def decide(
        self,
        event: FailureEvent,
        state: AgentState,
        attempt_count: int = 0,
    ) -> RecoveryDecision:
        failure_type = event.failure_type
        if failure_type is FailureType.ARGUMENT_ERROR:
            return RecoveryDecision(failure_type, RecoveryAction.REPLAN, True, "ask the model to re-plan from the tool error")
        if failure_type is FailureType.TOOL_EXECUTION_ERROR:
            if attempt_count < 1:
                return RecoveryDecision(failure_type, RecoveryAction.RETRY, True, "one bounded retry is allowed", 1)
            return RecoveryDecision(failure_type, RecoveryAction.ABSTAIN, True, "retry limit reached")
        if failure_type is FailureType.STATE_DRIFT:
            action = RecoveryAction.RESTORE_CHECKPOINT if state.checkpoint_id else RecoveryAction.REPLAN
            return RecoveryDecision(failure_type, action, True, "restore the last known state or re-plan")
        if failure_type is FailureType.LOOP_REPEATED_ACTION:
            return RecoveryDecision(failure_type, RecoveryAction.BREAK_LOOP, True, "stop repeated action propagation")
        if failure_type is FailureType.OBSERVATION_ERROR:
            return RecoveryDecision(failure_type, RecoveryAction.REFRESH_OBSERVATION, True, "request a fresh observation")
        if failure_type is FailureType.PROTOCOL_ERROR:
            return RecoveryDecision(failure_type, RecoveryAction.NORMALIZE_PROTOCOL, True, "normalize the model/tool protocol")
        if failure_type is FailureType.BUDGET_EXHAUSTION:
            return RecoveryDecision(failure_type, RecoveryAction.ABSTAIN, True, "do not exceed the declared budget")
        if failure_type is FailureType.ENVIRONMENT_ERROR:
            return RecoveryDecision(failure_type, RecoveryAction.ABSTAIN, True, "environment errors require an explicit boundary")
        return RecoveryDecision(failure_type, RecoveryAction.REPLAN, True, "re-plan using current state and history")


class RecoveryEngine:
    def __init__(self, policy: RecoveryPolicy | None = None) -> None:
        self.policy = policy or RecoveryPolicy()

    def recover(
        self,
        event: FailureEvent,
        state: AgentState,
        attempt_count: int = 0,
    ) -> tuple[RecoveryDecision, RecoveryRecord]:
        started = time.perf_counter()
        before = state.to_dict()
        decision = self.policy.decide(event, state, attempt_count)
        if decision.action in (RecoveryAction.BREAK_LOOP, RecoveryAction.ABSTAIN):
            state.termination_allowed = True
            state.last_tool_status = f"recovery:{decision.action.value}"
        if decision.action is RecoveryAction.RETRY:
            state.retry_count += 1
        after = state.to_dict()
        record = RecoveryRecord(
            failure_type=event.failure_type.value,
            recovery_action=decision.action.value,
            before_state=before,
            after_state=after,
            recovery_success=decision.allowed,
            additional_steps=decision.max_additional_steps,
            additional_tokens=0,
            latency_ms=(time.perf_counter() - started) * 1000,
            reason=decision.reason,
        )
        return decision, record
