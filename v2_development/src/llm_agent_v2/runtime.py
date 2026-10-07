"""Minimal auditable Agent Runtime for CPU-first development."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from .checkpoint import JsonCheckpointStore
from .failure import FailureClassifier, FailureEvent, FailureType
from .models import GpuRequiredError, GenerationRequest, ModelAdapter, ToolCall
from .recovery import RecoveryAction, RecoveryEngine, RecoveryRecord
from .state import AgentState, StateTransition
from .tools import ToolRegistry, ToolResult


@dataclass(frozen=True)
class RuntimeConfig:
    max_steps: int = 8
    token_budget: int = 4096
    checkpoint_on_tool_success: bool = True
    checkpoint_on_recovery: bool = True


@dataclass
class AgentRunResult:
    status: str
    final_answer: str | None
    state: AgentState
    transitions: list[StateTransition] = field(default_factory=list)
    failures: list[FailureEvent] = field(default_factory=list)
    recoveries: list[RecoveryRecord] = field(default_factory=list)
    checkpoint_ids: list[str] = field(default_factory=list)
    error: str | None = None
    latency_ms: float = 0.0

    def metrics(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "task_success": self.status == "success",
            "steps": self.state.steps_used,
            "tool_calls": len(self.transitions),
            "failures": len(self.failures),
            "recoveries": len(self.recoveries),
            "checkpoints": len(self.checkpoint_ids),
            "tokens_used": self.state.tokens_used,
            "latency_ms": round(self.latency_ms, 3),
            "failure_types": [event.failure_type.value for event in self.failures],
            "recovery_actions": [record.recovery_action for record in self.recoveries],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "final_answer": self.final_answer,
            "state": self.state.to_dict(),
            "transitions": [item.to_dict() for item in self.transitions],
            "failures": [item.to_dict() for item in self.failures],
            "recoveries": [item.to_dict() for item in self.recoveries],
            "checkpoint_ids": list(self.checkpoint_ids),
            "error": self.error,
            "latency_ms": self.latency_ms,
            "metrics": self.metrics(),
        }


class AgentRuntime:
    def __init__(
        self,
        model: ModelAdapter,
        tools: ToolRegistry,
        checkpoint_store: JsonCheckpointStore | None = None,
        config: RuntimeConfig | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.checkpoint_store = checkpoint_store
        self.config = config or RuntimeConfig()
        self.failure_classifier = FailureClassifier()
        self.recovery_engine = RecoveryEngine()

    def run(
        self,
        task_id: str,
        goal: str,
        initial_state: AgentState | None = None,
    ) -> AgentRunResult:
        started = time.perf_counter()
        state = initial_state.clone() if initial_state else AgentState(
            task_id=task_id,
            goal=goal,
            budget=self.config.max_steps,
            token_budget=self.config.token_budget,
        )
        state.assert_immutable_unchanged(
            AgentState(task_id=task_id, goal=goal, environment_id=state.environment_id)
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": goal}]
        transitions: list[StateTransition] = []
        failures: list[FailureEvent] = []
        recoveries: list[RecoveryRecord] = []
        checkpoint_ids: list[str] = []
        action_counts: dict[str, int] = {}

        for _ in range(self.config.max_steps):
            if state.termination_allowed:
                break
            if state.remaining_tokens <= 0:
                break
            request = GenerationRequest(
                messages=tuple(copy.deepcopy(messages)),
                state=state.clone(),
                tool_specs=tuple(self.tools.specs()),
                max_new_tokens=min(512, state.remaining_tokens),
            )
            try:
                response = self.model.generate(request)
            except GpuRequiredError as exc:
                return self._result(
                    status="gpu_required",
                    state=state,
                    transitions=transitions,
                    failures=failures,
                    recoveries=recoveries,
                    checkpoint_ids=checkpoint_ids,
                    error=str(exc),
                    started=started,
                )
            except Exception as exc:  # model boundary is part of runtime reliability
                return self._result(
                    status="model_error",
                    state=state,
                    transitions=transitions,
                    failures=failures,
                    recoveries=recoveries,
                    checkpoint_ids=checkpoint_ids,
                    error=f"{type(exc).__name__}: {exc}",
                    started=started,
                )

            token_count = response.usage.get("total_tokens", 0)
            if token_count <= 0:
                token_count = max(1, len(response.content or "") // 4)
            state.tokens_used += token_count

            if not response.tool_calls:
                final_answer = response.content or ""
                state.termination_allowed = True
                return self._result(
                    status="success",
                    state=state,
                    final_answer=final_answer,
                    transitions=transitions,
                    failures=failures,
                    recoveries=recoveries,
                    checkpoint_ids=checkpoint_ids,
                    started=started,
                )

            if len(response.tool_calls) > 1:
                event = FailureEvent(
                    failure_type=FailureType.PROTOCOL_ERROR,
                    message="CPU foundation executes one tool call per runtime step",
                    step=state.steps_used,
                    details={"tool_call_count": len(response.tool_calls)},
                )
                failures.append(event)
                state.failure_history.append(event.to_dict())
                _, recovery = self.recovery_engine.recover(event, state)
                recoveries.append(recovery)
                break

            call = response.tool_calls[0]
            action_key = f"{call.name}:{json.dumps(call.arguments, ensure_ascii=False, sort_keys=True)}"
            repeated = action_key in action_counts
            action_counts[action_key] = action_counts.get(action_key, 0) + 1
            before = state.clone()
            result = self.tools.invoke(
                call.name,
                call.arguments,
                context={"workspace_state": copy.deepcopy(state.workspace_state), "state": state.to_dict()},
            )
            state.apply_tool_observation(
                action_label=action_key,
                success=result.success,
                output=result.output,
                state_patch=result.state_patch,
            )
            event = self.failure_classifier.classify_tool_result(
                state=state,
                tool_name=call.name,
                error_type=result.error_type,
                message=result.output,
                repeated=repeated,
            )
            if event:
                failures.append(event)
                state.failure_history.append(event.to_dict())
                _, recovery = self.recovery_engine.recover(
                    event,
                    state,
                    attempt_count=action_counts[action_key] - 1,
                )
                recoveries.append(recovery)
                if self.checkpoint_store and self.config.checkpoint_on_recovery:
                    self._save_checkpoint(
                        state,
                        "recovery:" + recovery.recovery_action,
                        transitions,
                        checkpoint_ids,
                    )
            transition = StateTransition.from_states(
                before=before,
                action={"type": "tool_call", **call.to_dict()},
                observation={"tool_result": result.to_dict(), "failure": event.to_dict() if event else None},
                after=state,
            )
            transitions.append(transition)
            messages.append({"role": "assistant", "tool_call": call.to_dict()})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.call_id,
                    "name": call.name,
                    "content": result.output,
                    "success": result.success,
                }
            )
            if self.checkpoint_store and result.success and self.config.checkpoint_on_tool_success:
                self._save_checkpoint(state, "tool_success", transitions, checkpoint_ids)

        budget_event = self.failure_classifier.budget_exhausted(state)
        if not state.termination_allowed:
            failures.append(budget_event)
            state.failure_history.append(budget_event.to_dict())
            status = "budget_exhausted"
        else:
            status = "abstained"
        return self._result(
            status=status,
            state=state,
            transitions=transitions,
            failures=failures,
            recoveries=recoveries,
            checkpoint_ids=checkpoint_ids,
            started=started,
        )

    def _save_checkpoint(
        self,
        state: AgentState,
        reason: str,
        transitions: list[StateTransition],
        checkpoint_ids: list[str],
    ) -> None:
        if not self.checkpoint_store:
            return
        checkpoint_id = self.checkpoint_store.make_id(state.task_id)
        state.checkpoint_id = checkpoint_id
        checkpoint = self.checkpoint_store.save(
            state,
            reason,
            transitions=transitions,
            workspace_metadata={"environment_id": state.environment_id},
            checkpoint_id=checkpoint_id,
        )
        checkpoint_ids.append(checkpoint.checkpoint_id)

    @staticmethod
    def _result(
        status: str,
        state: AgentState,
        transitions: list[StateTransition],
        failures: list[FailureEvent],
        recoveries: list[RecoveryRecord],
        checkpoint_ids: list[str],
        started: float,
        final_answer: str | None = None,
        error: str | None = None,
    ) -> AgentRunResult:
        return AgentRunResult(
            status=status,
            final_answer=final_answer,
            state=state,
            transitions=transitions,
            failures=failures,
            recoveries=recoveries,
            checkpoint_ids=checkpoint_ids,
            error=error,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
