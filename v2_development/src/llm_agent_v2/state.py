"""Typed state and auditable state transitions."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Mapping


IMMUTABLE_FIELDS = ("task_id", "goal", "environment_id")


@dataclass
class AgentState:
    """Runtime state with explicit immutable, mutable and derived fields."""

    task_id: str
    goal: str
    environment_id: str = "cpu-local"

    completed_actions: list[str] = field(default_factory=list)
    pending_actions: list[str] = field(default_factory=list)
    tool_observations: list[dict[str, Any]] = field(default_factory=list)
    workspace_state: dict[str, Any] = field(default_factory=dict)
    evidence_refs: list[str] = field(default_factory=list)
    memory_refs: list[str] = field(default_factory=list)
    current_subgoal: str | None = None
    last_tool_status: str | None = None
    failure_history: list[dict[str, Any]] = field(default_factory=list)
    retry_count: int = 0

    budget: int = 8
    steps_used: int = 0
    token_budget: int = 4096
    tokens_used: int = 0
    uncertainty: float = 0.0
    checkpoint_id: str | None = None
    termination_allowed: bool = False
    version: int = 0

    @property
    def remaining_budget(self) -> int:
        return max(0, self.budget - self.steps_used)

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.token_budget - self.tokens_used)

    def clone(self) -> "AgentState":
        return AgentState.from_dict(self.to_dict())

    def immutable_signature(self) -> tuple[str, str, str]:
        return tuple(getattr(self, name) for name in IMMUTABLE_FIELDS)  # type: ignore[return-value]

    def assert_immutable_unchanged(self, before: "AgentState") -> None:
        if self.immutable_signature() != before.immutable_signature():
            raise ValueError("immutable AgentState fields changed during transition")

    def apply_tool_observation(
        self,
        action_label: str,
        success: bool,
        output: str,
        state_patch: Mapping[str, Any] | None = None,
    ) -> None:
        """Apply an observation while keeping state mutation explicit."""

        if state_patch:
            self._apply_patch(state_patch)

        self.completed_actions.append(action_label)
        self.tool_observations.append(
            {
                "action": action_label,
                "success": success,
                "output": output,
                "step": self.steps_used + 1,
            }
        )
        self.last_tool_status = "success" if success else "failure"
        self.steps_used += 1
        self.version += 1

    def _apply_patch(self, state_patch: Mapping[str, Any]) -> None:
        for key, value in state_patch.items():
            if key in IMMUTABLE_FIELDS:
                raise ValueError(f"state patch cannot change immutable field: {key}")
            if key == "workspace_state":
                if not isinstance(value, Mapping):
                    raise ValueError("workspace_state patch must be a mapping")
                self.workspace_state.update(copy.deepcopy(dict(value)))
            elif not hasattr(self, key):
                raise ValueError(f"unknown AgentState field in patch: {key}")
            else:
                setattr(self, key, copy.deepcopy(value))

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "task_id": self.task_id,
                "goal": self.goal,
                "environment_id": self.environment_id,
                "completed_actions": self.completed_actions,
                "pending_actions": self.pending_actions,
                "tool_observations": self.tool_observations,
                "workspace_state": self.workspace_state,
                "evidence_refs": self.evidence_refs,
                "memory_refs": self.memory_refs,
                "current_subgoal": self.current_subgoal,
                "last_tool_status": self.last_tool_status,
                "failure_history": self.failure_history,
                "retry_count": self.retry_count,
                "budget": self.budget,
                "steps_used": self.steps_used,
                "token_budget": self.token_budget,
                "tokens_used": self.tokens_used,
                "uncertainty": self.uncertainty,
                "checkpoint_id": self.checkpoint_id,
                "termination_allowed": self.termination_allowed,
                "version": self.version,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AgentState":
        known = {field_name for field_name in cls.__dataclass_fields__}
        values = {key: copy.deepcopy(value) for key, value in payload.items() if key in known}
        return cls(**values)


def compute_state_diff(before: AgentState, after: AgentState) -> dict[str, dict[str, Any]]:
    """Return a JSON-safe top-level diff between two state snapshots."""

    before_data = before.to_dict()
    after_data = after.to_dict()
    diff: dict[str, dict[str, Any]] = {}
    for key in before_data.keys() | after_data.keys():
        if before_data.get(key) != after_data.get(key):
            diff[key] = {
                "before": before_data.get(key),
                "after": after_data.get(key),
            }
    return diff


@dataclass(frozen=True)
class StateTransition:
    """One auditable State_t -> Action_t -> Observation_t -> State_t+1 step."""

    before_state: dict[str, Any]
    action: dict[str, Any]
    observation: dict[str, Any]
    after_state: dict[str, Any]
    state_diff: dict[str, dict[str, Any]]

    @classmethod
    def from_states(
        cls,
        before: AgentState,
        action: Mapping[str, Any],
        observation: Mapping[str, Any],
        after: AgentState,
    ) -> "StateTransition":
        after.assert_immutable_unchanged(before)
        return cls(
            before_state=before.to_dict(),
            action=copy.deepcopy(dict(action)),
            observation=copy.deepcopy(dict(observation)),
            after_state=after.to_dict(),
            state_diff=compute_state_diff(before, after),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "before_state": copy.deepcopy(self.before_state),
            "action": copy.deepcopy(self.action),
            "observation": copy.deepcopy(self.observation),
            "after_state": copy.deepcopy(self.after_state),
            "state_diff": copy.deepcopy(self.state_diff),
        }


def state_json(state: AgentState) -> str:
    """Stable JSON representation useful for trace snapshots and tests."""

    return json.dumps(state.to_dict(), ensure_ascii=False, sort_keys=True)
