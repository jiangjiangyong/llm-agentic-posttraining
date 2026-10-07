"""Small JSON checkpoint store for the first CPU-only implementation."""

from __future__ import annotations

import copy
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .state import AgentState, StateTransition


@dataclass(frozen=True)
class Checkpoint:
    checkpoint_id: str
    reason: str
    state: dict[str, Any]
    transitions: list[dict[str, Any]]
    workspace_metadata: dict[str, Any]
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "reason": self.reason,
            "state": copy.deepcopy(self.state),
            "transitions": copy.deepcopy(self.transitions),
            "workspace_metadata": copy.deepcopy(self.workspace_metadata),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Checkpoint":
        return cls(
            checkpoint_id=str(payload["checkpoint_id"]),
            reason=str(payload["reason"]),
            state=copy.deepcopy(dict(payload["state"])),
            transitions=copy.deepcopy(list(payload.get("transitions", []))),
            workspace_metadata=copy.deepcopy(dict(payload.get("workspace_metadata", {}))),
            created_at=float(payload["created_at"]),
        )


class JsonCheckpointStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def make_id(self, task_id: str) -> str:
        safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)
        return f"{safe_task_id}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"

    def save(
        self,
        state: AgentState,
        reason: str,
        transitions: Iterable[StateTransition | Mapping[str, Any]] = (),
        workspace_metadata: Mapping[str, Any] | None = None,
        checkpoint_id: str | None = None,
    ) -> Checkpoint:
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id or self.make_id(state.task_id),
            reason=reason,
            state=state.to_dict(),
            transitions=[
                item.to_dict() if isinstance(item, StateTransition) else copy.deepcopy(dict(item))
                for item in transitions
            ],
            workspace_metadata=copy.deepcopy(dict(workspace_metadata or {})),
            created_at=time.time(),
        )
        destination = self.root / f"{checkpoint.checkpoint_id}.json"
        temporary = self.root / f".{checkpoint.checkpoint_id}.tmp"
        temporary.write_text(
            json.dumps(checkpoint.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, destination)
        return checkpoint

    def load(self, checkpoint_id: str) -> Checkpoint:
        path = self.root / f"{checkpoint_id}.json"
        return Checkpoint.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_ids(self, task_id: str | None = None) -> list[str]:
        paths = sorted(self.root.glob("*.json"))
        ids = [path.stem for path in paths]
        if task_id is None:
            return ids
        prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id) + "-"
        return [checkpoint_id for checkpoint_id in ids if checkpoint_id.startswith(prefix)]
