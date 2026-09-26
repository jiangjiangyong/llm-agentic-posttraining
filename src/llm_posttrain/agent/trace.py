from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TraceEvent:
    step: int
    kind: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "kind": self.kind,
            "payload": self.payload,
        }


@dataclass
class AgentTrace:
    run_id: str
    task_id: str
    initial_messages: list[dict[str, Any]]
    prompt_messages: list[dict[str, Any]]
    events: list[TraceEvent] = field(default_factory=list)
    final_answer: str | None = None
    success: bool = False
    error: str | None = None

    def add_event(self, step: int, kind: str, payload: dict[str, Any]) -> None:
        self.events.append(TraceEvent(step=step, kind=kind, payload=payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "initial_messages": self.initial_messages,
            "prompt_messages": self.prompt_messages,
            "events": [event.to_dict() for event in self.events],
            "final_answer": self.final_answer,
            "success": self.success,
            "error": self.error,
        }

    def save_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
