"""Deterministic progress state for research-to-file agent trajectories."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

STATE_SYSTEM_SUFFIX = (
    " When a <progress_state> block is attached to a tool observation, treat it "
    "as execution state. Follow pending_step and do not report completion until "
    "termination_allowed is true."
)


@dataclass
class ResearchProgressTracker:
    evidence_acquired: bool = False
    file_written: bool = False
    file_verified: bool = False

    def update(self, tool_name: str, result: dict[str, Any]) -> str:
        ok = bool(result.get("ok"))
        output = result.get("output")
        if not isinstance(output, dict):
            output = {}
        if tool_name == "mock_search" and ok:
            results = output.get("results")
            self.evidence_acquired = isinstance(results, list) and bool(results)
        elif tool_name == "write_file" and ok:
            self.file_written = bool(output.get("path"))
        elif tool_name == "read_file" and ok:
            self.file_verified = output.get("content") is not None

        if self.evidence_acquired and self.file_written and self.file_verified:
            pending_step = "final"
            termination_allowed = True
        elif self.evidence_acquired and self.file_written:
            pending_step = "read_file"
            termination_allowed = False
        elif self.evidence_acquired:
            pending_step = "write_file"
            termination_allowed = False
        else:
            pending_step = "mock_search"
            termination_allowed = False

        return chr(10).join(
            [
                f"evidence_acquired: {str(self.evidence_acquired).lower()}",
                f"file_written: {str(self.file_written).lower()}",
                f"file_verified: {str(self.file_verified).lower()}",
                f"pending_step: {pending_step}",
                f"termination_allowed: {str(termination_allowed).lower()}",
            ]
        )


def append_progress_state(content: str, state: str) -> str:
    return (
        str(content).rstrip()
        + chr(10)
        + "<progress_state>"
        + chr(10)
        + state
        + chr(10)
        + "</progress_state>"
    )


def add_state_instruction(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = [dict(message) for message in messages]
    for message in result:
        if message.get("role") == "system":
            content = str(message.get("content", ""))
            if STATE_SYSTEM_SUFFIX not in content:
                message["content"] = content.rstrip() + chr(10) + STATE_SYSTEM_SUFFIX
            break
    return result
