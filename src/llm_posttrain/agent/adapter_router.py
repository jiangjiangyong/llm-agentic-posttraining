"""Protocol-aware adapter routing for isolated capability evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AdapterRoute:
    name: str
    adapter_path: str
    protocol: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "adapter_path": self.adapter_path,
            "protocol": self.protocol,
            "reason": self.reason,
        }


class ProtocolAdapterRouter:
    """Choose an adapter from explicit protocol markers only.

    The router never reads expected actions, target answers, categories, or
    holdout data. Ambiguous protocol prompts fail closed instead of silently
    selecting one capability.
    """

    RICH_MARKER = "CodeToolAgent Protocol v2"
    TOOL_MARKER = "Tool Calling Protocol v1"

    def __init__(
        self,
        *,
        rich_adapter: str,
        protocol_adapter: str,
        fallback_adapter: str,
    ) -> None:
        self.routes = {
            "rich_v2": AdapterRoute(
                name="rich_v2",
                adapter_path=rich_adapter,
                protocol=self.RICH_MARKER,
                reason="explicit CodeToolAgent Protocol v2 marker",
            ),
            "tool_v1": AdapterRoute(
                name="tool_v1",
                adapter_path=protocol_adapter,
                protocol=self.TOOL_MARKER,
                reason="explicit Tool Calling Protocol v1 marker",
            ),
            "stable_fallback": AdapterRoute(
                name="stable_fallback",
                adapter_path=fallback_adapter,
                protocol="none",
                reason="no recognized protocol marker",
            ),
        }

    @staticmethod
    def _system_text(messages: list[dict[str, Any]]) -> str:
        return "\n".join(
            str(message.get("content", ""))
            for message in messages
            if message.get("role") == "system"
        )

    def route(self, messages: list[dict[str, Any]]) -> AdapterRoute:
        text = self._system_text(messages)
        rich = self.RICH_MARKER in text
        tool = self.TOOL_MARKER in text
        if rich and tool:
            raise ValueError("ambiguous protocol markers; refusing to route")
        if rich:
            return self.routes["rich_v2"]
        if tool:
            return self.routes["tool_v1"]
        return self.routes["stable_fallback"]
