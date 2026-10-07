from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from llm_posttrain.agent.protocol import ToolCall, ToolCallParser


NormalizationStatus = Literal[
    "already_canonical",
    "legacy_to_canonical",
    "not_tool_call",
    "invalid_tool_call",
]


@dataclass(frozen=True)
class ProtocolNormalizationResult:
    """A loss-aware protocol normalization decision.

    The normalizer is an explicit protocol adapter, not a model-capability
    metric. Callers must keep ``raw_text`` and ``normalized_text`` separate.
    """

    raw_text: str
    normalized_text: str | None
    status: NormalizationStatus
    parsed_kind: str
    error: str | None = None

    @property
    def applied(self) -> bool:
        return self.status == "legacy_to_canonical"

    @property
    def canonical_available(self) -> bool:
        return self.status in {"already_canonical", "legacy_to_canonical"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "parsed_kind": self.parsed_kind,
            "applied": self.applied,
            "canonical_available": self.canonical_available,
            "error": self.error,
        }


class TextGenerationBackend(Protocol):
    def generate(self, messages: list[dict[str, Any]]) -> str:
        ...


class ProtocolNormalizingBackend:
    """Apply loss-aware protocol normalization at an execution boundary.

    The wrapped backend remains the source of the raw model output. The
    normalized text is only an executor-facing representation, and every
    decision is recorded so raw model behavior cannot be confused with
    canonical protocol learning.
    """

    def __init__(
        self,
        backend: TextGenerationBackend,
        normalizer: "CanonicalProtocolNormalizer | None" = None,
    ) -> None:
        self.backend = backend
        self.normalizer = normalizer or CanonicalProtocolNormalizer()
        self.records: list[ProtocolNormalizationResult] = []

    def generate(self, messages: list[dict[str, Any]]) -> str:
        raw = self.backend.generate(messages)
        result = self.normalizer.normalize(raw)
        self.records.append(result)
        return result.normalized_text if result.normalized_text is not None else raw


class CanonicalProtocolNormalizer:
    """Convert an already parseable tool call into the canonical wrapper.

    This class deliberately does not invent a tool name, arguments, or action
    from prose. It only serializes a ``ToolCall`` that the existing parser can
    already identify. Therefore any improvement measured with this adapter is
    protocol-layer reliability, not evidence that the model learned the
    canonical protocol.
    """

    _canonical_pattern = re.compile(
        r"^\s*<tool_call>\s*(.*?)\s*</tool_call>\s*$",
        flags=re.DOTALL | re.IGNORECASE,
    )

    def __init__(self, parser: ToolCallParser | None = None) -> None:
        self.parser = parser or ToolCallParser()

    @staticmethod
    def canonical_text(call: ToolCall) -> str:
        payload = json.dumps(
            {"name": call.name, "arguments": call.arguments},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return f"<tool_call>{payload}</tool_call>"

    @classmethod
    def is_canonical(cls, text: str) -> bool:
        match = cls._canonical_pattern.fullmatch(str(text))
        if match is None:
            return False
        try:
            payload = json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            return False
        return (
            isinstance(payload, dict)
            and set(payload) == {"name", "arguments"}
            and isinstance(payload["name"], str)
            and bool(payload["name"].strip())
            and isinstance(payload["arguments"], dict)
        )

    def normalize(self, text: str) -> ProtocolNormalizationResult:
        raw = text if isinstance(text, str) else str(text)
        if self.is_canonical(raw):
            return ProtocolNormalizationResult(
                raw_text=raw,
                normalized_text=raw,
                status="already_canonical",
                parsed_kind="tool_call",
            )

        parsed = self.parser.parse(raw)
        if parsed.kind != "tool_call" or parsed.tool_call is None:
            status: NormalizationStatus = (
                "invalid_tool_call" if parsed.kind == "invalid" else "not_tool_call"
            )
            return ProtocolNormalizationResult(
                raw_text=raw,
                normalized_text=None,
                status=status,
                parsed_kind=parsed.kind,
                error=parsed.error,
            )

        return ProtocolNormalizationResult(
            raw_text=raw,
            normalized_text=self.canonical_text(parsed.tool_call),
            status="legacy_to_canonical",
            parsed_kind=parsed.kind,
            error=parsed.error,
        )
