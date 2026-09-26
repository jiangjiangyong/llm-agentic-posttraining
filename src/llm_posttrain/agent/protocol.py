from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

ResponseKind = Literal["tool_call", "final", "invalid"]


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}


@dataclass(frozen=True)
class ParsedResponse:
    kind: ResponseKind
    text: str
    tool_call: ToolCall | None = None
    error: str | None = None


class ToolCallParser:
    """Parse canonical tool calls and tolerate trailing generation noise."""

    _tag_pattern = re.compile(
        r"<tool_call>\s*(.*?)\s*</tool_call>",
        flags=re.DOTALL | re.IGNORECASE,
    )
    _fence_pattern = re.compile(
        r"\x60\x60\x60(?:json)?\s*(.*?)\x60\x60\x60",
        flags=re.DOTALL | re.IGNORECASE,
    )
    _missing_expression_quote_pattern = re.compile(
        r'(?P<prefix>"expression"\s*:\s*")'
        r'(?P<value>[^"\r\n]*?)'
        r'(?P<closing>\s*\}\s*\})(?P<trailing>.*)\Z',
        flags=re.DOTALL,
    )

    def _payload_to_call(self, payload: Any) -> ToolCall | str | None:
        if not isinstance(payload, dict):
            return "tool payload must be a JSON object"
        name = payload.get("name")
        if name is None:
            name = payload.get("tool_name", payload.get("tool"))
        arguments = payload.get("arguments", payload.get("args"))
        if name is None and arguments is None:
            return None
        if not isinstance(name, str) or not name.strip():
            return "tool name must be a non-empty string"
        if not isinstance(arguments, dict):
            return "tool arguments must be a JSON object"
        return ToolCall(name=name.strip(), arguments=arguments)

    def _payload_result(self, payload: Any, text: str) -> ParsedResponse:
        result = self._payload_to_call(payload)
        if isinstance(result, str):
            return ParsedResponse(kind="invalid", text=text, error=result)
        if result is None:
            return ParsedResponse(kind="final", text=text)
        return ParsedResponse(kind="tool_call", text=text, tool_call=result)

    def _repair_missing_expression_quote(self, text: str) -> str | None:
        match = self._missing_expression_quote_pattern.search(text)
        if match is None:
            return None
        closing_start = match.start("closing")
        return f'{text[:closing_start]}"{text[closing_start:]}'

    def _parse_payload(self, text: str) -> ParsedResponse:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            repaired = self._repair_missing_expression_quote(text)
            if repaired is not None:
                try:
                    payload = json.loads(repaired)
                except json.JSONDecodeError:
                    pass
                else:
                    return self._payload_result(payload, text)
            return ParsedResponse(
                kind="invalid",
                text=text,
                error=f"invalid tool-call JSON: {exc}",
            )
        return self._payload_result(payload, text)

    def _parse_json_prefix(self, text: str) -> ParsedResponse:
        try:
            payload, _ = json.JSONDecoder().raw_decode(text)
        except json.JSONDecodeError as exc:
            repaired = self._repair_missing_expression_quote(text)
            if repaired is not None:
                try:
                    payload, _ = json.JSONDecoder().raw_decode(repaired)
                except json.JSONDecodeError:
                    pass
                else:
                    return self._payload_result(payload, text)
            return ParsedResponse(
                kind="invalid",
                text=text,
                error=f"invalid tool-call JSON: {exc}",
            )
        return self._payload_result(payload, text)

    def parse_many(self, text: str) -> list[ToolCall]:
        """Recover an ordered list from a legacy multi-action generation.

        Canonical generations still contain one tagged action. This helper is
        only used by the Code-Agent environment after the first action has
        already been recognized, so ordinary final JSON is not reinterpreted as
        a tool sequence.
        """
        raw = text if isinstance(text, str) else str(text)
        normalized = raw.strip()
        if normalized.startswith("|"):
            normalized = normalized[1:].lstrip()

        calls: list[ToolCall] = []
        tagged = list(self._tag_pattern.finditer(raw))
        if tagged:
            for match in tagged:
                parsed = self._parse_payload(match.group(1).strip())
                if parsed.kind == "tool_call" and parsed.tool_call is not None:
                    calls.append(parsed.tool_call)
            return calls

        cursor = 0
        decoder = json.JSONDecoder()
        while cursor < len(normalized):
            opening = normalized.find("{", cursor)
            if opening < 0:
                break
            try:
                payload, end = decoder.raw_decode(normalized[opening:])
            except json.JSONDecodeError:
                cursor = opening + 1
                continue
            result = self._payload_to_call(payload)
            if isinstance(result, ToolCall):
                calls.append(result)
            cursor = opening + max(end, 1)
        return calls

    def parse(self, text: str) -> ParsedResponse:
        raw = text if isinstance(text, str) else str(text)
        stripped = raw.strip()
        if not stripped:
            return ParsedResponse(kind="invalid", text=raw, error="empty model output")

        match = self._tag_pattern.search(raw)
        if match:
            return self._parse_payload(match.group(1).strip())
        if "<tool_call>" in raw.lower():
            return ParsedResponse(
                kind="invalid",
                text=raw,
                error="unterminated <tool_call> block",
            )

        normalized = stripped
        if normalized.startswith("|"):
            normalized = normalized[1:].lstrip()

        fenced = self._fence_pattern.fullmatch(normalized)
        candidate = fenced.group(1).strip() if fenced else normalized
        if candidate.startswith("{"):
            return self._parse_json_prefix(candidate)

        # Some base-model generations prepend a short answer marker before the
        # JSON action. Recover only JSON payloads that unambiguously describe a
        # tool call; ordinary prose and structured final JSON remain finals.
        for match in re.finditer(r"\{", normalized):
            if match.start() == 0:
                continue
            parsed = self._parse_json_prefix(normalized[match.start() :])
            if parsed.kind == "tool_call" and parsed.tool_call is not None:
                return ParsedResponse(
                    kind="tool_call",
                    text=raw,
                    tool_call=parsed.tool_call,
                )
        return ParsedResponse(kind="final", text=stripped)
