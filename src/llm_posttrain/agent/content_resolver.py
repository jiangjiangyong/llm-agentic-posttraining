from __future__ import annotations

import copy
import json
import re
from typing import Any, Iterable


_OBSERVATION_PATTERN = re.compile(
    r"<tool_observation>\s*(.*?)\s*</tool_observation>",
    flags=re.IGNORECASE | re.DOTALL,
)
_CONTEXT_PACKET_PATTERN = re.compile(
    r"Context\s+packet\s+([A-Za-z0-9_-]+)\s*:",
    flags=re.IGNORECASE,
)


def _message_texts(messages: Iterable[dict[str, Any]]) -> Iterable[str]:
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            yield content


def _context_marker(messages: Iterable[dict[str, Any]]) -> str | None:
    """Read the case marker from the task prompt, never from model output."""

    for content in _message_texts(messages):
        match = _CONTEXT_PACKET_PATTERN.search(content)
        if match:
            return match.group(1)
    return None


def _latest_search_snippet(
    messages: Iterable[dict[str, Any]],
) -> tuple[str, str | None] | None:
    """Return the latest successful mock-search snippet in tool observations.

    Only structured tool_observation payloads are trusted. Assistant text,
    expected targets, and arbitrary task fields are deliberately ignored.
    """

    observations: list[str] = []
    for content in _message_texts(messages):
        observations.extend(
            match.group(1) for match in _OBSERVATION_PATTERN.finditer(content)
        )

    for raw_payload in reversed(observations):
        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            continue
        if payload.get("tool_name") != "mock_search" or payload.get("ok") is not True:
            continue
        output = payload.get("output")
        if not isinstance(output, dict):
            continue
        results = output.get("results")
        if not isinstance(results, list):
            continue
        for result in results:
            if not isinstance(result, dict):
                continue
            snippet = result.get("snippet")
            if isinstance(snippet, str) and snippet.strip():
                result_id = result.get("id")
                return snippet.strip(), result_id if isinstance(result_id, str) else None
    return None


def _is_content_pointer(call: dict[str, Any]) -> bool:
    if call.get("name") != "write_file":
        return False
    arguments = call.get("arguments")
    if not isinstance(arguments, dict):
        return False
    return (
        "content" not in arguments
        and arguments.get("content_ref") == "result_snippet"
        and arguments.get("marker_ref") == "context_packet"
    )


class ContentRefResolver:
    """Materialize the narrow, allow-listed content_ref action protocol.

    The resolver only handles a write_file pointer to the latest successful
    mock_search snippet plus a case marker in the original task prompt. It
    never reads expected actions or holdout targets, and it leaves all other
    tool calls untouched.
    """

    def resolve(
        self,
        call: dict[str, Any],
        *,
        task_messages: list[dict[str, Any]],
        working_messages: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        raw_call = copy.deepcopy(call)
        if not _is_content_pointer(raw_call):
            return raw_call, None

        pointer = _latest_search_snippet(working_messages)
        marker = _context_marker(task_messages)
        if pointer is None or marker is None:
            # Leave the pointer untouched so normal schema validation emits a
            # visible failure instead of silently inventing file content.
            return raw_call, {
                "resolver": "content_ref_protocol_v1",
                "applied": False,
                "reason": "missing_verified_search_snippet_or_context_marker",
            }

        snippet, result_id = pointer
        arguments = dict(raw_call.get("arguments") or {})
        resolved = {
            "name": "write_file",
            "arguments": {
                "path": arguments.get("path"),
                "content": f"{snippet}\nCase marker: {marker}\n",
            },
        }
        return resolved, {
            "resolver": "content_ref_protocol_v1",
            "applied": True,
            "source": "latest_successful_mock_search_observation",
            "result_id": result_id,
            "marker": marker,
            "snippet_chars": len(snippet),
        }
