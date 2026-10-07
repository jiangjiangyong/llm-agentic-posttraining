from __future__ import annotations

import copy
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from llm_posttrain.tools.registry import ToolRegistry

from .protocol import ToolCallParser
from .trace import AgentTrace


class TextGenerationBackend(Protocol):
    def generate(self, messages: list[dict[str, Any]]) -> str:
        ...


@dataclass(frozen=True)
class AgentRunResult:
    trace: AgentTrace
    final_answer: str | None
    success: bool
    error: str | None
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_answer": self.final_answer,
            "success": self.success,
            "error": self.error,
            "elapsed_ms": self.elapsed_ms,
            "trace": self.trace.to_dict(),
        }


class AgentRuntime:
    def __init__(
        self,
        backend: TextGenerationBackend,
        registry: ToolRegistry,
        *,
        parser: ToolCallParser | None = None,
        max_steps: int = 3,
        observation_role: str = "tool",
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if observation_role not in {"tool", "user"}:
            raise ValueError("observation_role must be 'tool' or 'user'")
        self.backend = backend
        self.registry = registry
        self.parser = parser or ToolCallParser()
        self.max_steps = max_steps
        self.observation_role = observation_role

    def _tool_instruction(self) -> str:
        schema_text = json.dumps(
            self.registry.schemas(),
            ensure_ascii=False,
            indent=2,
        )
        return (
            "You are running inside Tool Calling Protocol v1.\n"
            "Use the calculator when arithmetic is required.\n"
            "Available tools:\n"
            f"{schema_text}\n"
            "When a tool is needed, output exactly one block:\n"
            '<tool_call>{"name":"calculator","arguments":{"expression":"2 + 2"}}</tool_call>\n'
            "After the tool observation, output the final answer in plain text."
        )

    @staticmethod
    def _contains_protocol_instruction(content: str) -> bool:
        return (
            "You are running inside Tool Calling Protocol v1." in content
            and "<tool_call>" in content
            and "</tool_call>" in content
        )

    def _prepare_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        prepared = copy.deepcopy(messages)
        instruction = self._tool_instruction()
        system_index = next(
            (
                index
                for index, message in enumerate(prepared)
                if message.get("role") == "system"
            ),
            None,
        )
        if system_index is None:
            prepared.insert(0, {"role": "system", "content": instruction})
        else:
            original = str(prepared[system_index].get("content", "")).rstrip()
            if not self._contains_protocol_instruction(original):
                prepared[system_index]["content"] = f"{original}\n\n{instruction}"
        return prepared

    def _observation_message(
        self,
        tool_name: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        content = (
            "<tool_observation>\n"
            + json.dumps(result, ensure_ascii=False)
            + "\n</tool_observation>"
        )
        message: dict[str, Any] = {
            "role": self.observation_role,
            "content": content,
        }
        if self.observation_role == "tool":
            message["name"] = tool_name
        return message

    def run(
        self,
        messages: list[dict[str, Any]],
        *,
        task_id: str = "anonymous",
    ) -> AgentRunResult:
        started = time.perf_counter()
        prepared = self._prepare_messages(messages)
        working_messages = copy.deepcopy(prepared)
        trace = AgentTrace(
            run_id=uuid.uuid4().hex,
            task_id=task_id,
            initial_messages=copy.deepcopy(messages),
            prompt_messages=copy.deepcopy(prepared),
        )

        for step in range(self.max_steps):
            raw_output = self.backend.generate(copy.deepcopy(working_messages))
            if not isinstance(raw_output, str):
                raw_output = str(raw_output)
            trace.add_event(
                step=step,
                kind="model_output",
                payload={"text": raw_output},
            )
            parsed = self.parser.parse(raw_output)
            if parsed.kind == "tool_call" and parsed.tool_call is not None:
                call = parsed.tool_call
                trace.add_event(
                    step=step,
                    kind="tool_call",
                    payload=call.to_dict(),
                )
                execution = self.registry.execute(call.name, call.arguments)
                trace.add_event(
                    step=step,
                    kind="tool_result",
                    payload=execution.to_dict(),
                )
                working_messages.append({"role": "assistant", "content": raw_output})
                working_messages.append(
                    self._observation_message(call.name, execution.to_dict())
                )
                continue

            if parsed.kind == "final":
                trace.final_answer = parsed.text
                trace.success = True
                trace.add_event(
                    step=step,
                    kind="final_answer",
                    payload={"text": parsed.text},
                )
                return AgentRunResult(
                    trace=trace,
                    final_answer=parsed.text,
                    success=True,
                    error=None,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                )

            error = parsed.error or "invalid model output"
            trace.error = error
            trace.add_event(
                step=step,
                kind="parse_error",
                payload={"error": error},
            )
            return AgentRunResult(
                trace=trace,
                final_answer=None,
                success=False,
                error=error,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )

        error = f"maximum agent steps exceeded: {self.max_steps}"
        trace.error = error
        trace.add_event(
            step=self.max_steps,
            kind="runtime_error",
            payload={"error": error},
        )
        return AgentRunResult(
            trace=trace,
            final_answer=None,
            success=False,
            error=error,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
