from __future__ import annotations

import copy
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from llm_posttrain.rewards.calculator import CalculatorReward, RewardBreakdown
from llm_posttrain.tools.registry import ToolRegistry

from .protocol import ToolCallParser


@dataclass(frozen=True)
class EpisodeResult:
    episode_id: str
    task_id: str
    initial_messages: list[dict[str, Any]]
    steps: list[dict[str, Any]]
    tool_output: str
    final_output: str | None
    execution: dict[str, Any] | None
    reward: RewardBreakdown
    semantic_success: bool
    strict_success: bool
    elapsed_ms: float

    @property
    def success(self) -> bool:
        return self.semantic_success

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "initial_messages": self.initial_messages,
            "steps": self.steps,
            "tool_output": self.tool_output,
            "final_output": self.final_output,
            "execution": self.execution,
            "reward": self.reward.to_dict(),
            "success": self.semantic_success,
            "semantic_success": self.semantic_success,
            "strict_success": self.strict_success,
            "elapsed_ms": self.elapsed_ms,
        }


class CalculatorEnvironment:
    """A deterministic two-turn environment for calculator tool-use rollouts."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        parser: ToolCallParser | None = None,
        reward: CalculatorReward | None = None,
    ) -> None:
        self.registry = registry
        self.parser = parser or ToolCallParser()
        self.reward = reward or CalculatorReward(self.parser)

    def observation_message(self, tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
        content = (
            "<tool_observation>\n"
            + json.dumps(result, ensure_ascii=False)
            + "\n</tool_observation>"
        )
        return {"role": "tool", "name": tool_name, "content": content}

    def run(
        self,
        task: dict[str, Any],
        generate: Callable[[list[dict[str, Any]]], str],
        *,
        task_id: str | None = None,
    ) -> EpisodeResult:
        started = time.perf_counter()
        expected = task["expected"]
        working_messages = copy.deepcopy(task["messages"])
        steps: list[dict[str, Any]] = []
        tool_output = ""
        final_output: str | None = None
        execution: dict[str, Any] | None = None

        first_output = generate(copy.deepcopy(working_messages))
        if not isinstance(first_output, str):
            first_output = str(first_output)
        tool_output = first_output
        parsed = self.parser.parse(first_output)
        first_step: dict[str, Any] = {
            "step": 0,
            "messages": copy.deepcopy(working_messages),
            "model_output": first_output,
            "parsed_kind": parsed.kind,
            "parse_error": parsed.error,
        }
        if parsed.tool_call is not None:
            first_step["tool_call"] = parsed.tool_call.to_dict()
        steps.append(first_step)

        if parsed.kind == "tool_call" and parsed.tool_call is not None:
            call = parsed.tool_call
            result = self.registry.execute(call.name, call.arguments)
            execution = result.to_dict()
            first_step["tool_result"] = execution
            working_messages.append({"role": "assistant", "content": first_output})
            working_messages.append(self.observation_message(call.name, execution))
            second_output = generate(copy.deepcopy(working_messages))
            if not isinstance(second_output, str):
                second_output = str(second_output)
            final_output = second_output
            second_parsed = self.parser.parse(second_output)
            steps.append(
                {
                    "step": 1,
                    "messages": copy.deepcopy(working_messages),
                    "model_output": second_output,
                    "parsed_kind": second_parsed.kind,
                    "parse_error": second_parsed.error,
                }
            )
        reward = self.reward.score(
            tool_output=tool_output,
            final_output=final_output,
            expected_tool_name=str(expected["tool_name"]),
            expected_arguments=dict(expected["arguments"]),
            expected_final_answer=str(expected["final_answer"]),
            execution=execution,
        )
        semantic_success = bool(
            reward.format_reward == 1.0
            and reward.tool_selection_reward == 1.0
            and reward.argument_reward == 1.0
            and reward.execution_reward == 1.0
            and reward.final_answer_reward == 1.0
        )
        strict_success = bool(
            semantic_success and reward.canonical_format_reward == 1.0
        )
        return EpisodeResult(
            episode_id=uuid.uuid4().hex,
            task_id=str(task_id or task.get("id", "anonymous")),
            initial_messages=copy.deepcopy(task["messages"]),
            steps=steps,
            tool_output=tool_output,
            final_output=final_output,
            execution=execution,
            reward=reward,
            semantic_success=semantic_success,
            strict_success=strict_success,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
