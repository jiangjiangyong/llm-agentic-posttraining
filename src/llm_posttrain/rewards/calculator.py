from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from llm_posttrain.agent.protocol import ToolCallParser


@dataclass(frozen=True)
class RewardBreakdown:
    format_reward: float
    canonical_format_reward: float
    tool_selection_reward: float
    argument_reward: float
    execution_reward: float
    final_answer_reward: float
    total_reward: float
    parser_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_reward": self.format_reward,
            "canonical_format_reward": self.canonical_format_reward,
            "tool_selection_reward": self.tool_selection_reward,
            "argument_reward": self.argument_reward,
            "execution_reward": self.execution_reward,
            "final_answer_reward": self.final_answer_reward,
            "total_reward": self.total_reward,
            "parser_error": self.parser_error,
        }


class CalculatorReward:
    """Score a two-turn calculator episode without executing model code."""

    _canonical_pattern = re.compile(
        r"^\s*<tool_call>\s*(.*?)\s*</tool_call>\s*$",
        re.DOTALL | re.IGNORECASE,
    )

    def __init__(self, parser: ToolCallParser | None = None) -> None:
        self.parser = parser or ToolCallParser()

    def score(
        self,
        *,
        tool_output: str,
        final_output: str | None,
        expected_tool_name: str,
        expected_arguments: dict[str, Any],
        expected_final_answer: str,
        execution: dict[str, Any] | None,
    ) -> RewardBreakdown:
        parsed = self.parser.parse(tool_output)
        format_reward = float(parsed.kind == "tool_call" and parsed.tool_call is not None)
        canonical_reward = 0.0
        canonical_match = self._canonical_pattern.fullmatch(tool_output)
        if canonical_match is not None and format_reward > 0:
            try:
                payload = json.loads(canonical_match.group(1).strip())
            except json.JSONDecodeError:
                payload = None
            canonical_reward = float(
                isinstance(payload, dict)
                and set(payload) == {"name", "arguments"}
                and isinstance(payload["name"], str)
                and isinstance(payload["arguments"], dict)
            )
        tool_selection = 0.0
        argument = 0.0
        if parsed.tool_call is not None:
            tool_selection = float(parsed.tool_call.name == expected_tool_name)
            argument = float(parsed.tool_call.arguments == expected_arguments)
        execution_reward = float(bool(execution and execution.get("ok") is True))
        final_reward = float(
            final_output is not None
            and str(final_output).strip() == str(expected_final_answer).strip()
        )
        total = (
            0.10 * format_reward
            + 0.10 * canonical_reward
            + 0.15 * tool_selection
            + 0.25 * argument
            + 0.20 * execution_reward
            + 0.20 * final_reward
        )
        return RewardBreakdown(
            format_reward=format_reward,
            canonical_format_reward=canonical_reward,
            tool_selection_reward=tool_selection,
            argument_reward=argument,
            execution_reward=execution_reward,
            final_answer_reward=final_reward,
            total_reward=total,
            parser_error=parsed.error,
        )


def reward_weights() -> dict[str, float]:
    return {
        "format_reward": 0.10,
        "canonical_format_reward": 0.10,
        "tool_selection_reward": 0.15,
        "argument_reward": 0.25,
        "execution_reward": 0.20,
        "final_answer_reward": 0.20,
    }
