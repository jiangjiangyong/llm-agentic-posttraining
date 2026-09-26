from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_posttrain.agent.protocol import ToolCallParser
from llm_posttrain.tools.file_ops import safe_workspace_path
from llm_posttrain.tools.json_validator import validate_json_value


@dataclass(frozen=True)
class CodeAgentRewardBreakdown:
    format_reward: float
    canonical_format_reward: float
    tool_selection_reward: float
    argument_reward: float
    execution_reward: float
    unit_test_reward: float
    final_answer_reward: float
    efficiency_reward: float
    total_reward: float
    parser_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_reward": self.format_reward,
            "canonical_format_reward": self.canonical_format_reward,
            "tool_selection_reward": self.tool_selection_reward,
            "argument_reward": self.argument_reward,
            "execution_reward": self.execution_reward,
            "unit_test_reward": self.unit_test_reward,
            "final_answer_reward": self.final_answer_reward,
            "efficiency_reward": self.efficiency_reward,
            "total_reward": self.total_reward,
            "parser_error": self.parser_error,
        }


_CANONICAL_PATTERN = re.compile(
    r"^\s*<tool_call>\s*(.*?)\s*</tool_call>\s*$",
    re.DOTALL | re.IGNORECASE,
)


def _canonical_tool_output(text: str) -> bool:
    match = _CANONICAL_PATTERN.fullmatch(text)
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
        and isinstance(payload["arguments"], dict)
    )


def _partial_match(actual: dict[str, Any], expected: dict[str, Any]) -> float:
    if not expected:
        return 1.0
    return sum(
        int(actual.get(key) == value)
        for key, value in expected.items()
    ) / len(expected)


def _file_checks_passed(
    workspace: str | Path,
    checks: list[dict[str, Any]],
) -> float:
    if not checks:
        return 1.0
    passed = 0
    for check in checks:
        path = check.get("path")
        if not isinstance(path, str):
            continue
        try:
            target = safe_workspace_path(workspace, path)
        except Exception:
            continue
        if not target.exists() or not target.is_file():
            continue
        content = target.read_text(encoding="utf-8")
        contains = check.get("contains", [])
        if isinstance(contains, list) and all(
            isinstance(value, str) and value in content for value in contains
        ):
            passed += 1
    return passed / len(checks)


def _output_checks_passed(
    result: dict[str, Any],
    expected: dict[str, Any],
) -> bool:
    output = result.get("output")
    if not isinstance(output, dict):
        return False
    return all(output.get(key) == value for key, value in expected.items())


def _parse_json_prefix(text: str) -> Any | None:
    candidate = text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        try:
            value, _ = json.JSONDecoder().raw_decode(candidate)
        except json.JSONDecodeError:
            return None
        return value


class CodeAgentReward:
    """Reward a multi-tool code episode without executing model code here."""

    def __init__(self, parser: ToolCallParser | None = None) -> None:
        self.parser = parser or ToolCallParser()

    def score(
        self,
        *,
        task: dict[str, Any],
        tool_outputs: list[str],
        tool_calls: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
        final_output: str | None,
        workspace: str | Path,
    ) -> CodeAgentRewardBreakdown:
        expected = task.get("expected", {})
        parsed_first = self.parser.parse(tool_outputs[0]) if tool_outputs else None
        format_reward = float(
            parsed_first is not None
            and parsed_first.kind == "tool_call"
            and parsed_first.tool_call is not None
        )
        canonical_reward = (
            sum(float(_canonical_tool_output(text)) for text in tool_outputs)
            / len(tool_outputs)
            if tool_outputs
            else 0.0
        )

        required_tools = expected.get("required_tools", [])
        actual_names = [str(call.get("name")) for call in tool_calls]
        if required_tools:
            tool_selection = sum(
                int(name in actual_names) for name in required_tools
            ) / len(required_tools)
        else:
            tool_selection = float(bool(actual_names))

        argument_scores: list[float] = []
        expected_arguments = expected.get("expected_arguments", {})
        if isinstance(expected_arguments, dict):
            for name, wanted in expected_arguments.items():
                matches = [
                    call.get("arguments", {})
                    for call in tool_calls
                    if call.get("name") == name
                ]
                if matches and isinstance(wanted, dict):
                    argument_scores.append(
                        max(
                            _partial_match(actual, wanted)
                            for actual in matches
                            if isinstance(actual, dict)
                        )
                    )
                else:
                    argument_scores.append(0.0)
        file_checks = expected.get("file_checks", [])
        if isinstance(file_checks, list) and file_checks:
            argument_scores.append(_file_checks_passed(workspace, file_checks))
        argument_reward = (
            sum(argument_scores) / len(argument_scores)
            if argument_scores
            else float(bool(tool_calls))
        )

        required_execution_tools = expected.get("execution_tools", [])
        successful_results = [
            result for result in tool_results if result.get("ok") is True
        ]
        expected_tool_outputs = expected.get("expected_tool_outputs", {})
        if required_execution_tools:
            execution_reward = sum(
                int(
                    any(
                        result.get("tool_name") == name
                        and (
                            not isinstance(expected_tool_outputs, dict)
                            or not isinstance(expected_tool_outputs.get(name), dict)
                            or _output_checks_passed(
                                result,
                                expected_tool_outputs[name],
                            )
                        )
                        for result in successful_results
                    )
                )
                for name in required_execution_tools
            ) / len(required_execution_tools)
        else:
            execution_reward = float(bool(successful_results))

        unit_test_required = bool(expected.get("unit_test_required", False))
        unit_test_results = [
            result.get("output", {})
            for result in tool_results
            if result.get("tool_name") == "run_unit_tests"
        ]
        passed_tests = any(
            isinstance(output, dict) and output.get("passed") is True
            for output in unit_test_results
        )
        unit_test_reward = float(passed_tests) if unit_test_required else 1.0

        final_answer_reward = 0.0
        if final_output is not None:
            expected_final = expected.get("final_answer")
            if isinstance(expected_final, str):
                final_answer_reward = float(
                    final_output.strip() == expected_final.strip()
                )
            contains = expected.get("final_answer_contains", [])
            if isinstance(contains, list) and contains:
                final_answer_reward = float(
                    all(
                        isinstance(value, str) and value in final_output
                        for value in contains
                    )
                )
            schema = expected.get("final_json_schema")
            if isinstance(schema, dict):
                value = _parse_json_prefix(final_output)
                final_answer_reward = float(
                    value is not None
                    and not validate_json_value(value, schema)
                )

        ideal_steps = int(
            expected.get(
                "expected_tool_calls",
                max(1, len(required_tools)),
            )
        )
        excess = max(0, len(tool_calls) - ideal_steps)
        efficiency_reward = max(0.0, 1.0 - 0.1 * excess)

        total = (
            0.10 * format_reward
            + 0.05 * canonical_reward
            + 0.15 * tool_selection
            + 0.10 * argument_reward
            + 0.15 * execution_reward
            + 0.20 * unit_test_reward
            + 0.20 * final_answer_reward
            + 0.05 * efficiency_reward
        )
        return CodeAgentRewardBreakdown(
            format_reward=format_reward,
            canonical_format_reward=canonical_reward,
            tool_selection_reward=tool_selection,
            argument_reward=argument_reward,
            execution_reward=execution_reward,
            unit_test_reward=unit_test_reward,
            final_answer_reward=final_answer_reward,
            efficiency_reward=efficiency_reward,
            total_reward=total,
            parser_error=parsed_first.error if parsed_first else "no tool output",
        )


def code_agent_reward_weights() -> dict[str, float]:
    return {
        "format": 0.10,
        "canonical_format": 0.05,
        "tool_selection": 0.15,
        "arguments": 0.10,
        "execution": 0.15,
        "unit_test": 0.20,
        "final_answer": 0.20,
        "efficiency": 0.05,
    }
