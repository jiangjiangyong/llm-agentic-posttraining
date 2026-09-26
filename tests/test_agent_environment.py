from typing import Any

from llm_posttrain.agent.environment import CalculatorEnvironment
from llm_posttrain.tools.registry import build_default_registry


class SequenceGenerator:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.seen_messages: list[list[dict[str, Any]]] = []

    def __call__(self, messages: list[dict[str, Any]]) -> str:
        self.seen_messages.append(messages)
        return self.responses[len(self.seen_messages) - 1]


def make_task() -> dict[str, Any]:
    return {
        "id": "environment_test",
        "messages": [{"role": "user", "content": "calculate 2 + 2"}],
        "expected": {
            "tool_name": "calculator",
            "arguments": {"expression": "2 + 2"},
            "final_answer": "4",
        },
    }


def test_environment_executes_two_turn_tool_trajectory() -> None:
    generator = SequenceGenerator(
        [
            (
                '<tool_call>{"name":"calculator",'
                '"arguments":{"expression":"2 + 2"}}</tool_call>'
            ),
            "4",
        ]
    )
    result = CalculatorEnvironment(build_default_registry()).run(
        make_task(), generator, task_id="environment_test"
    )

    assert result.success is True
    assert len(result.steps) == 2
    assert result.execution is not None
    assert result.execution["ok"] is True
    assert result.reward.total_reward == 1.0
    assert generator.seen_messages[1][-1]["role"] == "tool"


def test_environment_records_invalid_first_turn_without_execution() -> None:
    generator = SequenceGenerator(["<tool_call>{bad json</tool_call>"])
    result = CalculatorEnvironment(build_default_registry()).run(
        make_task(), generator
    )

    assert result.success is False
    assert len(result.steps) == 1
    assert result.execution is None
    assert result.final_output is None
    assert result.reward.total_reward == 0.0
    assert result.steps[0]["parsed_kind"] == "invalid"

def test_environment_marks_json_compatibility_as_semantic_success() -> None:
    generator = SequenceGenerator(
        [
            (
                '{"type":"tool_call","name":"calculator",'
                '"arguments":{"expression":"2 + 2"}}'
            ),
            "4",
        ]
    )
    result = CalculatorEnvironment(build_default_registry()).run(
        make_task(), generator, task_id="json_compatibility"
    )

    assert result.success is True
    assert result.semantic_success is True
    assert result.strict_success is False
    assert abs(result.reward.total_reward - 0.9) < 1e-9
