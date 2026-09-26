from __future__ import annotations

import json
from typing import Any

from llm_posttrain.agent.code_environment import CodeAgentEnvironment
from llm_posttrain.agent.protocol import ToolCallParser


class SequenceGenerator:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.index = 0

    def __call__(self, messages: list[dict[str, Any]]) -> str:
        del messages
        response = self.responses[self.index]
        self.index += 1
        return response


def test_parser_accepts_observed_legacy_tool_alias_and_pipe_prefix() -> None:
    parsed = ToolCallParser().parse(
        '|{"tool":"python_executor","arguments":{"code":"print(1)"}}'
    )
    assert parsed.kind == "tool_call"
    assert parsed.tool_call is not None
    assert parsed.tool_call.name == "python_executor"
    assert parsed.tool_call.arguments == {"code": "print(1)"}


def test_parser_recovers_tool_json_after_short_generation_marker() -> None:
    parsed = ToolCallParser().parse(
        'CONFIG READY. {"name":"read_file","arguments":{"path":"config.json"}}'
    )
    assert parsed.kind == "tool_call"
    assert parsed.tool_call is not None
    assert parsed.tool_call.name == "read_file"


def test_code_environment_scores_write_and_unit_test_trajectory() -> None:
    code = "def add(left, right):\n    return left + right\n"
    tests = "assert add(2, 3) == 5\nassert add(-1, 1) == 0"
    generator = SequenceGenerator(
        [
            (
                '<tool_call>{"name":"write_file","arguments":'
                '{"path":"solution.py","content":'
                + repr(code).replace("'", '"')
                + "}}</tool_call>"
            ),
            (
                '<tool_call>{"name":"run_unit_tests","arguments":'
                '{"path":"solution.py","tests":'
                + repr(tests).replace("'", '"')
                + "}}</tool_call>"
            ),
            "PASS: add",
        ]
    )
    task = {
        "id": "code_environment_test",
        "messages": [{"role": "user", "content": "implement add"}],
        "expected": {
            "required_tools": ["write_file", "run_unit_tests"],
            "expected_tool_calls": 2,
            "expected_arguments": {
                "write_file": {"path": "solution.py"},
                "run_unit_tests": {"path": "solution.py"},
            },
            "execution_tools": ["write_file", "run_unit_tests"],
            "unit_test_required": True,
            "file_checks": [{"path": "solution.py", "contains": ["def add"]}],
            "final_answer": "PASS: add",
        },
    }
    result = CodeAgentEnvironment(max_steps=5).run(task, generator)

    assert result.semantic_success is True
    assert result.strict_success is True
    assert result.reward.unit_test_reward == 1.0
    assert result.reward.execution_reward == 1.0
    assert len(result.tool_calls) == 2
    assert result.workspace_files == ["solution.py"]


def test_code_environment_recovers_multiple_legacy_tool_payloads() -> None:
    write_call = json.dumps(
        {
            "name": "write_file",
            "arguments": {"path": "config.json", "content": "{\"ok\": true}"},
        }
    )
    read_call = json.dumps(
        {"name": "read_file", "arguments": {"path": "config.json"}}
    )
    generator = SequenceGenerator([write_call + " noise " + read_call, "CONFIG READY"])
    task = {
        "id": "multi_action_recovery_test",
        "messages": [{"role": "user", "content": "write and read config"}],
        "expected": {
            "required_tools": ["write_file", "read_file"],
            "expected_tool_calls": 2,
            "expected_arguments": {
                "write_file": {"path": "config.json"},
                "read_file": {"path": "config.json"},
            },
            "execution_tools": ["write_file", "read_file"],
            "file_checks": [{"path": "config.json", "contains": ["ok"]}],
            "final_answer_contains": ["CONFIG READY"],
        },
    }
    result = CodeAgentEnvironment(max_steps=3).run(task, generator)

    assert result.semantic_success is True
    assert result.strict_success is False
    assert [call["name"] for call in result.tool_calls] == [
        "write_file",
        "read_file",
    ]


def test_code_environment_drops_unknown_suffix_actions() -> None:
    executor_call = json.dumps(
        {
            "name": "python_executor",
            "arguments": {"code": "print(145)"},
        }
    )
    unknown_suffix = json.dumps({"name": "done", "arguments": {}})
    generator = SequenceGenerator([executor_call + " " + unknown_suffix, "RESULT: 145"])
    task = {
        "id": "unknown_suffix_test",
        "messages": [{"role": "user", "content": "run the program"}],
        "expected": {
            "required_tools": ["python_executor"],
            "expected_tool_calls": 1,
            "execution_tools": ["python_executor"],
            "expected_tool_outputs": {"python_executor": {"result_text": "145"}},
            "final_answer": "RESULT: 145",
        },
    }
    result = CodeAgentEnvironment(max_steps=3).run(task, generator)

    assert result.semantic_success is True
    assert [call["name"] for call in result.tool_calls] == ["python_executor"]
