from __future__ import annotations

import tempfile
from pathlib import Path

from llm_posttrain.tools.code_registry import build_code_agent_registry
from llm_posttrain.tools.code_executor import python_executor
from llm_posttrain.tools.json_validator import validate_json
from llm_posttrain.tools.mock_search import mock_search


def test_python_executor_runs_and_blocks_dangerous_names() -> None:
    result = python_executor({"code": "print(2 + 3)"})
    assert result["ok"] is True
    assert result["result_text"] == "5"

    blocked = python_executor({"code": "print(open('secret.txt').read())"})
    assert blocked["ok"] is False
    assert blocked["status"] == "policy_error"


def test_code_registry_supports_files_and_unit_tests() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        registry, workspace = build_code_agent_registry(temp_dir)
        written = registry.execute(
            "write_file",
            {
                "path": "solution.py",
                "content": "def add(left, right):\n    return left + right\n",
            },
        )
        assert written.ok is True
        tested = registry.execute(
            "run_unit_tests",
            {
                "path": "solution.py",
                "tests": "assert add(2, 3) == 5\nassert add(-1, 1) == 0",
            },
        )
        assert tested.ok is True
        assert tested.output["passed"] is True
        assert (Path(workspace) / "solution.py").exists()


def test_unit_test_tool_accepts_string_list_compatibility() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        registry, _ = build_code_agent_registry(temp_dir)
        written = registry.execute(
            "write_file",
            {
                "path": "solution.py",
                "content": "def add(left, right):\n    return left + right\n",
            },
        )
        assert written.ok is True
        tested = registry.execute(
            "run_unit_tests",
            {
                "path": "solution.py",
                "tests": [
                    "assert add(2, 3) == 5",
                    "assert add(-1, 1) == 0",
                ],
            },
        )
        assert tested.ok is True
        assert tested.output["passed"] is True


def test_json_validator_and_mock_search_are_deterministic() -> None:
    valid = validate_json(
        {
            "json_text": '{"priority":"high"}',
            "schema": {
                "type": "object",
                "properties": {"priority": {"type": "string"}},
                "required": ["priority"],
                "additionalProperties": False,
            },
        }
    )
    assert valid["valid"] is True

    results = mock_search({"query": "unit test reward", "top_k": 2})
    assert results["results"]
    assert results["results"][0]["id"] == "doc-reward-unit-test"
