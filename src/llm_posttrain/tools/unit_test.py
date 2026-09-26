from __future__ import annotations

from pathlib import Path
from typing import Any

from .code_executor import (
    CodeSafetyError,
    ExecutionPolicy,
    execute_python,
    validate_python_source,
)
from .file_ops import safe_workspace_path
from .schema import ToolSpec


def run_unit_tests(
    arguments: dict[str, Any],
    *,
    workspace: str | Path,
    policy: ExecutionPolicy | None = None,
) -> dict[str, Any]:
    path = arguments.get("path")
    tests = arguments.get("tests")
    if not isinstance(path, str) or not path.strip():
        raise CodeSafetyError("path must be a non-empty string")
    if isinstance(tests, list) and all(isinstance(item, str) for item in tests):
        tests = "\n".join(tests)
    if not isinstance(tests, str) or not tests.strip():
        raise CodeSafetyError("tests must be a non-empty string")
    source_path = safe_workspace_path(workspace, path)
    if not source_path.exists() or not source_path.is_file():
        return {
            "passed": False,
            "status": "missing_source",
            "path": path,
            "stdout": "",
            "stderr": f"source file does not exist: {path}",
            "return_code": None,
            "timed_out": False,
        }
    source = source_path.read_text(encoding="utf-8")
    validate_python_source(source, policy=policy)
    validate_python_source(tests, policy=policy)
    combined = (
        source.rstrip()
        + "\n\n# --- agent unit tests ---\n"
        + tests.rstrip()
        + "\n"
    )
    result = execute_python(
        combined,
        workdir=workspace,
        policy=policy,
    )
    return {
        **result,
        "passed": bool(result["ok"]),
        "path": path,
        "test_count_hint": tests.count("assert "),
    }


def make_unit_test_tool(workspace: str | Path) -> ToolSpec:
    def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        return run_unit_tests(arguments, workspace=workspace)

    return ToolSpec(
        name="run_unit_tests",
        description=(
            "Run assertion-based unit tests against a Python file in the "
            "episode workspace."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative Python source path.",
                },
                "tests": {
                    "type": "string",
                    "description": "Python assertions that call the source functions.",
                },
            },
            "required": ["path", "tests"],
            "additionalProperties": False,
        },
        handler=handler,
    )
