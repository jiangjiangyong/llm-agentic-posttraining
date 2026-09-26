from __future__ import annotations

import tempfile
from pathlib import Path

from .calculator import calculator_tool_spec
from .code_executor import python_executor_tool_spec
from .file_ops import make_file_tools
from .json_validator import json_validator_tool_spec
from .mock_search import mock_search_tool_spec
from .registry import ToolRegistry
from .unit_test import make_unit_test_tool


CODE_AGENT_TOOL_NAMES = (
    "calculator",
    "python_executor",
    "validate_json",
    "mock_search",
    "write_file",
    "read_file",
    "list_files",
    "run_unit_tests",
)


def build_code_agent_registry(
    workspace: str | Path | None = None,
) -> tuple[ToolRegistry, Path]:
    if workspace is None:
        workspace_path = Path(tempfile.mkdtemp(prefix="llm_agent_workspace_"))
    else:
        workspace_path = Path(workspace).resolve()
        workspace_path.mkdir(parents=True, exist_ok=True)
    specs = [
        calculator_tool_spec(),
        python_executor_tool_spec(),
        json_validator_tool_spec(),
        mock_search_tool_spec(),
        *make_file_tools(workspace_path),
        make_unit_test_tool(workspace_path),
    ]
    return ToolRegistry(specs), workspace_path

