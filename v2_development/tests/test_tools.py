from __future__ import annotations

import unittest

from llm_agent_v2.tools import build_cpu_registry


class ToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_cpu_registry()

    def test_calculator_is_deterministic_and_rejects_calls(self) -> None:
        result = self.registry.invoke("calculator", {"expression": "2 * (3 + 4)"})
        self.assertTrue(result.success)
        self.assertEqual(result.output, "14")

        rejected = self.registry.invoke("calculator", {"expression": "__import__('os')"})
        self.assertFalse(rejected.success)
        self.assertEqual(rejected.error_type, "tool_execution")

    def test_workspace_tools_use_explicit_state_patch(self) -> None:
        result = self.registry.invoke("workspace_set", {"key": "x", "value": 3})
        self.assertTrue(result.success)
        self.assertEqual(result.state_patch, {"workspace_state": {"x": 3}})
        observed = self.registry.invoke(
            "workspace_get", {"key": "x"}, {"workspace_state": {"x": 3}}
        )
        self.assertTrue(observed.success)
        self.assertEqual(observed.output, "3")

    def test_invalid_and_unknown_calls_are_typed(self) -> None:
        invalid = self.registry.invoke("calculator", {})
        self.assertEqual(invalid.error_type, "invalid_argument")
        unknown = self.registry.invoke("missing", {})
        self.assertEqual(unknown.error_type, "unknown_tool")
