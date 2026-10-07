from __future__ import annotations

import unittest

from llm_agent_v2.state import AgentState, StateTransition, compute_state_diff


class StateTests(unittest.TestCase):
    def test_tool_observation_updates_mutable_state_and_diff(self) -> None:
        before = AgentState(task_id="t1", goal="store value")
        after = before.clone()
        after.apply_tool_observation(
            "workspace_set:{\"key\":\"x\"}",
            True,
            "ok",
            state_patch={"workspace_state": {"x": 7}},
        )
        transition = StateTransition.from_states(
            before,
            {"type": "tool_call", "name": "workspace_set"},
            {"success": True},
            after,
        )
        self.assertEqual(after.workspace_state["x"], 7)
        self.assertEqual(transition.state_diff["workspace_state"]["after"]["x"], 7)
        self.assertEqual(compute_state_diff(before, after)["steps_used"]["after"], 1)

    def test_immutable_fields_cannot_be_patched(self) -> None:
        state = AgentState(task_id="t1", goal="goal")
        with self.assertRaises(ValueError):
            state.apply_tool_observation("bad", True, "", {"goal": "changed"})
