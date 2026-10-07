from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llm_agent_v2.checkpoint import JsonCheckpointStore
from llm_agent_v2.state import AgentState


class CheckpointTests(unittest.TestCase):
    def test_json_checkpoint_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = JsonCheckpointStore(Path(directory))
            state = AgentState(task_id="task/1", goal="checkpoint me")
            checkpoint = store.save(state, "initial")
            loaded = store.load(checkpoint.checkpoint_id)
            self.assertEqual(loaded.state["task_id"], "task/1")
            self.assertEqual(store.list_ids("task/1"), [checkpoint.checkpoint_id])
