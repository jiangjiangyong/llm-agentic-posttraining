from __future__ import annotations

import tempfile
import unittest

from llm_agent_v2.checkpoint import JsonCheckpointStore
from llm_agent_v2.failure import FailureType
from llm_agent_v2.models import (
    GpuRequiredError,
    HuggingFaceModelAdapter,
    MockModelAdapter,
    ModelResponse,
    ToolCall,
)
from llm_agent_v2.runtime import AgentRuntime, RuntimeConfig
from llm_agent_v2.tools import build_cpu_registry


class RuntimeTests(unittest.TestCase):
    def test_cpu_runtime_records_transition_and_checkpoint(self) -> None:
        model = MockModelAdapter(
            [
                ModelResponse(
                    tool_calls=(ToolCall("workspace_set", {"key": "x", "value": 5}, "c1"),),
                    usage={"total_tokens": 5},
                ),
                ModelResponse(content="done", usage={"total_tokens": 3}),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            result = AgentRuntime(
                model,
                build_cpu_registry(),
                JsonCheckpointStore(directory),
                RuntimeConfig(max_steps=4, token_budget=100),
            ).run("task-1", "set x")
        self.assertEqual(result.status, "success")
        self.assertEqual(result.state.workspace_state["x"], 5)
        self.assertEqual(len(result.transitions), 1)
        self.assertEqual(len(result.checkpoint_ids), 1)
        self.assertTrue(result.metrics()["task_success"])

    def test_repeated_action_is_classified_and_breaks_loop(self) -> None:
        repeated_call = ToolCall("calculator", {"expression": "1 + 1"}, "same")
        model = MockModelAdapter(
            [
                ModelResponse(tool_calls=(repeated_call,), usage={"total_tokens": 1}),
                ModelResponse(tool_calls=(repeated_call,), usage={"total_tokens": 1}),
            ]
        )
        result = AgentRuntime(
            model,
            build_cpu_registry(),
            config=RuntimeConfig(max_steps=4, token_budget=100),
        ).run("loop-task", "do not repeat")
        self.assertEqual(result.status, "abstained")
        self.assertIn(FailureType.LOOP_REPEATED_ACTION, [item.failure_type for item in result.failures])
        self.assertEqual(result.recoveries[-1].recovery_action, "break_loop")

    def test_deferred_gpu_adapter_is_explicit(self) -> None:
        adapter = HuggingFaceModelAdapter("future-model")
        with self.assertRaises(GpuRequiredError):
            adapter.generate(None)  # type: ignore[arg-type]
        self.assertEqual(adapter.get_model_info()["status"], "deferred_until_gpu")
