#!/usr/bin/env python3
"""Run a deterministic CPU-only runtime smoke test."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_agent_v2 import (  # noqa: E402
    AgentRuntime,
    JsonCheckpointStore,
    MockModelAdapter,
    ModelResponse,
    RuntimeConfig,
    ToolCall,
    build_cpu_registry,
)


def main() -> int:
    model = MockModelAdapter(
        [
            ModelResponse(
                tool_calls=(ToolCall("workspace_set", {"key": "answer", "value": 2}, "call-1"),),
                usage={"total_tokens": 12},
            ),
            ModelResponse(
                tool_calls=(ToolCall("calculator", {"expression": "2 + 3"}, "call-2"),),
                usage={"total_tokens": 10},
            ),
            ModelResponse(content="The deterministic CPU workflow completed.", usage={"total_tokens": 8}),
        ]
    )
    with tempfile.TemporaryDirectory(prefix="llm-agent-v2-smoke-") as checkpoint_dir:
        result = AgentRuntime(
            model=model,
            tools=build_cpu_registry(),
            checkpoint_store=JsonCheckpointStore(checkpoint_dir),
            config=RuntimeConfig(max_steps=8, token_budget=256),
        ).run(task_id="cpu-smoke", goal="store an answer and calculate 2 + 3")
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
