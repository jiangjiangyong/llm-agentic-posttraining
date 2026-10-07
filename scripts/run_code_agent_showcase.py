from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llm_posttrain.agent.code_environment import CodeAgentEnvironment


class SequenceGenerator:
    """Deterministic generator used only for the runtime showcase smoke."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.index = 0

    def __call__(self, messages: list[dict[str, Any]]) -> str:
        del messages
        if self.index >= len(self.responses):
            raise RuntimeError("showcase generator exhausted")
        response = self.responses[self.index]
        self.index += 1
        return response


def _tool_call(name: str, arguments: dict[str, Any]) -> str:
    payload = json.dumps(
        {"name": name, "arguments": arguments},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"<tool_call>{payload}</tool_call>"


def build_showcase_task() -> dict[str, Any]:
    return {
        "id": "runtime_showcase_content_ref_v1",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Search the Python executor timeout policy, write the verified fact "
                    "to research/showcase.md, read it back, and report READY. "
                    "Context packet SHOWCASE42: cedar, river, quartz."
                ),
            }
        ],
        "expected": {
            "required_tools": ["mock_search", "write_file", "read_file"],
            "expected_tool_calls": 3,
            "expected_arguments": {
                "mock_search": {"query": "python executor timeout policy"},
                "write_file": {"path": "research/showcase.md"},
                "read_file": {"path": "research/showcase.md"},
            },
            "execution_tools": ["mock_search", "write_file", "read_file"],
            "file_checks": [
                {
                    "path": "research/showcase.md",
                    "contains": ["three second timeout", "Case marker: SHOWCASE42"],
                }
            ],
            "final_answer_contains": ["READY"],
        },
    }


def build_showcase_generator() -> SequenceGenerator:
    return SequenceGenerator(
        [
            _tool_call(
                "mock_search",
                {"query": "python executor timeout policy", "top_k": 1},
            ),
            _tool_call(
                "write_file",
                {
                    "path": "research/showcase.md",
                    "content_ref": "result_snippet",
                    "marker_ref": "context_packet",
                },
            ),
            _tool_call("read_file", {"path": "research/showcase.md"}),
            "READY",
        ]
    )


def run_showcase() -> Any:
    return CodeAgentEnvironment(max_steps=5).run(
        build_showcase_task(),
        build_showcase_generator(),
        task_id="runtime_showcase_content_ref_v1",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the deterministic Code-Agent runtime showcase smoke."
    )
    parser.add_argument(
        "--output",
        default="artifacts/code_agent_showcase_smoke_v1.json",
    )
    parser.add_argument(
        "--report",
        default="docs/code_agent_showcase_smoke_v1.md",
    )
    args = parser.parse_args()

    result = run_showcase()
    steps = result.steps
    resolution = steps[1]["content_ref_resolutions"][0]
    raw_pointer = steps[1]["tool_calls"][0]["arguments"].get("content_ref")
    resolved_content = steps[1]["resolved_tool_calls"][0]["arguments"].get("content")
    assertions = {
        "semantic_success": result.semantic_success,
        "strict_success": result.strict_success,
        "content_ref_applied": resolution.get("applied") is True,
        "raw_pointer_preserved": raw_pointer == "result_snippet",
        "resolved_content_materialized": isinstance(resolved_content, str)
        and "three second timeout" in resolved_content,
        "workspace_snapshot": result.workspace_files == ["research/showcase.md"],
    }
    artifact = {
        "schema": "llm_agentic_code_agent_showcase_smoke_v1",
        "mode": "deterministic_runtime_smoke",
        "model_inference": False,
        "claim_boundary": [
            "This smoke exercises the real CodeAgentEnvironment and resolver.",
            "The fixed generator is not a model capability benchmark.",
            "No holdout target or expected action is read by the resolver.",
        ],
        "assertions": assertions,
        "result": result.to_dict(),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# Code-Agent runtime showcase smoke",
                "",
                "Gate: `PASS_DETERMINISTIC_RUNTIME_SHOWCASE`"
                if all(assertions.values())
                else "Gate: `FAIL_DETERMINISTIC_RUNTIME_SHOWCASE`",
                "",
                "This is a deterministic runtime integration smoke, not a model capability benchmark.",
                "",
                "- Chain: `mock_search → content_ref write_file → read_file → final`",
                f"- Semantic success: `{result.semantic_success}`",
                f"- Strict success: `{result.strict_success}`",
                f"- Resolver applied: `{resolution.get('applied')}`",
                f"- Workspace files: `{result.workspace_files}`",
                f"- Artifact: `{output_path}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "gate": (
                    "PASS_DETERMINISTIC_RUNTIME_SHOWCASE"
                    if all(assertions.values())
                    else "FAIL_DETERMINISTIC_RUNTIME_SHOWCASE"
                ),
                "artifact": str(output_path),
                "report": str(report_path),
                "assertions": assertions,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if all(assertions.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
