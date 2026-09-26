from __future__ import annotations

import copy
import json
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from llm_posttrain.agent.protocol import ToolCallParser
from llm_posttrain.agent.runtime import AgentRuntime
from llm_posttrain.agent.trace import AgentTrace
from llm_posttrain.rewards.code_agent import (
    CodeAgentReward,
    CodeAgentRewardBreakdown,
)
from llm_posttrain.tools.code_registry import build_code_agent_registry


@dataclass(frozen=True)
class CodeAgentEpisodeResult:
    episode_id: str
    task_id: str
    initial_messages: list[dict[str, Any]]
    steps: list[dict[str, Any]]
    tool_outputs: list[str]
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    final_output: str | None
    reward: CodeAgentRewardBreakdown
    semantic_success: bool
    strict_success: bool
    runtime_success: bool
    elapsed_ms: float
    workspace_files: list[str]

    @property
    def success(self) -> bool:
        return self.semantic_success

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "initial_messages": self.initial_messages,
            "steps": self.steps,
            "tool_outputs": self.tool_outputs,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "final_output": self.final_output,
            "reward": self.reward.to_dict(),
            "success": self.semantic_success,
            "semantic_success": self.semantic_success,
            "strict_success": self.strict_success,
            "runtime_success": self.runtime_success,
            "elapsed_ms": self.elapsed_ms,
            "workspace_files": self.workspace_files,
        }


class CodeAgentEnvironment:
    """Multi-tool, deterministic code-agent environment with an isolated workspace."""

    def __init__(
        self,
        *,
        parser: ToolCallParser | None = None,
        reward: CodeAgentReward | None = None,
        max_steps: int = 6,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.parser = parser or ToolCallParser()
        self.reward = reward or CodeAgentReward(self.parser)
        self.max_steps = max_steps

    @staticmethod
    def _snapshot_workspace(workspace: Path) -> list[str]:
        return sorted(
            path.relative_to(workspace).as_posix()
            for path in workspace.rglob("*")
            if path.is_file() and path.name != "__agent_main__.py"
        )

    def run(
        self,
        task: dict[str, Any],
        generate: Callable[[list[dict[str, Any]]], str],
        *,
        task_id: str | None = None,
    ) -> CodeAgentEpisodeResult:
        started = time.perf_counter()
        task_name = str(task_id or task.get("id", "anonymous"))
        initial_messages = task.get("prompt_messages", task.get("messages"))
        if not isinstance(initial_messages, list) or not initial_messages:
            raise ValueError("task must contain non-empty prompt_messages")
        with tempfile.TemporaryDirectory(prefix="llm_code_episode_") as temp_dir:
            registry, workspace = build_code_agent_registry(temp_dir)
            runtime = AgentRuntime(
                backend=generate,
                registry=registry,
                parser=self.parser,
                max_steps=self.max_steps,
            )
            has_code_protocol = any(
                message.get("role") == "system"
                and "CodeToolAgent Protocol v2" in str(message.get("content", ""))
                for message in initial_messages
            )
            prepared = (
                copy.deepcopy(initial_messages)
                if has_code_protocol
                else runtime._prepare_messages(initial_messages)
            )
            working_messages = copy.deepcopy(prepared)
            trace = AgentTrace(
                run_id=uuid.uuid4().hex,
                task_id=task_name,
                initial_messages=copy.deepcopy(initial_messages),
                prompt_messages=copy.deepcopy(prepared),
            )
            steps: list[dict[str, Any]] = []
            tool_outputs: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            tool_results: list[dict[str, Any]] = []
            final_output: str | None = None
            runtime_success = False

            for step in range(self.max_steps):
                try:
                    raw_output = generate(copy.deepcopy(working_messages))
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    trace.error = error
                    trace.add_event(
                        step=step,
                        kind="runtime_error",
                        payload={"error": error},
                    )
                    steps.append(
                        {
                            "step": step,
                            "messages": copy.deepcopy(working_messages),
                            "model_output": None,
                            "parsed_kind": "invalid",
                            "parse_error": error,
                        }
                    )
                    break

                if not isinstance(raw_output, str):
                    raw_output = str(raw_output)
                parsed = self.parser.parse(raw_output)
                row: dict[str, Any] = {
                    "step": step,
                    "messages": copy.deepcopy(working_messages),
                    "model_output": raw_output,
                    "parsed_kind": parsed.kind,
                    "parse_error": parsed.error,
                }
                trace.add_event(
                    step=step,
                    kind="model_output",
                    payload={"text": raw_output},
                )

                if parsed.kind == "tool_call" and parsed.tool_call is not None:
                    recovered_calls = self.parser.parse_many(raw_output)
                    calls: list[Any] = []
                    seen_calls: set[str] = set()
                    for candidate in recovered_calls:
                        try:
                            registry.get(candidate.name)
                        except Exception:
                            # Keep unknown tools visible in the original model
                            # output, but do not execute hallucinated suffixes.
                            continue
                        signature = json.dumps(
                            candidate.to_dict(),
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        if signature not in seen_calls:
                            seen_calls.add(signature)
                            calls.append(candidate)
                    if not calls:
                        calls = [parsed.tool_call]
                    call_dicts = [call.to_dict() for call in calls]
                    results: list[dict[str, Any]] = []
                    row["tool_call"] = call_dicts[0]
                    row["tool_calls"] = call_dicts
                    steps.append(row)
                    tool_outputs.append(raw_output)
                    working_messages.append(
                        {"role": "assistant", "content": raw_output}
                    )
                    for call in calls:
                        call_dict = call.to_dict()
                        result = registry.execute(
                            call.name,
                            call.arguments,
                        ).to_dict()
                        results.append(result)
                        tool_calls.append(call_dict)
                        tool_results.append(result)
                        trace.add_event(
                            step=step,
                            kind="tool_call",
                            payload=call_dict,
                        )
                        trace.add_event(
                            step=step,
                            kind="tool_result",
                            payload=result,
                        )
                        working_messages.append(
                            runtime._observation_message(call.name, result)
                        )
                    row["tool_result"] = results[0]
                    row["tool_results"] = results
                    continue

                steps.append(row)
                if parsed.kind == "final":
                    final_output = parsed.text
                    runtime_success = True
                    trace.final_answer = parsed.text
                    trace.add_event(
                        step=step,
                        kind="final_answer",
                        payload={"text": parsed.text},
                    )
                else:
                    error = parsed.error or "invalid model output"
                    trace.error = error
                    trace.add_event(
                        step=step,
                        kind="parse_error",
                        payload={"error": error},
                    )
                break

            reward = self.reward.score(
                task=task,
                tool_outputs=tool_outputs,
                tool_calls=tool_calls,
                tool_results=tool_results,
                final_output=final_output,
                workspace=workspace,
            )
            semantic_success = bool(
                reward.format_reward == 1.0
                and reward.tool_selection_reward == 1.0
                and reward.argument_reward == 1.0
                and reward.execution_reward == 1.0
                and reward.unit_test_reward == 1.0
                and reward.final_answer_reward == 1.0
            )
            strict_success = bool(
                semantic_success and reward.canonical_format_reward == 1.0
            )
            trace.success = semantic_success
            workspace_files = self._snapshot_workspace(workspace)

        return CodeAgentEpisodeResult(
            episode_id=uuid.uuid4().hex,
            task_id=task_name,
            initial_messages=copy.deepcopy(initial_messages),
            steps=steps,
            tool_outputs=tool_outputs,
            tool_calls=tool_calls,
            tool_results=tool_results,
            final_output=final_output,
            reward=reward,
            semantic_success=semantic_success,
            strict_success=strict_success,
            runtime_success=runtime_success,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            workspace_files=workspace_files,
        )
