import json
from pathlib import Path
import pytest

from llm_posttrain.agent.runtime import AgentRuntime
from llm_posttrain.tools.registry import build_default_registry


def make_runtime() -> AgentRuntime:
    return AgentRuntime(
        backend=lambda _messages: "",
        registry=build_default_registry(),
        max_steps=3,
    )


def test_runtime_instruction_matches_canonical_wording() -> None:
    instruction = make_runtime()._tool_instruction()
    assert "Use the calculator when arithmetic is required." in instruction
    assert "Available tools:\n" in instruction
    assert "When a tool is needed, output exactly one block:\n" in instruction
    assert "Do not add Markdown fences" not in instruction


def test_existing_protocol_system_is_not_duplicated() -> None:
    runtime = make_runtime()
    instruction = runtime._tool_instruction()
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": "Use the calculator to compute 2 + 2."},
    ]
    prepared = runtime._prepare_messages(messages)
    assert prepared == messages
    assert prepared[0]["content"].count("You are running inside Tool Calling Protocol v1.") == 1


def test_canonical_train_system_prompt_is_preserved() -> None:
    path = Path("data/canonical_target_paired/train.jsonl")
    if not path.exists():
        pytest.skip("optional canonical training data is not present")
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    runtime = make_runtime()
    prepared = runtime._prepare_messages(record["messages"])
    assert prepared[0]["content"] == record["messages"][0]["content"]
    assert prepared[0]["content"].count("You are running inside Tool Calling Protocol v1.") == 1


def test_generic_system_gets_one_protocol_instruction_and_is_idempotent() -> None:
    runtime = make_runtime()
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Use the calculator to compute 2 + 2."},
    ]
    prepared = runtime._prepare_messages(messages)
    prepared_again = runtime._prepare_messages(prepared)
    assert prepared_again == prepared
    assert prepared_again[0]["content"].count("You are running inside Tool Calling Protocol v1.") == 1
