from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_code_agent_dataset import (
    build_records,
    make_research_to_file_task,
    make_search_task,
)
from llm_posttrain.agent.code_environment import CodeAgentEnvironment


class SequenceGenerator:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.index = 0

    def __call__(self, messages: list[dict[str, object]]) -> str:
        del messages
        response = self.responses[self.index]
        self.index += 1
        return response


def test_rich_dataset_covers_search_to_file_chain() -> None:
    task = make_research_to_file_task(0, "test")
    assistant_turns = [
        str(message["content"])
        for message in task["messages"]
        if message.get("role") == "assistant"
    ]
    result = CodeAgentEnvironment(max_steps=5).run(
        task,
        SequenceGenerator(assistant_turns),
    )

    assert result.semantic_success is True
    assert result.strict_success is True
    assert [call["name"] for call in result.tool_calls] == [
        "mock_search",
        "write_file",
        "read_file",
    ]
    assert result.workspace_files == ["finding.md"]


def test_rich_dataset_has_six_categories_and_search_variants() -> None:
    records = build_records(12, "test", 20260915)
    assert len(records) == 12
    assert {
        record["category"] for record in records
    } == {
        "code_unit_test",
        "python_execution",
        "structured_json",
        "workspace_file",
        "mock_search",
        "research_to_file",
    }

    queries = {
        make_search_task(index, "test")["expected"]["expected_arguments"][
            "mock_search"
        ]["query"]
        for index in range(5)
    }
    assert len(queries) == 5
