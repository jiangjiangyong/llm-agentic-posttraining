import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_failure_flywheel import (  # noqa: E402
    build_dpo_records,
    build_sft_records,
    failure_types,
    split_grouped,
    validate_dpo_record,
    validate_sft_record,
)
from llm_posttrain.agent.protocol import ToolCallParser
from llm_posttrain.tools.registry import build_default_registry


def make_row(task_id: str, *, final_failure: bool = False) -> dict:
    return {
        "id": f"{task_id}:group_0:rollout_0",
        "episode_id": f"episode_{task_id}",
        "task_id": f"{task_id}:group_0:rollout_0",
        "initial_messages": [{"role": "user", "content": f"calculate {task_id}"}],
        "expected": {
            "tool_name": "calculator",
            "arguments": {"expression": "2 + 2"},
            "final_answer": "4",
        },
        "tool_output": '<tool_call>{"name":"calculator","arguments":{"expression":"2 + 2"}}</tool_call>',
        "final_output": "wrong" if final_failure else "4",
        "reward": {
            "format_reward": 1.0,
            "canonical_format_reward": 1.0,
            "tool_selection_reward": 1.0,
            "argument_reward": 1.0,
            "execution_reward": 1.0,
            "final_answer_reward": 0.0 if final_failure else 1.0,
            "total_reward": 0.8 if final_failure else 1.0,
        },
    }


def test_failure_types_identify_final_answer_failure() -> None:
    assert failure_types(make_row("task_a", final_failure=True)) == [
        "final_answer_failure"
    ]


def test_builders_create_valid_sft_and_dpo_records() -> None:
    rows = {"task_a": [make_row("task_a", final_failure=True)], "task_b": [make_row("task_b")]}
    registry = build_default_registry()
    parser = ToolCallParser()
    sft_records = build_sft_records(rows, registry)
    dpo_records = build_dpo_records(rows, registry)

    assert len(sft_records) == 2
    assert len(dpo_records) == 1
    validate_sft_record(sft_records[0], registry, parser)
    validate_dpo_record(dpo_records[0], parser)


def test_split_grouped_keeps_task_groups_together() -> None:
    records = [
        {"id": "a1", "metadata": {"task_id": "a"}},
        {"id": "a2", "metadata": {"task_id": "a"}},
        {"id": "b1", "metadata": {"task_id": "b"}},
        {"id": "c1", "metadata": {"task_id": "c"}},
    ]
    train, valid = split_grouped(records, 0.25, 7)
    train_tasks = {item["metadata"]["task_id"] for item in train}
    valid_tasks = {item["metadata"]["task_id"] for item in valid}

    assert train_tasks.isdisjoint(valid_tasks)
    assert {item["id"] for item in train + valid} == {"a1", "a2", "b1", "c1"}
