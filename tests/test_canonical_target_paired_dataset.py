from __future__ import annotations

import pytest

from scripts.build_canonical_target_paired_dataset import pair_record


def test_pair_changes_only_tool_serialization() -> None:
    source = {
        "id": "pair-1",
        "messages": [
            {"role": "user", "content": "Use calculator on 2 + 3."},
            {
                "role": "assistant",
                "content": '<tool_call>{"type":"tool_call","name":"calculator","arguments":{"expression":"2 + 3"}}</tool_call>',
            },
            {"role": "tool", "name": "calculator", "content": "5"},
            {"role": "assistant", "content": "5"},
        ],
    }
    paired, audit = pair_record(source, split="train")
    assistant_messages = [
        message["content"]
        for message in paired["messages"]
        if message["role"] == "assistant"
    ]
    assert assistant_messages[0] == '<tool_call>{"name":"calculator","arguments":{"expression":"2 + 3"}}</tool_call>'
    assert assistant_messages[1] == "5"
    assert paired["metadata"]["paired_tool_call_count"] == 1
    assert paired["metadata"]["paired_targets"][0]["tool_name"] == "calculator"
    assert audit["serialization_changed"] == 1


def test_non_tool_record_is_unchanged() -> None:
    source = {
        "id": "pair-2",
        "messages": [
            {"role": "user", "content": "Return a JSON object."},
            {"role": "assistant", "content": '{"ok":true}'},
        ],
    }
    paired, audit = pair_record(source, split="train")
    assert paired["messages"] == source["messages"]
    assert audit["tool_call_pairs"] == 0
    assert paired["metadata"]["serialization_changed_count"] == 0


def test_invalid_tool_target_is_rejected() -> None:
    source = {
        "id": "pair-3",
        "messages": [
            {"role": "user", "content": "Use a tool."},
            {"role": "assistant", "content": "<tool_call>{bad}</tool_call>"},
        ],
    }
    with pytest.raises(ValueError, match="not parseable"):
        pair_record(source, split="train")
