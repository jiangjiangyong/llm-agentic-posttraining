import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_canonical_sft_dataset import (
    canonicalize_record,
    tool_system_message,
    validate_records,
)
from llm_posttrain.agent.protocol import ToolCallParser
from llm_posttrain.tools.registry import build_default_registry


def test_legacy_tool_call_is_rewritten_to_canonical_protocol() -> None:
    registry = build_default_registry()
    execution = registry.execute(
        "calculator",
        {"expression": "125 * 36"},
    )
    observation = (
        "<tool_observation>\n"
        + json.dumps(execution.to_dict(), ensure_ascii=False)
        + "\n</tool_observation>"
    )
    record = {
        "id": "legacy_tool",
        "category": "tool_calling",
        "messages": [
            {"role": "system", "content": "Use the calculator."},
            {"role": "user", "content": "Calculate 125 * 36."},
            {
                "role": "assistant",
                "content": (
                    '<tool_call>{"type":"tool_call","name":"calculator",'
                    '"arguments":{"expression":"125 * 36"}}</tool_call>'
                ),
            },
            {"role": "tool", "name": "calculator", "content": observation},
            {"role": "assistant", "content": "4500"},
        ],
        "metadata": {
            "expression": "125 * 36",
            "answer": "4500",
        },
    }

    rewritten = canonicalize_record(
        record,
        ToolCallParser(),
        registry,
    )
    assistant_call = rewritten["messages"][2]["content"]
    assert rewritten["messages"][0]["content"] == tool_system_message(registry)
    assert assistant_call == (
        '<tool_call>{"name":"calculator","arguments":'
        '{"expression":"125 * 36"}}</tool_call>'
    )
    assert '"type"' not in assistant_call
    report = validate_records([rewritten], [])
    assert report["canonical_tool_call_records"] == 1
