import json
import sys
import pytest
from pathlib import Path

from llm_posttrain.agent.protocol import ToolCallParser

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_protocol_alignment_records_are_canonical(tmp_path) -> None:
    from build_protocol_alignment_dataset import make_record, validate_records
    from llm_posttrain.tools.registry import build_default_registry

    record = make_record(0, build_default_registry())
    call = record["messages"][2]["content"]
    payload = json.loads(call[len("<tool_call>") : -len("</tool_call>")])
    assert payload == {
        "name": "calculator",
        "arguments": {"expression": "700 + 4"},
    }
    parsed = ToolCallParser().parse(call)
    assert parsed.kind == "tool_call"
    assert parsed.tool_call is not None
    evaluation_path = Path("data/evaluation/tool_calling_smoke.jsonl")
    if not evaluation_path.exists():
        pytest.skip("optional evaluation data is not present")
    assert validate_records([record], [str(evaluation_path)])[
        "evaluation_prompt_overlap_count"
    ] == 0
