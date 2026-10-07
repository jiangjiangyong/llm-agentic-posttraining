from __future__ import annotations

from scripts.run_experiment_1_failure_taxonomy import (
    aggregate,
    classify_episode,
    is_canonical_tool_output,
    recovery_kind,
)


def _task() -> dict:
    return {
        "id": "case-1",
        "category": "mock_search",
        "task_type": "mock_search",
        "expected": {
            "required_tools": ["mock_search"],
            "expected_arguments": {"mock_search": {"query": "fact card"}},
            "execution_tools": ["mock_search"],
            "expected_tool_outputs": {"mock_search": {"query": "fact card"}},
            "final_answer_contains": ["fact"],
        },
    }


def _row(**overrides: object) -> dict:
    row = {
        "task_id": "case-1",
        "category": "mock_search",
        "steps": [
            {
                "step": 0,
                "model_output": '{"name":"mock_search","arguments":{"query":"wrong"}}',
                "parsed_kind": "tool_call",
                "parse_error": None,
                "tool_call": {"name": "mock_search", "arguments": {"query": "wrong"}},
                "tool_calls": [{"name": "mock_search", "arguments": {"query": "wrong"}}],
                "tool_results": [
                    {
                        "tool_name": "mock_search",
                        "ok": True,
                        "output": {"query": "wrong", "results": []},
                    }
                ],
            },
            {
                "step": 1,
                "model_output": "fact",
                "parsed_kind": "final",
                "parse_error": None,
            },
        ],
        "tool_outputs": ['{"name":"mock_search","arguments":{"query":"wrong"}}'],
        "tool_calls": [{"name": "mock_search", "arguments": {"query": "wrong"}}],
        "tool_results": [
            {
                "tool_name": "mock_search",
                "ok": True,
                "output": {"query": "wrong", "results": []},
            }
        ],
        "final_output": "fact",
        "semantic_success": False,
        "strict_success": False,
        "runtime_success": True,
        "reward": {
            "argument_reward": 0.0,
            "execution_reward": 0.0,
            "unit_test_reward": 1.0,
            "final_answer_reward": 1.0,
            "total_reward": 0.2,
        },
        "workspace_files": [],
    }
    row.update(overrides)
    return row


def test_canonical_and_recovery_labels() -> None:
    assert is_canonical_tool_output(
        '<tool_call>{"name":"mock_search","arguments":{}}</tool_call>'
    )
    assert not is_canonical_tool_output('{"name":"mock_search","arguments":{}}')
    assert recovery_kind('{"name":"mock_search","arguments":{}}') == (
        "bare_json_or_json_prefix"
    )


def test_primary_failure_has_fixed_priority_and_secondary_signals() -> None:
    result = classify_episode(_task(), _row(), max_steps=5)
    assert result["primary_failure"] == "argument"
    assert "execution" in result["secondary_failures"]
    assert "protocol_wrapper" in result["secondary_failures"]
    assert result["parser_recovery_used"] is True


def test_parser_rescue_is_only_semantic_success_without_strict_success() -> None:
    row = _row(
        semantic_success=True,
        strict_success=False,
        reward={
            "argument_reward": 1.0,
            "execution_reward": 1.0,
            "unit_test_reward": 1.0,
            "final_answer_reward": 1.0,
            "total_reward": 0.9,
        },
    )
    result = classify_episode(_task(), row, max_steps=5)
    assert result["parser_rescued_task"] is True


def test_aggregate_has_category_metrics_without_recursive_nesting() -> None:
    row = _row()
    row["classification"] = classify_episode(_task(), row, max_steps=5)
    metrics = aggregate([row])
    assert metrics["episodes"] == 1
    assert metrics["by_category"]["mock_search"]["episodes"] == 1
    assert "by_category" not in metrics["by_category"]["mock_search"]
