import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_agent_benchmark import aggregate_agent, aggregate_general, aggregate_tool


def test_aggregate_general_reports_category_accuracy() -> None:
    metrics = aggregate_general(
        [
            {"category": "math", "passed": True, "latency_ms": 10},
            {"category": "math", "passed": False, "latency_ms": 20},
            {"category": "json", "passed": True, "latency_ms": 30},
        ]
    )

    assert metrics["total"] == 3
    assert metrics["passed"] == 2
    assert metrics["by_category"]["math"]["accuracy"] == 0.5


def test_aggregate_tool_uses_calls_as_conditional_denominator() -> None:
    metrics = aggregate_tool(
        [
            {
                "tool_call_present": True,
                "tool_name_correct": True,
                "arguments_correct": True,
                "final_answer_exact": True,
                "task_success": True,
                "elapsed_ms": 10,
            },
            {
                "tool_call_present": False,
                "tool_name_correct": False,
                "arguments_correct": False,
                "final_answer_exact": False,
                "task_success": False,
                "elapsed_ms": 20,
            },
        ]
    )

    assert metrics["tool_call_parse_rate"] == 0.5
    assert metrics["tool_name_accuracy_given_call"] == 1.0
    assert metrics["argument_accuracy_given_call"] == 1.0
    assert metrics["task_success_rate"] == 0.5


def test_aggregate_agent_preserves_reward_components() -> None:
    reward = {
        "format_reward": 1.0,
        "canonical_format_reward": 0.5,
        "tool_selection_reward": 1.0,
        "argument_reward": 0.0,
        "execution_reward": 1.0,
        "final_answer_reward": 0.0,
        "total_reward": 0.45,
    }
    metrics = aggregate_agent(
        [
            {"reward": reward, "success": False, "elapsed_ms": 5},
            {"reward": reward, "success": True, "elapsed_ms": 7},
        ]
    )

    assert metrics["total"] == 2
    assert metrics["mean_reward"] == 0.45
    assert metrics["strict_success_rate"] == 0.5
    assert metrics["mean_components"]["argument_reward"] == 0.0
