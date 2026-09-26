from llm_posttrain.rewards.calculator import CalculatorReward


def test_calculator_reward_full_success() -> None:
    reward = CalculatorReward().score(
        tool_output=(
            '<tool_call>{"name":"calculator",'
            '"arguments":{"expression":"2 + 2"}}</tool_call>'
        ),
        final_output="4",
        expected_tool_name="calculator",
        expected_arguments={"expression": "2 + 2"},
        expected_final_answer="4",
        execution={"ok": True, "output": {"result_text": "4"}},
    )

    assert reward.total_reward == 1.0
    assert reward.parser_error is None


def test_calculator_reward_distinguishes_noncanonical_output() -> None:
    reward = CalculatorReward().score(
        tool_output=(
            '{"name":"calculator",'
            '"arguments":{"expression":"2 + 2"}}'
        ),
        final_output="4",
        expected_tool_name="calculator",
        expected_arguments={"expression": "2 + 2"},
        expected_final_answer="4",
        execution={"ok": True},
    )

    assert reward.format_reward == 1.0
    assert reward.canonical_format_reward == 0.0
    assert abs(reward.total_reward - 0.9) < 1e-9


def test_calculator_reward_invalid_tool_output_gets_no_execution_credit() -> None:
    reward = CalculatorReward().score(
        tool_output="<tool_call>{bad json</tool_call>",
        final_output=None,
        expected_tool_name="calculator",
        expected_arguments={"expression": "2 + 2"},
        expected_final_answer="4",
        execution=None,
    )

    assert reward.total_reward == 0.0
    assert reward.parser_error is not None

def test_calculator_reward_rejects_typed_wrapper_as_canonical() -> None:
    reward = CalculatorReward().score(
        tool_output=(
            '<tool_call>{"type":"tool_call","name":"calculator",'
            '"arguments":{"expression":"2 + 2"}}</tool_call>'
        ),
        final_output="4",
        expected_tool_name="calculator",
        expected_arguments={"expression": "2 + 2"},
        expected_final_answer="4",
        execution={"ok": True, "output": {"result_text": "4"}},
    )

    assert reward.format_reward == 1.0
    assert reward.canonical_format_reward == 0.0
    assert abs(reward.total_reward - 0.9) < 1e-9
