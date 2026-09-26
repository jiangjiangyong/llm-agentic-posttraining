from llm_posttrain.rl.grpo import group_relative_advantages


def test_group_relative_advantages_are_centered_and_ordered() -> None:
    advantages = group_relative_advantages([0.0, 0.5, 1.0])

    assert len(advantages) == 3
    assert advantages[0] < 0 < advantages[2]
    assert abs(sum(advantages)) < 1e-5


def test_group_relative_advantages_are_zero_for_constant_rewards() -> None:
    assert group_relative_advantages([1.0, 1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0, 0.0]
