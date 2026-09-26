import torch

from llm_posttrain.training.dpo import dpo_loss


def test_dpo_loss_prefers_positive_margin() -> None:
    loss, metrics = dpo_loss(
        torch.tensor([[-1.0]]),
        torch.tensor([[-2.0]]),
        torch.tensor([[-1.0]]),
        torch.tensor([[-2.0]]),
        beta=0.1,
    )
    assert torch.isfinite(loss)
    assert metrics["preference_accuracy"] == 0.0


def test_dpo_loss_reports_reference_relative_margin() -> None:
    _, metrics = dpo_loss(
        torch.tensor([[-1.0]]),
        torch.tensor([[-3.0]]),
        torch.tensor([[-2.0]]),
        torch.tensor([[-3.0]]),
        beta=0.1,
    )
    assert metrics["preference_accuracy"] == 1.0
    assert metrics["margin"] > 0.0
