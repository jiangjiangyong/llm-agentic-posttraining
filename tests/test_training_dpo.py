import torch

from llm_posttrain.training.dpo import dpo_loss, tokenize_prompt_messages


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


class _DPOFakeTokenizer:
    def _encode(self, text: str) -> list[int]:
        return [1000 + ord(char) for char in text]

    def __call__(self, text: str, *, add_special_tokens: bool):
        del add_special_tokens
        return {"input_ids": self._encode(text)}

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool = False,
    ):
        del enable_thinking
        rendered = "".join(
            "<|im_start|>" + str(message["role"]) + "\n"
            + str(message.get("content", "")) + "<|im_end|>\n"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        return self._encode(rendered) if tokenize else rendered


def _find_ids(sequence: list[int], needle: list[int]) -> int:
    width = len(needle)
    for index in range(len(sequence) - width + 1):
        if sequence[index : index + width] == needle:
            return index
    raise AssertionError("needle not found")


def test_trajectory_dpo_masks_tool_observations():
    tokenizer = _DPOFakeTokenizer()
    prompt = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    completion = [
        {"role": "assistant", "content": "TOOL_CALL"},
        {"role": "tool", "content": "TOOL_OBSERVATION"},
        {"role": "assistant", "content": "FINAL_ANSWER"},
    ]
    input_ids, _, response_mask = tokenize_prompt_messages(
        tokenizer, prompt, completion, max_seq_length=256
    )
    tool_ids = tokenizer("TOOL_CALL", add_special_tokens=False)["input_ids"]
    observation_ids = tokenizer("TOOL_OBSERVATION", add_special_tokens=False)["input_ids"]
    final_ids = tokenizer("FINAL_ANSWER", add_special_tokens=False)["input_ids"]
    tool_start = _find_ids(input_ids, tool_ids)
    observation_start = _find_ids(input_ids, observation_ids)
    final_start = _find_ids(input_ids, final_ids)
    assert all(response_mask[tool_start : tool_start + len(tool_ids)])
    assert not any(response_mask[observation_start : observation_start + len(observation_ids)])
    assert all(response_mask[final_start : final_start + len(final_ids)])
