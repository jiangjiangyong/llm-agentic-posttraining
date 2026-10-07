from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F


def _chat_token_ids(tokenizer: Any, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> list[int]:
    kwargs = {"tokenize": True, "add_generation_prompt": add_generation_prompt}
    try:
        value = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except (TypeError, ValueError):
        value = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(value, dict):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token_id) for token_id in value]


def _text_token_ids(tokenizer: Any, text: str) -> list[int]:
    value = tokenizer(text, add_special_tokens=False)["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token_id) for token_id in value]


def _find_subsequence(sequence: list[int], needle: list[int], start: int) -> int:
    if not needle:
        raise ValueError("cannot search for an empty token sequence")
    width = len(needle)
    for index in range(start, len(sequence) - width + 1):
        if sequence[index : index + width] == needle:
            return index
    return -1


def tokenize_prompt_messages(
    tokenizer: Any,
    prompt: list[dict[str, Any]],
    completion_messages: list[dict[str, Any]],
    *,
    max_seq_length: int,
) -> tuple[list[int], list[int], list[int]]:
    """Tokenize a multi-turn completion while scoring only assistant content."""
    prompt_ids = _chat_token_ids(tokenizer, prompt, add_generation_prompt=True)
    full_messages = list(prompt) + list(completion_messages)
    full_ids = _chat_token_ids(tokenizer, full_messages, add_generation_prompt=False)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Chat template changed the prompt prefix for trajectory DPO")
    if len(prompt_ids) >= max_seq_length:
        raise ValueError("Prompt is too long for max_seq_length; response would be unsupervised")
    if len(full_ids) > max_seq_length:
        full_ids = full_ids[:max_seq_length]
    response_mask = [0] * len(full_ids)
    assistant_header_ids = _text_token_ids(tokenizer, "<|im_start|>assistant\n")
    cursor = len(prompt_ids)
    assistant_count = 0
    first_completion_assistant = True
    for message in completion_messages:
        if message.get("role") != "assistant":
            continue
        assistant_count += 1
        if first_completion_assistant:
            # add_generation_prompt already emitted this header in prompt_ids.
            header_start = cursor - len(assistant_header_ids)
            content_search_start = cursor
            first_completion_assistant = False
        else:
            header_start = _find_subsequence(full_ids, assistant_header_ids, cursor)
            if header_start < 0:
                raise ValueError("assistant header not found in trajectory completion")
            content_search_start = header_start + len(assistant_header_ids)
        content_ids = _text_token_ids(tokenizer, str(message.get("content", "")))
        content_start = _find_subsequence(full_ids, content_ids, content_search_start)
        if content_start < 0:
            raise ValueError("assistant content not found in trajectory completion")
        content_end = content_start + len(content_ids)
        for index in range(max(content_start, len(prompt_ids)), min(content_end, len(full_ids))):
            response_mask[index] = 1
        cursor = content_end
    if assistant_count == 0 or not any(response_mask):
        raise ValueError("trajectory completion has no supervised assistant content")
    return full_ids, [1] * len(full_ids), response_mask


@dataclass(frozen=True)
class DPOItem:
    record_id: str
    chosen_input_ids: list[int]
    chosen_attention_mask: list[int]
    chosen_response_mask: list[int]
    rejected_input_ids: list[int]
    rejected_attention_mask: list[int]
    rejected_response_mask: list[int]


def tokenize_prompt_completion(
    tokenizer: Any,
    prompt: list[dict[str, Any]],
    completion: str,
    *,
    max_seq_length: int,
) -> tuple[list[int], list[int], list[int]]:
    prompt_ids = _chat_token_ids(tokenizer, prompt, add_generation_prompt=True)
    full_messages = list(prompt) + [{"role": "assistant", "content": completion}]
    full_ids = _chat_token_ids(tokenizer, full_messages, add_generation_prompt=False)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Chat template changed the prompt prefix between generation and training")
    if len(prompt_ids) >= max_seq_length:
        raise ValueError("Prompt is too long for max_seq_length; response would be unsupervised")
    if len(full_ids) > max_seq_length:
        full_ids = full_ids[:max_seq_length]
    response_mask = [0] * len(full_ids)
    for index in range(len(prompt_ids), len(full_ids)):
        response_mask[index] = 1
    if not any(response_mask):
        raise ValueError("Completion was truncated completely")
    return full_ids, [1] * len(full_ids), response_mask


class DPODataset:
    def __init__(self, tokenizer: Any, records: list[dict[str, Any]], max_seq_length: int) -> None:
        self.items: list[DPOItem] = []
        for record in records:
            prompt = record.get("prompt")
            if not isinstance(prompt, list) or not prompt:
                raise ValueError(f"{record.get('id')}: prompt must be a non-empty message list")
            chosen_messages = record.get("chosen_messages")
            rejected_messages = record.get("rejected_messages")
            if chosen_messages is not None or rejected_messages is not None:
                if not isinstance(chosen_messages, list) or not isinstance(rejected_messages, list):
                    raise ValueError(
                        f"{record.get('id')}: trajectory DPO completions must be message lists"
                    )
                chosen_ids, chosen_attention, chosen_mask = tokenize_prompt_messages(
                    tokenizer, prompt, chosen_messages, max_seq_length=max_seq_length
                )
                rejected_ids, rejected_attention, rejected_mask = tokenize_prompt_messages(
                    tokenizer, prompt, rejected_messages, max_seq_length=max_seq_length
                )
            else:
                chosen = str(record.get("chosen", ""))
                rejected = str(record.get("rejected", ""))
                if not chosen.strip() or not rejected.strip():
                    raise ValueError(f"{record.get('id')}: chosen and rejected must be non-empty")
                chosen_ids, chosen_attention, chosen_mask = tokenize_prompt_completion(
                    tokenizer, prompt, chosen, max_seq_length=max_seq_length
                )
                rejected_ids, rejected_attention, rejected_mask = tokenize_prompt_completion(
                    tokenizer, prompt, rejected, max_seq_length=max_seq_length
                )
            self.items.append(
                DPOItem(
                    record_id=str(record.get("id", len(self.items))),
                    chosen_input_ids=chosen_ids,
                    chosen_attention_mask=chosen_attention,
                    chosen_response_mask=chosen_mask,
                    rejected_input_ids=rejected_ids,
                    rejected_attention_mask=rejected_attention,
                    rejected_response_mask=rejected_mask,
                )
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> DPOItem:
        return self.items[index]

    def token_summary(self) -> dict[str, int]:
        return {
            "records": len(self.items),
            "chosen_tokens": sum(len(item.chosen_input_ids) for item in self.items),
            "rejected_tokens": sum(len(item.rejected_input_ids) for item in self.items),
            "chosen_response_tokens": sum(sum(item.chosen_response_mask) for item in self.items),
            "rejected_response_tokens": sum(sum(item.rejected_response_mask) for item in self.items),
        }


def _pad_sequences(
    sequences: list[list[int]],
    masks: list[list[int]],
    response_masks: list[list[int]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    max_length = max(len(sequence) for sequence in sequences)
    input_ids: list[list[int]] = []
    attention: list[list[int]] = []
    responses: list[list[int]] = []
    for sequence, mask, response_mask in zip(sequences, masks, response_masks):
        padding = max_length - len(sequence)
        input_ids.append(sequence + [pad_token_id] * padding)
        attention.append(mask + [0] * padding)
        responses.append(response_mask + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "response_mask": torch.tensor(responses, dtype=torch.bool),
    }


class DPOCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, items: list[DPOItem]) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "chosen": _pad_sequences(
                [item.chosen_input_ids for item in items],
                [item.chosen_attention_mask for item in items],
                [item.chosen_response_mask for item in items],
                self.pad_token_id,
            ),
            "rejected": _pad_sequences(
                [item.rejected_input_ids for item in items],
                [item.rejected_attention_mask for item in items],
                [item.rejected_response_mask for item in items],
                self.pad_token_id,
            ),
        }


def sequence_logprob(
    model: Any,
    batch: dict[str, torch.Tensor],
    *,
    length_normalize: bool = True,
) -> torch.Tensor:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    logits = outputs.logits[:, :-1, :].float()
    labels = batch["input_ids"][:, 1:]
    token_logprobs = F.log_softmax(logits, dim=-1).gather(
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)
    response_mask = batch["response_mask"][:, 1:].to(token_logprobs.dtype)
    summed = (token_logprobs * response_mask).sum(dim=-1)
    if not length_normalize:
        return summed
    token_count = response_mask.sum(dim=-1).clamp_min(1.0)
    return summed / token_count


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
    *,
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    policy_logratio = policy_chosen - policy_rejected
    reference_logratio = reference_chosen - reference_rejected
    logits = beta * (policy_logratio - reference_logratio)
    loss = -F.logsigmoid(logits).mean()
    chosen_reward = beta * (policy_chosen - reference_chosen)
    rejected_reward = beta * (policy_rejected - reference_rejected)
    metrics = {
        "loss": float(loss.detach().cpu()),
        "preference_accuracy": float((logits > 0).float().mean().detach().cpu()),
        "margin": float(logits.mean().detach().cpu()),
        "chosen_reward": float(chosen_reward.mean().detach().cpu()),
        "rejected_reward": float(rejected_reward.mean().detach().cpu()),
    }
    return loss, metrics
