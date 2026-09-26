from __future__ import annotations

from typing import Any

import torch

from llm_posttrain.models.loader import decode_generation, model_input_device, render_chat_prompt
from llm_posttrain.training.dpo import sequence_logprob, tokenize_prompt_completion


def sample_completion(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool = True,
) -> str:
    prompt = render_chat_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = model_input_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)
    prompt_length = int(inputs["input_ids"].shape[1])
    completion_ids = output_ids[:, prompt_length:]
    return decode_generation(tokenizer, completion_ids)


def group_relative_advantages(rewards: list[float], eps: float = 1e-6) -> list[float]:
    if not rewards:
        return []
    values = torch.tensor(rewards, dtype=torch.float32)
    centered = values - values.mean()
    std = values.std(unbiased=False)
    if float(std) < eps:
        return [0.0 for _ in rewards]
    return (centered / (std + eps)).tolist()


def completion_logprob(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    completion: str,
    *,
    max_seq_length: int,
) -> tuple[torch.Tensor, int]:
    input_ids, attention_mask, response_mask = tokenize_prompt_completion(
        tokenizer,
        messages,
        completion,
        max_seq_length=max_seq_length,
    )
    device = model_input_device(model)
    batch = {
        "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long, device=device),
        "response_mask": torch.tensor([response_mask], dtype=torch.bool, device=device),
    }
    return sequence_logprob(model, batch, length_normalize=False)[0], sum(response_mask)


def trajectory_logprob(
    model: Any,
    tokenizer: Any,
    episode: Any,
    *,
    max_seq_length: int,
    normalize_by_tokens: bool = True,
) -> tuple[torch.Tensor, int]:
    total_logprob: torch.Tensor | None = None
    total_tokens = 0
    for step in episode.steps:
        logprob, token_count = completion_logprob(
            model,
            tokenizer,
            step["messages"],
            str(step["model_output"]),
            max_seq_length=max_seq_length,
        )
        total_logprob = logprob if total_logprob is None else total_logprob + logprob
        total_tokens += token_count
    if total_logprob is None:
        total_logprob = torch.zeros((), device=model_input_device(model), requires_grad=True)
    if normalize_by_tokens:
        total_logprob = total_logprob / max(total_tokens, 1)
    return total_logprob, total_tokens
