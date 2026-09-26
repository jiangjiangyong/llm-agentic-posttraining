from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


def load_causal_lm(model_path: str | Path, device_map: str = "auto"):
    model_path = str(Path(model_path))
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }

    if torch.cuda.is_available():
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["device_map"] = device_map
    else:
        load_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.eval()
    return tokenizer, model


def render_chat_prompt(tokenizer, messages: list[dict[str, Any]]) -> str:
    common = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,
            **common,
        )
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(messages, **common)


def decode_generation(tokenizer: Any, token_ids: Any) -> str:
    text = tokenizer.batch_decode(
        token_ids,
        skip_special_tokens=False,
    )[0]
    for marker in ("<|im_end|>", "<|endoftext|>"):
        text = text.replace(marker, "")
    return text.strip()

def model_input_device(model) -> torch.device:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for value in device_map.values():
            if isinstance(value, int):
                return torch.device(f"cuda:{value}")
            if isinstance(value, str) and value not in {"cpu", "disk"}:
                return torch.device(value)

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelRunner:
    def __init__(
        self,
        model_path: str | Path,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        device_map: str = "auto",
    ) -> None:
        self.model_path = str(model_path)
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        print(f"[ModelRunner] Loading tokenizer from {self.model_path}")
        self.tokenizer, self.model = load_causal_lm(
            self.model_path,
            device_map=device_map,
        )
        print("[ModelRunner] Loading model complete")

    def generate(self, messages: list[dict[str, Any]]) -> str:
        return self.generate_with_stats(messages).text

    def generate_with_stats(
        self,
        messages: list[dict[str, Any]],
    ) -> GenerationResult:
        prompt = render_chat_prompt(self.tokenizer, messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = model_input_device(self.model)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        started = time.perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000

        prompt_length = int(inputs["input_ids"].shape[1])
        new_tokens = output_ids[:, prompt_length:]
        text = decode_generation(self.tokenizer, new_tokens)

        return GenerationResult(
            text=text,
            prompt_tokens=prompt_length,
            completion_tokens=int(new_tokens.shape[1]),
            latency_ms=elapsed_ms,
        )


def build_runner(config: dict[str, Any]) -> ModelRunner:
    model_config = config.get("model", {})
    inference_config = config.get("inference", {})
    return ModelRunner(
        model_path=model_config["local_path"],
        max_new_tokens=int(inference_config.get("max_new_tokens", 128)),
        do_sample=bool(inference_config.get("do_sample", False)),
        device_map=str(model_config.get("device_map", "auto")),
    )
