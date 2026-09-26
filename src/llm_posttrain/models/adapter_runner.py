from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .loader import (
    GenerationResult,
    decode_generation,
    model_input_device,
    render_chat_prompt,
)


def load_quantized_adapter(
    base_model_path: str | Path,
    adapter_path: str | Path,
):
    base_model_path = str(Path(base_model_path))
    adapter_path = str(Path(adapter_path))
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        local_files_only=True,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=False,
        local_files_only=True,
    )
    model.eval()
    model.config.use_cache = True
    return tokenizer, model


class AdapterModelRunner:
    def __init__(
        self,
        base_model_path: str | Path,
        adapter_path: str | Path,
        max_new_tokens: int = 128,
        do_sample: bool = False,
    ) -> None:
        self.base_model_path = str(base_model_path)
        self.adapter_path = str(adapter_path)
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        print(f"[AdapterModelRunner] Loading base model from {self.base_model_path}")
        print(f"[AdapterModelRunner] Loading adapter from {self.adapter_path}")
        self.tokenizer, self.model = load_quantized_adapter(
            self.base_model_path,
            self.adapter_path,
        )
        print("[AdapterModelRunner] Loading model and adapter complete")

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


def build_adapter_runner(
    model_config: dict[str, Any],
    adapter_path: str | Path,
    *,
    max_new_tokens: int = 128,
    do_sample: bool = False,
) -> AdapterModelRunner:
    return AdapterModelRunner(
        base_model_path=model_config["model"]["local_path"],
        adapter_path=adapter_path,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )
