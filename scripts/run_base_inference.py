from __future__ import annotations

import argparse

import torch

from llm_posttrain.config import load_yaml
from llm_posttrain.models.loader import ModelRunner, render_chat_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_model.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    model_config = config["model"]
    inference_config = config.get("inference", {})
    model_path = args.model or model_config["local_path"]
    max_new_tokens = args.max_new_tokens or int(inference_config.get("max_new_tokens", 128))

    runner = ModelRunner(
        model_path=model_path,
        max_new_tokens=max_new_tokens,
        do_sample=bool(inference_config.get("do_sample", False)),
        device_map=str(model_config.get("device_map", "auto")),
    )
    messages = [
        {"role": "system", "content": "You are a precise math assistant. Output only the final answer."},
        {"role": "user", "content": "Calculate 125 * 36."},
    ]
    prompt = render_chat_prompt(runner.tokenizer, messages)
    result = runner.generate_with_stats(messages)
    print(f"prompt_chars: {len(prompt)}")
    print(f"prompt_tokens: {result.prompt_tokens}")
    print(f"completion_tokens: {result.completion_tokens}")
    print(f"latency_ms: {result.latency_ms:.1f}")
    print("--- model output ---")
    print(result.text)
    if torch.cuda.is_available():
        print(f"max_memory_allocated_gib: {torch.cuda.max_memory_allocated() / 1024**3:.2f}")


if __name__ == "__main__":
    main()
