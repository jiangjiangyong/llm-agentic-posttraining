from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from llm_posttrain.config import load_yaml
from llm_posttrain.models.loader import render_chat_prompt


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at line {line_number}")
            records.append(value)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect chat-template rendering and tokenization for one eval sample."
    )
    parser.add_argument("--config", default="configs/base_model.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--max-display-tokens", type=int, default=160)
    args = parser.parse_args()

    config = load_yaml(args.config)
    model_path = args.model or config["model"]["local_path"]
    dataset_path = config["evaluation"]["dataset_path"]
    samples = read_jsonl(dataset_path)
    if not samples:
        raise ValueError(f"No samples found in {dataset_path}")

    sample = samples[0]
    if args.sample_id is not None:
        matches = [item for item in samples if item.get("id") == args.sample_id]
        if not matches:
            available = ", ".join(str(item.get("id")) for item in samples)
            raise ValueError(
                f"Unknown sample id {args.sample_id!r}. Available ids: {available}"
            )
        sample = matches[0]

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    prompt = render_chat_prompt(tokenizer, sample["messages"])
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"][0].tolist()
    token_strings = tokenizer.convert_ids_to_tokens(input_ids)

    print(f"model_path: {model_path}")
    print(f"dataset_path: {dataset_path}")
    print(f"sample_id: {sample.get('id')}")
    print(f"message_count: {len(sample['messages'])}")
    print(f"prompt_chars: {len(prompt)}")
    print(f"prompt_tokens: {len(input_ids)}")
    print("--- messages ---")
    print(json.dumps(sample["messages"], ensure_ascii=False, indent=2))
    print("--- rendered prompt ---")
    print(prompt)
    print("--- token trace ---")
    limit = min(args.max_display_tokens, len(input_ids))
    for index in range(limit):
        token_id = input_ids[index]
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        print(
            f"{index:04d} id={token_id:<8d} "
            f"token={token_strings[index]!r} decoded={decoded!r}"
        )
    if limit < len(input_ids):
        print(f"... {len(input_ids) - limit} more tokens omitted")


if __name__ == "__main__":
    main()
