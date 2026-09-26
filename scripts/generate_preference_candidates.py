from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from llm_posttrain.config import load_yaml
from llm_posttrain.models.adapter_runner import build_adapter_runner
from llm_posttrain.models.loader import build_runner


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at line {line_number} in {path}")
            records.append(value)
    return records


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model candidates for preference pairs.")
    parser.add_argument("--backend", choices=("base", "adapter"), required=True)
    parser.add_argument("--source", default="data/preference/source.jsonl")
    parser.add_argument("--model-config", default="configs/base_model.yaml")
    parser.add_argument("--adapter-path", default="models/adapters/sft_qwen3_1.7b")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--do-sample", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = read_jsonl(args.source)
    if not records:
        raise ValueError(f"No preference source records found in {args.source}")
    model_config = load_yaml(args.model_config)
    if args.backend == "base":
        runner = build_runner(model_config)
        runner.max_new_tokens = args.max_new_tokens
        runner.do_sample = args.do_sample
        model_path = model_config["model"]["local_path"]
    else:
        runner = build_adapter_runner(
            model_config,
            args.adapter_path,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
        )
        model_path = model_config["model"]["local_path"]

    output_path = Path(args.output or f"data/preference/candidates/{args.backend}.jsonl")
    candidates: list[dict[str, Any]] = []
    for record in tqdm(records, desc=f"Generating ({args.backend})"):
        result = runner.generate_with_stats(record["messages"])
        candidates.append(
            {
                "source_id": record["id"],
                "backend": args.backend,
                "candidate": result.text,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "latency_ms": result.latency_ms,
                "model_path": model_path,
                "adapter_path": args.adapter_path if args.backend == "adapter" else None,
            }
        )
    write_jsonl(output_path, candidates)
    print(f"Generated {len(candidates)} candidates")
    print(f"Candidates: {output_path}")


if __name__ == "__main__":
    main()
