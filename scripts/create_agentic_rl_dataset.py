from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from llm_posttrain.tools.registry import build_default_registry


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


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


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tool_system_message(registry: Any) -> str:
    schema_text = json.dumps(registry.schemas(), ensure_ascii=False, indent=2)
    return (
        "You are running inside Tool Calling Protocol v1.\n"
        "Available tools are listed below.\n"
        f"{schema_text}\n"
        "If a tool is needed, output exactly one block in this form:\n"
        '<tool_call>{"name":"calculator","arguments":{"expression":"2 + 2"}}</tool_call>\n'
        "Do not add Markdown fences around a tool call. After the tool observation, "
        "output the final answer in plain text."
    )


def make_expression(index: int) -> str:
    left = 1001 + index * 7
    right = 4 + (index % 17)
    third = 2 + (index % 8)
    mode = index % 6
    if mode == 0:
        return f"{left} + {right}"
    if mode == 1:
        return f"{left} * {right}"
    if mode == 2:
        return f"({left} + {right}) * {third}"
    if mode == 3:
        return f"{left + right} / {third}"
    if mode == 4:
        return f"{left + right} // {third}"
    return f"{left} ** 2"


def build_records(num_examples: int, seed: int) -> list[dict[str, Any]]:
    if num_examples < 8:
        raise ValueError("num_examples must be at least 8")
    registry = build_default_registry()
    variants = (
        "Use the calculator tool to evaluate {expression}. Return only the result after the tool responds.",
        "Please call the arithmetic tool for {expression}, then provide the final answer.",
        "Solve {expression} by using the available calculator; do not calculate it mentally.",
        "I need the exact result of {expression}. First use the calculator tool.",
    )
    records: list[dict[str, Any]] = []
    for index in range(num_examples):
        expression = make_expression(700 + index)
        execution = registry.execute("calculator", {"expression": expression})
        if not execution.ok:
            raise ValueError(f"Failed to create expression {expression}: {execution.error}")
        records.append(
            {
                "id": f"rl_calc_{index:04d}",
                "category": "calculator_agent",
                "messages": [
                    {"role": "system", "content": tool_system_message(registry)},
                    {"role": "user", "content": variants[index % len(variants)].format(expression=expression)},
                ],
                "expected": {
                    "tool_name": "calculator",
                    "arguments": {"expression": expression},
                    "final_answer": execution.output["result_text"],
                },
                "metadata": {"expression": expression, "task_type": "two_turn_calculator"},
            }
        )
    random.Random(seed).shuffle(records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Create isolated Calculator Environment tasks for GRPO.")
    parser.add_argument("--num-examples", type=int, default=24)
    parser.add_argument("--validation-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--train-output", default="data/agentic_rl/train.jsonl")
    parser.add_argument("--valid-output", default="data/agentic_rl/valid.jsonl")
    parser.add_argument("--manifest-output", default="artifacts/agentic_rl_manifest.json")
    parser.add_argument(
        "--eval-dataset",
        action="append",
        default=["data/evaluation/base_smoke.jsonl", "data/evaluation/tool_calling_smoke.jsonl"],
    )
    args = parser.parse_args()
    if not 0 < args.validation_ratio < 0.5:
        raise ValueError("validation ratio must be between 0 and 0.5")
    records = build_records(args.num_examples, args.seed)
    eval_prompts: set[str] = set()
    eval_hashes: dict[str, str] = {}
    for path in args.eval_dataset:
        eval_hashes[path] = file_sha256(path)
        for record in read_jsonl(path):
            for message in record.get("messages", []):
                if message.get("role") == "user":
                    eval_prompts.add(normalize_text(str(message.get("content", ""))))
    source_prompts = {
        normalize_text(str(message.get("content", "")))
        for record in records
        for message in record["messages"]
        if message.get("role") == "user"
    }
    overlap = sorted(eval_prompts & source_prompts)
    if overlap:
        raise ValueError(f"Agentic RL source overlaps frozen evaluation prompts: {overlap}")
    split = max(1, round(len(records) * args.validation_ratio))
    valid = records[:split]
    train = records[split:]
    write_jsonl(args.train_output, train)
    write_jsonl(args.valid_output, valid)
    manifest = {
        "version": "agentic_rl_source_v1",
        "seed": args.seed,
        "source_count": len(records),
        "train_count": len(train),
        "validation_count": len(valid),
        "train_output": args.train_output,
        "validation_output": args.valid_output,
        "evaluation_datasets": eval_hashes,
        "evaluation_prompt_overlap_count": len(overlap),
        "environment": "CalculatorEnvironment; two model turns; deterministic tool execution",
    }
    output_path = Path(args.manifest_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
