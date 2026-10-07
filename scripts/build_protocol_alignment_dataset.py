from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from llm_posttrain.agent.protocol import ToolCallParser
from llm_posttrain.rewards.calculator import CalculatorReward
from llm_posttrain.tools.registry import ToolRegistry, build_default_registry


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
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def tool_system_message(registry: ToolRegistry) -> str:
    schema_text = json.dumps(registry.schemas(), ensure_ascii=False, indent=2)
    return (
        "You are running inside Tool Calling Protocol v1.\n"
        "Use the calculator when arithmetic is required.\n"
        f"Available tools:\n{schema_text}\n"
        "When a tool is needed, output exactly one canonical block and nothing else:\n"
        '<tool_call>{"name":"calculator","arguments":{"expression":"2 + 2"}}</tool_call>\n'
        "After the tool observation, output only the final answer in plain text."
    )


def make_expression(index: int) -> str:
    left = 700 + index * 13
    right = 4 + (index % 17)
    third = 2 + (index % 9)
    mode = index % 6
    if mode == 0:
        return f"{left} + {right}"
    if mode == 1:
        return f"{left} - {right}"
    if mode == 2:
        return f"{left} * {right}"
    if mode == 3:
        return f"({left} + {right}) * {third}"
    if mode == 4:
        return f"{left + right} // {third}"
    return f"{left + right} / {third}"


PROMPT_TEMPLATES = (
    "Use the calculator tool to compute {expression}. Return only the result after the tool responds.",
    "Call the calculator for {expression}; do not calculate it mentally.",
    "I need the exact result of {expression}. First use the available calculator tool.",
    "Please use calculator on {expression}, then provide the final answer.",
    "Solve {expression} with the calculator tool and return the result as plain text.",
    "The expression is {expression}. Use the registered calculator before answering.",
)


def make_record(index: int, registry: ToolRegistry) -> dict[str, Any]:
    expression = make_expression(index)
    execution = registry.execute("calculator", {"expression": expression})
    if not execution.ok:
        raise ValueError(f"calculator failed for {expression}: {execution.error}")

    # This is deliberately the exact canonical protocol used by the strict reward.
    call_payload = {
        "name": "calculator",
        "arguments": {"expression": expression},
    }
    call_text = f"<tool_call>{json.dumps(call_payload, ensure_ascii=False, separators=(',', ':'))}</tool_call>"
    observation = (
        "<tool_observation>\n"
        + json.dumps(execution.to_dict(), ensure_ascii=False)
        + "\n</tool_observation>"
    )
    prompt = PROMPT_TEMPLATES[index % len(PROMPT_TEMPLATES)].format(
        expression=expression
    )
    return {
        "id": f"protocol_align_{index:04d}",
        "category": "tool_calling_protocol",
        "messages": [
            {"role": "system", "content": tool_system_message(registry)},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": call_text},
            {"role": "tool", "name": "calculator", "content": observation},
            {"role": "assistant", "content": execution.output["result_text"]},
        ],
        "metadata": {
            "task_type": "canonical_calculator_protocol",
            "expression": expression,
            "answer": execution.output["result_text"],
            "canonical_format": True,
            "call_payload_has_type_field": False,
        },
    }


def validate_records(
    records: list[dict[str, Any]],
    evaluation_paths: list[str | Path],
) -> dict[str, Any]:
    parser = ToolCallParser()
    reward = CalculatorReward(parser)
    registry = build_default_registry()
    evaluation_prompts = {
        normalize(str(message["content"]))
        for path in evaluation_paths
        for sample in read_jsonl(path)
        for message in sample.get("messages", [])
        if message.get("role") == "user"
    }
    overlap = 0
    for record in records:
        messages = record["messages"]
        if [message["role"] for message in messages] != [
            "system",
            "user",
            "assistant",
            "tool",
            "assistant",
        ]:
            raise ValueError(f"{record['id']}: invalid message role sequence")
        user_prompt = normalize(messages[1]["content"])
        overlap += int(user_prompt in evaluation_prompts)
        parsed = parser.parse(messages[2]["content"])
        if parsed.kind != "tool_call" or parsed.tool_call is None:
            raise ValueError(f"{record['id']}: canonical call did not parse")
        if "type" in json.loads(messages[2]["content"][len("<tool_call>") : -len("</tool_call>")]):
            raise ValueError(f"{record['id']}: type field should be absent")
        execution = registry.execute(
            parsed.tool_call.name,
            parsed.tool_call.arguments,
        ).to_dict()
        score = reward.score(
            tool_output=messages[2]["content"],
            final_output=messages[4]["content"],
            expected_tool_name="calculator",
            expected_arguments={"expression": record["metadata"]["expression"]},
            expected_final_answer=record["metadata"]["answer"],
            execution=execution,
        )
        if score.canonical_format_reward != 1.0 or score.total_reward < 0.9:
            raise ValueError(f"{record['id']}: reward validation failed: {score}")
    return {
        "records": len(records),
        "canonical_records": len(records),
        "evaluation_prompt_overlap_count": overlap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build strict canonical tool-call alignment data.")
    parser.add_argument("--num-examples", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--train-output", default="data/protocol_alignment/train.jsonl")
    parser.add_argument("--valid-output", default="data/protocol_alignment/valid.jsonl")
    parser.add_argument("--manifest-output", default="artifacts/protocol_alignment_manifest.json")
    parser.add_argument("--evaluation-dataset", action="append", default=[])
    args = parser.parse_args()
    if args.num_examples < 16:
        raise ValueError("num_examples must be at least 16")

    random.seed(args.seed)
    registry = build_default_registry()
    records = [make_record(index, registry) for index in range(args.num_examples)]
    random.shuffle(records)
    valid_count = max(1, round(len(records) * 0.20))
    valid_records = records[:valid_count]
    train_records = records[valid_count:]
    evaluation_paths = args.evaluation_dataset or [
        "data/evaluation/base_smoke.jsonl",
        "data/evaluation/tool_calling_smoke.jsonl",
        "data/agentic_rl/valid.jsonl",
    ]
    validation = validate_records(records, evaluation_paths)
    write_jsonl(args.train_output, train_records)
    write_jsonl(args.valid_output, valid_records)
    manifest = {
        "version": "protocol_alignment_v1",
        "seed": args.seed,
        "num_examples": len(records),
        "train_count": len(train_records),
        "validation_count": len(valid_records),
        "train_output": args.train_output,
        "validation_output": args.valid_output,
        "train_sha256": sha256(args.train_output),
        "validation_sha256": sha256(args.valid_output),
        "evaluation_datasets": {
            str(path): sha256(path) for path in evaluation_paths
        },
        "evaluation_prompt_overlap_count": validation[
            "evaluation_prompt_overlap_count"
        ],
        "contract": "canonical <tool_call> wrapper; no type field; verified execution; plain-text final answer",
    }
    output = Path(args.manifest_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
