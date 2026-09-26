from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from llm_posttrain.agent.protocol import ToolCallParser
from llm_posttrain.tools.registry import ToolRegistry, build_default_registry


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


def tool_system_message(registry: ToolRegistry) -> str:
    schema_text = json.dumps(registry.schemas(), ensure_ascii=False, indent=2)
    return (
        "You are running inside Tool Calling Protocol v1.\n"
        "Use the calculator when arithmetic is required.\n"
        f"Available tools:\n{schema_text}\n"
        "When a tool is needed, output exactly one block:\n"
        '<tool_call>{"name":"calculator","arguments":{"expression":"2 + 2"}}</tool_call>\n'
        "After the tool observation, output the final answer in plain text."
    )


def make_tool_record(
    index: int,
    expression: str,
    registry: ToolRegistry,
) -> dict[str, Any]:
    execution = registry.execute("calculator", {"expression": expression})
    if not execution.ok:
        raise ValueError(f"Generated expression failed: {expression}: {execution.error}")
    call_payload = {
        "type": "tool_call",
        "name": "calculator",
        "arguments": {"expression": expression},
    }
    observation = (
        "<tool_observation>\n"
        + json.dumps(execution.to_dict(), ensure_ascii=False)
        + "\n</tool_observation>"
    )
    return {
        "id": f"sft_tool_{index:04d}",
        "category": "tool_calling",
        "messages": [
            {"role": "system", "content": tool_system_message(registry)},
            {
                "role": "user",
                "content": f"Use the calculator tool to compute {expression}. Return the result.",
            },
            {
                "role": "assistant",
                "content": f"<tool_call>{json.dumps(call_payload, ensure_ascii=False)}</tool_call>",
            },
            {
                "role": "tool",
                "name": "calculator",
                "content": observation,
            },
            {
                "role": "assistant",
                "content": execution.output["result_text"],
            },
        ],
        "metadata": {
            "task_type": "calculator",
            "expression": expression,
            "answer": execution.output["result_text"],
        },
    }


def make_math_record(
    index: int,
    expression: str,
    registry: ToolRegistry,
) -> dict[str, Any]:
    execution = registry.execute("calculator", {"expression": expression})
    if not execution.ok:
        raise ValueError(f"Generated expression failed: {expression}: {execution.error}")
    return {
        "id": f"sft_math_{index:04d}",
        "category": "math",
        "messages": [
            {
                "role": "system",
                "content": "You are a precise math assistant. Output only the final number.",
            },
            {
                "role": "user",
                "content": f"Calculate {expression}.",
            },
            {
                "role": "assistant",
                "content": execution.output["result_text"],
            },
        ],
        "metadata": {
            "task_type": "arithmetic",
            "expression": expression,
            "answer": execution.output["result_text"],
        },
    }


def make_json_record(
    index: int,
    description: str,
    category: str,
    priority: str,
) -> dict[str, Any]:
    answer = {"category": category, "priority": priority}
    return {
        "id": f"sft_json_{index:04d}",
        "category": "json",
        "messages": [
            {
                "role": "system",
                "content": "Return valid JSON only with category and priority.",
            },
            {
                "role": "user",
                "content": (
                    f"Classify this support issue: {description}. "
                    f"The correct category is {category} and priority is {priority}."
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(answer, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        "metadata": {
            "task_type": "structured_classification",
            "answer": answer,
        },
    }


_CODE_TASKS = [
    ("clamp_value", "clamp_value(x, low, high)", "return max(low, min(x, high))"),
    ("count_vowels", "count_vowels(text)", "return sum(char.lower() in 'aeiou' for char in text)"),
    ("is_palindrome", "is_palindrome(text)", "return text == text[::-1]"),
    ("safe_get", "safe_get(mapping, key, default)", "return mapping.get(key, default)"),
    ("sum_positive", "sum_positive(values)", "return sum(value for value in values if value > 0)"),
    ("max_pair", "max_pair(left, right)", "return max(left, right)"),
    ("min_pair", "min_pair(left, right)", "return min(left, right)"),
    ("normalize_space", "normalize_space(text)", "return ' '.join(text.split())"),
    ("first_or_none", "first_or_none(values)", "return values[0] if values else None"),
    ("is_multiple", "is_multiple(value, divisor)", "return divisor != 0 and value % divisor == 0"),
    ("to_upper_words", "to_upper_words(words)", "return [word.upper() for word in words]"),
    ("product_values", "product_values(values)", "return math.prod(values)"),
    ("strip_prefix", "strip_prefix(text, prefix)", "return text[len(prefix):] if text.startswith(prefix) else text"),
    ("unique_sorted", "unique_sorted(values)", "return sorted(set(values))"),
    ("difference", "difference(left, right)", "return left - right"),
    ("mean_value", "mean_value(values)", "return sum(values) / len(values) if values else 0"),
    ("reverse_words", "reverse_words(text)", "return ' '.join(text.split()[::-1])"),
    ("find_index", "find_index(values, target)", "return values.index(target) if target in values else -1"),
    ("make_pairs", "make_pairs(values)", "return list(zip(values[::2], values[1::2]))"),
    ("digit_sum", "digit_sum(number)", "return sum(int(char) for char in str(abs(number)))"),
]


def make_code_record(index: int, task: tuple[str, str, str]) -> dict[str, Any]:
    name, signature, body = task
    if name == "product_values":
        imports = "import math\n\n"
    else:
        imports = ""
    code = f"{imports}def {signature}:\n    {body}"
    return {
        "id": f"sft_code_{index:04d}",
        "category": "code",
        "messages": [
            {
                "role": "system",
                "content": "Output valid Python code only. Do not use Markdown fences.",
            },
            {
                "role": "user",
                "content": f"Implement {signature}, following its name and arguments.",
            },
            {"role": "assistant", "content": code},
        ],
        "metadata": {
            "task_type": "python_function",
            "required_symbols": [name],
        },
    }


def make_expression(index: int) -> str:
    left = 17 + index
    right = 3 + (index % 11)
    third = 2 + (index % 7)
    mode = index % 5
    if mode == 0:
        return f"{left} + {right}"
    if mode == 1:
        return f"{left} * {right}"
    if mode == 2:
        return f"({left} + {right}) * {third}"
    if mode == 3:
        return f"{left + right} / {third}"
    return f"{left + right} // {third}"


_JSON_TASKS = [
    ("A user cannot sign in after several wrong passwords", "account", "high"),
    ("A paid order is waiting for shipment", "order", "high"),
    ("The customer wants to change a profile photo", "profile", "low"),
    ("The monthly report export returns an empty file", "report", "medium"),
    ("A card payment was declined at checkout", "payment", "high"),
    ("The user asks how to update a delivery address", "order", "medium"),
    ("A notification email uses the wrong display name", "notification", "low"),
    ("The dashboard takes too long to load", "performance", "medium"),
    ("A password reset link has expired", "account", "medium"),
    ("The invoice has an incorrect tax amount", "billing", "high"),
    ("A user wants to close an unused workspace", "workspace", "low"),
    ("The search result misses an exact product title", "search", "medium"),
]


def build_records(num_examples: int, seed: int) -> list[dict[str, Any]]:
    if num_examples < 8:
        raise ValueError("num_examples must be at least 8")
    registry = build_default_registry()
    tool_count = max(1, round(num_examples * 0.50))
    code_count = max(1, round(num_examples * 0.20))
    json_count = max(1, round(num_examples * 0.20))
    math_count = num_examples - tool_count - code_count - json_count
    if math_count < 1:
        math_count = 1
        tool_count = num_examples - code_count - json_count - math_count
    records: list[dict[str, Any]] = []

    for index in range(tool_count):
        records.append(make_tool_record(index, make_expression(index), registry))
    for index in range(code_count):
        records.append(make_code_record(index, _CODE_TASKS[index % len(_CODE_TASKS)]))
    for index in range(json_count):
        description, category, priority = _JSON_TASKS[index % len(_JSON_TASKS)]
        records.append(make_json_record(index, description, category, priority))
    for index in range(math_count):
        records.append(make_math_record(index, make_expression(index + 100), registry))

    random.Random(seed).shuffle(records)
    return records


def extract_tool_observation(content: str) -> dict[str, Any]:
    match = re.fullmatch(
        r"<tool_observation>\s*(.*?)\s*</tool_observation>",
        content.strip(),
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError("tool observation does not use the canonical wrapper")
    payload = json.loads(match.group(1))
    if not isinstance(payload, dict):
        raise ValueError("tool observation must be a JSON object")
    return payload


def validate_record(
    record: dict[str, Any],
    registry: ToolRegistry,
    parser: ToolCallParser,
) -> None:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{record.get('id')}: messages must be a non-empty list")
    assistant_messages = [
        message for message in messages if message.get("role") == "assistant"
    ]
    if not assistant_messages:
        raise ValueError(f"{record.get('id')}: missing assistant message")

    if record["category"] == "tool_calling":
        call_messages = [
            message for message in assistant_messages
            if "<tool_call>" in str(message.get("content", ""))
        ]
        if len(call_messages) != 1:
            raise ValueError(f"{record.get('id')}: expected one assistant tool call")
        parsed = parser.parse(call_messages[0]["content"])
        if parsed.kind != "tool_call" or parsed.tool_call is None:
            raise ValueError(f"{record.get('id')}: tool call did not parse")
        execution = registry.execute(
            parsed.tool_call.name,
            parsed.tool_call.arguments,
        )
        if not execution.ok:
            raise ValueError(f"{record.get('id')}: tool call failed: {execution.error}")
        observation_messages = [
            message for message in messages if message.get("role") == "tool"
        ]
        if len(observation_messages) != 1:
            raise ValueError(f"{record.get('id')}: expected one tool observation")
        observation = extract_tool_observation(observation_messages[0]["content"])
        if observation != execution.to_dict():
            raise ValueError(f"{record.get('id')}: observation mismatch")
    elif record["category"] == "code":
        ast.parse(assistant_messages[-1]["content"])
    elif record["category"] == "json":
        parsed_json = json.loads(assistant_messages[-1]["content"])
        if not isinstance(parsed_json, dict):
            raise ValueError(f"{record.get('id')}: JSON answer is not an object")
    elif record["category"] == "math":
        if not assistant_messages[-1]["content"].strip():
            raise ValueError(f"{record.get('id')}: empty math answer")
    else:
        raise ValueError(f"{record.get('id')}: unknown category")


def split_stratified(
    records: list[dict[str, Any]],
    validation_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < validation_ratio < 0.5:
        raise ValueError("validation_ratio must be between 0 and 0.5")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["category"]].append(record)
    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    for category in sorted(groups):
        group = list(groups[category])
        rng.shuffle(group)
        valid_count = max(1, round(len(group) * validation_ratio))
        valid.extend(group[:valid_count])
        train.extend(group[valid_count:])
    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-examples", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--train-output", default="data/sft/train.jsonl")
    parser.add_argument("--valid-output", default="data/sft/valid.jsonl")
    parser.add_argument("--manifest-output", default="artifacts/sft_manifest.json")
    parser.add_argument(
        "--eval-dataset",
        action="append",
        default=[
            "data/evaluation/base_smoke.jsonl",
            "data/evaluation/tool_calling_smoke.jsonl",
        ],
    )
    args = parser.parse_args()

    records = build_records(args.num_examples, args.seed)
    registry = build_default_registry()
    parser_impl = ToolCallParser()
    for record in records:
        validate_record(record, registry, parser_impl)

    eval_prompts: set[str] = set()
    eval_hashes: dict[str, str] = {}
    for eval_path in args.eval_dataset:
        eval_hashes[eval_path] = file_sha256(eval_path)
        for record in read_jsonl(eval_path):
            for message in record.get("messages", []):
                if message.get("role") == "user":
                    eval_prompts.add(normalize_text(str(message.get("content", ""))))
    generated_prompts = {
        normalize_text(
            str(message.get("content", ""))
        )
        for record in records
        for message in record["messages"]
        if message.get("role") == "user"
    }
    overlap = sorted(eval_prompts & generated_prompts)
    if overlap:
        raise ValueError(f"Generated SFT data overlaps evaluation prompts: {overlap}")

    train, valid = split_stratified(records, args.validation_ratio, args.seed)
    write_jsonl(args.train_output, train)
    write_jsonl(args.valid_output, valid)
    manifest = {
        "version": "sft_messages_v1",
        "seed": args.seed,
        "num_examples": len(records),
        "train_count": len(train),
        "validation_count": len(valid),
        "category_counts": dict(sorted(Counter(record["category"] for record in records).items())),
        "train_output": args.train_output,
        "validation_output": args.valid_output,
        "evaluation_datasets": eval_hashes,
        "evaluation_prompt_overlap_count": len(overlap),
        "purpose": "training data; never evaluate on these generated records",
    }
    output_path = Path(args.manifest_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
