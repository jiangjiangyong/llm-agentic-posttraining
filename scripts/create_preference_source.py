from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from llm_posttrain.agent.protocol import ToolCallParser
from llm_posttrain.tools.registry import ToolRegistry, build_default_registry


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

_JSON_TASKS = [
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
    ("The mobile app shows a blank screen after an update", "mobile", "high"),
]


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


def make_expression(index: int) -> str:
    left = 101 + index * 3
    right = 7 + (index % 13)
    third = 2 + (index % 9)
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


def make_tool_record(index: int, expression: str, registry: ToolRegistry) -> dict[str, Any]:
    execution = registry.execute("calculator", {"expression": expression})
    if not execution.ok:
        raise ValueError(f"Unable to evaluate source expression {expression}: {execution.error}")
    call = {
        "type": "tool_call",
        "name": "calculator",
        "arguments": {"expression": expression},
    }
    user_variants = (
        f"Please use the calculator tool to evaluate {expression} and return the result.",
        f"Call the available arithmetic tool for {expression}; answer after the tool runs.",
        f"Compute {expression} with the calculator and give the final numeric result.",
    )
    return {
        "id": f"pref_tool_{index:04d}",
        "category": "tool_calling",
        "messages": [
            {"role": "system", "content": tool_system_message(registry)},
            {"role": "user", "content": user_variants[index % len(user_variants)]},
        ],
        "gold": {
            "response": f"<tool_call>{json.dumps(call, ensure_ascii=False, separators=(',', ':'))}</tool_call>",
            "tool_call": {"name": "calculator", "arguments": {"expression": expression}},
            "final_answer": execution.output["result_text"],
        },
        "metadata": {"task_type": "calculator", "expression": expression},
    }


def make_code_record(index: int, task: tuple[str, str, str]) -> dict[str, Any]:
    name, signature, body = task
    imports = "import math\n\n" if name == "product_values" else ""
    code = f"{imports}def {signature}:\n    {body}"
    return {
        "id": f"pref_code_{index:04d}",
        "category": "code",
        "messages": [
            {"role": "system", "content": "Return valid Python code only. Do not use Markdown fences."},
            {"role": "user", "content": f"Implement the Python function {signature}."},
        ],
        "gold": {"response": code, "expected": {"required_symbols": [name]}},
        "metadata": {"task_type": "python_function", "required_symbols": [name]},
    }


def make_json_record(index: int, task: tuple[str, str, str]) -> dict[str, Any]:
    description, category, priority = task
    answer = {"category": category, "priority": priority}
    return {
        "id": f"pref_json_{index:04d}",
        "category": "json",
        "messages": [
            {"role": "system", "content": "Return valid JSON only with exactly category and priority fields."},
            {
                "role": "user",
                "content": f"Classify this support ticket. Ticket: {description}",
            },
        ],
        "gold": {"response": json.dumps(answer, ensure_ascii=False, separators=(",", ":")), "expected": answer},
        "metadata": {"task_type": "structured_classification", "answer": answer},
    }


def make_math_record(index: int, expression: str, registry: ToolRegistry) -> dict[str, Any]:
    execution = registry.execute("calculator", {"expression": expression})
    if not execution.ok:
        raise ValueError(f"Unable to evaluate source expression {expression}: {execution.error}")
    return {
        "id": f"pref_math_{index:04d}",
        "category": "math",
        "messages": [
            {"role": "system", "content": "You are a precise math assistant. Output only the final number."},
            {"role": "user", "content": f"Calculate {expression}."},
        ],
        "gold": {"response": execution.output["result_text"], "expected": execution.output["result_text"]},
        "metadata": {"task_type": "arithmetic", "expression": expression},
    }


def build_records(num_examples: int, seed: int) -> list[dict[str, Any]]:
    if num_examples < 16:
        raise ValueError("num_examples must be at least 16")
    registry = build_default_registry()
    tool_count = max(1, int(num_examples * 0.50))
    code_count = max(1, int(num_examples * 0.1875))
    json_count = max(1, int(num_examples * 0.1875))
    math_count = num_examples - tool_count - code_count - json_count
    if math_count < 1:
        math_count = 1
        tool_count = num_examples - code_count - json_count - math_count
    records: list[dict[str, Any]] = []
    for index in range(tool_count):
        records.append(make_tool_record(index, make_expression(200 + index), registry))
    for index in range(code_count):
        records.append(make_code_record(index, _CODE_TASKS[index % len(_CODE_TASKS)]))
    for index in range(json_count):
        records.append(make_json_record(index, _JSON_TASKS[index % len(_JSON_TASKS)]))
    for index in range(math_count):
        records.append(make_math_record(index, make_expression(500 + index), registry))
    random.Random(seed).shuffle(records)
    return records


def validate_source(records: list[dict[str, Any]], registry: ToolRegistry) -> None:
    parser = ToolCallParser()
    seen_ids: set[str] = set()
    for record in records:
        record_id = str(record.get("id", ""))
        if not record_id or record_id in seen_ids:
            raise ValueError(f"Duplicate or empty source id: {record_id}")
        seen_ids.add(record_id)
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{record_id}: messages must be a non-empty list")
        response = str(record.get("gold", {}).get("response", ""))
        if not response.strip():
            raise ValueError(f"{record_id}: empty gold response")
        category = record.get("category")
        if category == "tool_calling":
            parsed = parser.parse(response)
            call = parsed.tool_call
            if parsed.kind != "tool_call" or call is None:
                raise ValueError(f"{record_id}: gold tool call did not parse")
            result = registry.execute(call.name, call.arguments)
            if not result.ok:
                raise ValueError(f"{record_id}: gold tool call failed: {result.error}")
        elif category == "code":
            ast.parse(response)
        elif category == "json":
            value = json.loads(response)
            if not isinstance(value, dict):
                raise ValueError(f"{record_id}: gold JSON is not an object")
        elif category == "math":
            if not response.strip():
                raise ValueError(f"{record_id}: empty math answer")
        else:
            raise ValueError(f"{record_id}: unknown category {category}")


def check_evaluation_overlap(records: list[dict[str, Any]], eval_paths: list[str]) -> dict[str, str]:
    eval_prompts: set[str] = set()
    hashes: dict[str, str] = {}
    for eval_path in eval_paths:
        hashes[eval_path] = file_sha256(eval_path)
        for record in read_jsonl(eval_path):
            for message in record.get("messages", []):
                if message.get("role") == "user":
                    eval_prompts.add(normalize_text(str(message.get("content", ""))))
    source_prompts = {
        normalize_text(str(message.get("content", "")))
        for record in records
        for message in record.get("messages", [])
        if message.get("role") == "user"
    }
    overlap = sorted(eval_prompts & source_prompts)
    if overlap:
        raise ValueError(f"Preference source overlaps frozen evaluation prompts: {overlap}")
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description="Create isolated preference source tasks.")
    parser.add_argument("--num-examples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--output", default="data/preference/source.jsonl")
    parser.add_argument("--manifest-output", default="artifacts/preference_source_manifest.json")
    parser.add_argument(
        "--eval-dataset",
        action="append",
        default=["data/evaluation/base_smoke.jsonl", "data/evaluation/tool_calling_smoke.jsonl"],
    )
    args = parser.parse_args()
    records = build_records(args.num_examples, args.seed)
    validate_source(records, build_default_registry())
    eval_hashes = check_evaluation_overlap(records, args.eval_dataset)
    write_jsonl(args.output, records)
    manifest = {
        "version": "preference_source_v1",
        "seed": args.seed,
        "num_examples": len(records),
        "category_counts": dict(sorted(Counter(record["category"] for record in records).items())),
        "source_output": args.output,
        "evaluation_datasets": eval_hashes,
        "evaluation_prompt_overlap_count": 0,
        "purpose": "isolated prompts used to create chosen/rejected preference pairs",
    }
    output_path = Path(args.manifest_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
