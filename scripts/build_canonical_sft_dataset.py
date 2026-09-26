from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from llm_posttrain.agent.protocol import ToolCallParser
from llm_posttrain.tools.registry import ToolRegistry, build_default_registry


TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
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


def canonical_tool_call(call: Any) -> str:
    payload = {
        "name": call.name,
        "arguments": call.arguments,
    }
    compact = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"{TOOL_CALL_OPEN}{compact}{TOOL_CALL_CLOSE}"



def tool_system_message(registry: ToolRegistry) -> str:
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

def _tool_call_payload(content: str) -> dict[str, Any]:
    if not (
        content.startswith(TOOL_CALL_OPEN)
        and content.endswith(TOOL_CALL_CLOSE)
    ):
        raise ValueError("tool call is not wrapped in the canonical delimiters")
    payload_text = content[len(TOOL_CALL_OPEN) : -len(TOOL_CALL_CLOSE)]
    payload = json.loads(payload_text)
    if not isinstance(payload, dict):
        raise ValueError("tool call payload must be a JSON object")
    if "type" in payload:
        raise ValueError("canonical tool call must not contain a type field")
    if set(payload) != {"name", "arguments"}:
        raise ValueError("canonical tool call must contain name and arguments")
    if not isinstance(payload["name"], str):
        raise ValueError("tool call name must be a string")
    if not isinstance(payload["arguments"], dict):
        raise ValueError("tool call arguments must be an object")
    return payload


def _user_prompts(record: dict[str, Any]) -> list[str]:
    return [
        normalize(str(message.get("content", "")))
        for message in record.get("messages", [])
        if message.get("role") == "user"
    ]


def _validate_execution(
    record: dict[str, Any],
    call: Any,
    registry: ToolRegistry,
) -> None:
    execution = registry.execute(call.name, call.arguments)
    if not execution.ok:
        raise ValueError(
            f"{record.get('id')}: tool execution failed: {execution.error}"
        )

    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        return
    expected_expression = metadata.get("expression")
    if expected_expression is not None:
        actual_expression = call.arguments.get("expression")
        if actual_expression != expected_expression:
            raise ValueError(
                f"{record.get('id')}: expression metadata disagrees with call"
            )
    expected_answer = metadata.get("answer")
    if expected_answer is not None:
        actual_answer = execution.output.get("result_text")
        if str(actual_answer) != str(expected_answer):
            raise ValueError(
                f"{record.get('id')}: answer metadata disagrees with execution"
            )


def canonicalize_record(
    record: dict[str, Any],
    parser: ToolCallParser,
    registry: ToolRegistry,
) -> dict[str, Any]:
    result = copy.deepcopy(record)
    messages = result.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{record.get('id')}: messages must be a non-empty list")

    found_tool_call = False
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content", ""))
        if TOOL_CALL_OPEN not in content:
            continue
        parsed = parser.parse(content)
        if parsed.kind != "tool_call" or parsed.tool_call is None:
            raise ValueError(f"{record.get('id')}: assistant tool call did not parse")
        message["content"] = canonical_tool_call(parsed.tool_call)
        _validate_execution(result, parsed.tool_call, registry)
        found_tool_call = True

    if found_tool_call:
        system_message = tool_system_message(registry)
        for message in messages:
            if message.get("role") == "system":
                message["content"] = system_message
                break
        else:
            messages.insert(0, {"role": "system", "content": system_message})
        metadata = result.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata["canonical_format"] = True
            metadata["call_payload_has_type_field"] = False
    return result


def validate_records(
    records: list[dict[str, Any]],
    evaluation_paths: list[str | Path],
) -> dict[str, Any]:
    parser = ToolCallParser()
    registry = build_default_registry()
    evaluation_prompts = {
        prompt
        for path in evaluation_paths
        for sample in read_jsonl(path)
        for prompt in _user_prompts(sample)
    }

    ids: set[str] = set()
    prompts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    canonical_tool_records = 0
    evaluation_prompt_overlap_count = 0
    for record in records:
        record_id = str(record.get("id", ""))
        if not record_id:
            raise ValueError("every record must have an id")
        if record_id in ids:
            raise ValueError(f"duplicate record id: {record_id}")
        ids.add(record_id)
        category_counts[str(record.get("category", "unknown"))] += 1

        for prompt in _user_prompts(record):
            prompts[prompt] += 1
            evaluation_prompt_overlap_count += int(prompt in evaluation_prompts)

        has_tool_call = False
        for message in record.get("messages", []):
            if message.get("role") != "assistant":
                continue
            content = str(message.get("content", ""))
            if TOOL_CALL_OPEN not in content:
                continue
            parsed = parser.parse(content)
            if parsed.kind != "tool_call" or parsed.tool_call is None:
                raise ValueError(f"{record_id}: canonical tool call did not parse")
            _tool_call_payload(content)
            _validate_execution(record, parsed.tool_call, registry)
            has_tool_call = True
        canonical_tool_records += int(has_tool_call)

    duplicate_prompt_count = sum(
        count - 1 for count in prompts.values() if count > 1
    )
    return {
        "records": len(records),
        "category_counts": dict(sorted(category_counts.items())),
        "canonical_tool_call_records": canonical_tool_records,
        "unique_user_prompts": len(prompts),
        "duplicate_user_prompt_count": duplicate_prompt_count,
        "evaluation_prompt_overlap_count": evaluation_prompt_overlap_count,
    }


def _load_and_canonicalize(
    path: str | Path,
    parser: ToolCallParser,
    registry: ToolRegistry,
) -> list[dict[str, Any]]:
    return [
        canonicalize_record(record, parser, registry)
        for record in read_jsonl(path)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge existing SFT data with canonical tool-call data."
    )
    parser.add_argument("--base-train", default="data/sft/train.jsonl")
    parser.add_argument("--base-valid", default="data/sft/valid.jsonl")
    parser.add_argument(
        "--protocol-train",
        default="data/protocol_alignment/train.jsonl",
    )
    parser.add_argument(
        "--protocol-valid",
        default="data/protocol_alignment/valid.jsonl",
    )
    parser.add_argument(
        "--train-output",
        default="data/canonical_sft/train.jsonl",
    )
    parser.add_argument(
        "--valid-output",
        default="data/canonical_sft/valid.jsonl",
    )
    parser.add_argument(
        "--manifest-output",
        default="artifacts/canonical_sft_manifest.json",
    )
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--evaluation-dataset", action="append", default=[])
    args = parser.parse_args()

    tool_parser = ToolCallParser()
    registry = build_default_registry()
    base_train = _load_and_canonicalize(args.base_train, tool_parser, registry)
    base_valid = _load_and_canonicalize(args.base_valid, tool_parser, registry)
    protocol_train = _load_and_canonicalize(
        args.protocol_train,
        tool_parser,
        registry,
    )
    protocol_valid = _load_and_canonicalize(
        args.protocol_valid,
        tool_parser,
        registry,
    )

    train_records = base_train + protocol_train
    valid_records = base_valid + protocol_valid
    randomizer = random.Random(args.seed)
    randomizer.shuffle(train_records)
    randomizer.shuffle(valid_records)
    combined = train_records + valid_records

    evaluation_paths = args.evaluation_dataset or [
        "data/evaluation/base_smoke.jsonl",
        "data/evaluation/tool_calling_smoke.jsonl",
        "data/agentic_rl/valid.jsonl",
    ]
    validation = validate_records(combined, evaluation_paths)
    if validation["evaluation_prompt_overlap_count"] != 0:
        raise ValueError("canonical SFT data overlaps with an evaluation prompt")

    write_jsonl(args.train_output, train_records)
    write_jsonl(args.valid_output, valid_records)
    manifest = {
        "version": "canonical_sft_messages_v1",
        "seed": args.seed,
        "sources": {
            "base_train": {
                "path": args.base_train,
                "count": len(base_train),
                "sha256": sha256(args.base_train),
            },
            "base_valid": {
                "path": args.base_valid,
                "count": len(base_valid),
                "sha256": sha256(args.base_valid),
            },
            "protocol_train": {
                "path": args.protocol_train,
                "count": len(protocol_train),
                "sha256": sha256(args.protocol_train),
            },
            "protocol_valid": {
                "path": args.protocol_valid,
                "count": len(protocol_valid),
                "sha256": sha256(args.protocol_valid),
            },
        },
        "train_output": args.train_output,
        "valid_output": args.valid_output,
        "train_count": len(train_records),
        "validation_count": len(valid_records),
        "train_sha256": None,
        "validation_sha256": None,
        "validation": validation,
        "evaluation_datasets": {
            str(path): sha256(path) for path in evaluation_paths
        },
        "contract": (
            "all assistant tool calls use canonical <tool_call> JSON with "
            "name and arguments only; calculator execution is verified"
        ),
    }
    manifest["train_sha256"] = sha256(args.train_output)
    manifest["validation_sha256"] = sha256(args.valid_output)
    output = Path(args.manifest_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
