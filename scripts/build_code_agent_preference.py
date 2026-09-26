from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def rejected_for(record: dict[str, Any]) -> str:
    category = record.get("category")
    expected = record.get("expected", {})
    if category == "code_unit_test":
        return (
            '<tool_call>{"name":"python_executor","arguments":'
            '{"code":"print(0)"}}</tool_call>'
        )
    if category == "python_execution":
        return "I cannot execute Python code."
    if category == "structured_json":
        return '{"priority":"high"}'
    if category == "workspace_file":
        return (
            '<tool_call>{"name":"read_file","arguments":'
            '{"path":"missing.json"}}</tool_call>'
        )
    if category == "mock_search":
        return "The project has no information about this topic."
    if category == "research_to_file":
        return (
            '<tool_call>{"name":"write_file","arguments":'
            '{"path":"finding.md","content":"unsupported fact"}}</tool_call>'
        )
    final_answer = expected.get("final_answer", "")
    return str(final_answer) if final_answer else "invalid answer"


def make_pair(record: dict[str, Any]) -> dict[str, Any]:
    messages = record["messages"]
    prompt = messages[:2]
    assistant_messages = [
        message for message in messages[2:] if message.get("role") == "assistant"
    ]
    if not assistant_messages:
        raise ValueError(f"{record['id']}: no assistant completion")
    chosen = str(assistant_messages[0]["content"])
    rejected = rejected_for(record)
    return {
        "id": f"code_dpo_{record['id']}",
        "category": record["category"],
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "metadata": {
            "source_id": record["id"],
            "chosen_source": "verified_oracle",
            "rejected_source": "rule_negative",
        },
    }


def split_records(
    records: list[dict[str, Any]],
    valid_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid_count = max(1, round(len(records) * valid_ratio))
    valid = records[-valid_count:]
    train = records[:-valid_count]
    if not train:
        raise ValueError("preference train split would be empty")
    return train, valid


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/code_agent/train.jsonl")
    parser.add_argument("--valid-input", default="data/code_agent/valid.jsonl")
    parser.add_argument("--train-output", default="data/code_agent_preference/train.jsonl")
    parser.add_argument("--valid-output", default="data/code_agent_preference/valid.jsonl")
    args = parser.parse_args()

    source = read_jsonl(args.input) + read_jsonl(args.valid_input)
    pairs = [make_pair(record) for record in source]
    train, valid = split_records(pairs, valid_ratio=0.2)
    write_jsonl(args.train_output, train)
    write_jsonl(args.valid_output, valid)
    summary = {
        "source": len(source),
        "pairs": len(pairs),
        "train": len(train),
        "valid": len(valid),
        "categories": sorted({record["category"] for record in pairs}),
        "sha256": {
            "train": sha256(args.train_output),
            "valid": sha256(args.valid_output),
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
