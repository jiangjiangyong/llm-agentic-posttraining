from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from llm_posttrain.agent.protocol import ToolCallParser
from llm_posttrain.tools.registry import ToolRegistry, build_default_registry


TOOL_FAILURES = {
    "tool_parse_failure",
    "canonical_format_failure",
    "tool_selection_failure",
    "argument_failure",
    "execution_failure",
}


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


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def task_id_for(row: dict[str, Any]) -> str:
    raw = str(row.get("task_id") or row.get("id") or "anonymous")
    return raw.split(":group_", 1)[0].split(":rollout_", 1)[0]


def failure_types(row: dict[str, Any]) -> list[str]:
    reward = row.get("reward", {})
    failures: list[str] = []
    if float(reward.get("format_reward", 0.0)) < 1.0:
        failures.append("tool_parse_failure")
    elif float(reward.get("canonical_format_reward", 0.0)) < 1.0:
        failures.append("canonical_format_failure")
    if float(reward.get("tool_selection_reward", 0.0)) < 1.0:
        failures.append("tool_selection_failure")
    if float(reward.get("argument_reward", 0.0)) < 1.0:
        failures.append("argument_failure")
    if float(reward.get("execution_reward", 0.0)) < 1.0:
        failures.append("execution_failure")
    if float(reward.get("final_answer_reward", 0.0)) < 1.0:
        failures.append("final_answer_failure")
    return failures


def primary_failure(row: dict[str, Any]) -> str | None:
    failures = failure_types(row)
    if not failures:
        return None
    return failures[0]


def canonical_tool_call(expected: dict[str, Any]) -> str:
    payload = {
        "type": "tool_call",
        "name": expected["tool_name"],
        "arguments": expected["arguments"],
    }
    return f"<tool_call>{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}</tool_call>"


def oracle_messages(
    row: dict[str, Any],
    registry: ToolRegistry,
) -> tuple[list[dict[str, Any]], str, dict[str, Any], str]:
    initial = row.get("initial_messages")
    expected = row.get("expected")
    if not isinstance(initial, list) or not initial:
        raise ValueError(f"{row.get('id')}: initial_messages must be non-empty")
    if not isinstance(expected, dict):
        raise ValueError(f"{row.get('id')}: expected must be an object")
    tool_name = str(expected["tool_name"])
    arguments = dict(expected["arguments"])
    execution = registry.execute(tool_name, arguments)
    if not execution.ok:
        raise ValueError(f"{row.get('id')}: oracle execution failed: {execution.error}")
    call = canonical_tool_call(expected)
    observation = (
        "<tool_observation>\n"
        + json.dumps(execution.to_dict(), ensure_ascii=False)
        + "\n</tool_observation>"
    )
    return (
        list(initial),
        call,
        {"role": "tool", "name": tool_name, "content": observation},
        str(expected["final_answer"]),
    )


def build_sft_records(
    failed_by_task: dict[str, list[dict[str, Any]]],
    registry: ToolRegistry,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task_id, rows in sorted(failed_by_task.items()):
        representative = max(
            rows,
            key=lambda row: float(row.get("reward", {}).get("total_reward", 0.0)),
        )
        initial, call, observation, final_answer = oracle_messages(
            representative, registry
        )
        failure_counter = Counter(
            failure for row in rows for failure in failure_types(row)
        )
        messages = initial + [
            {"role": "assistant", "content": call},
            observation,
            {"role": "assistant", "content": final_answer},
        ]
        records.append(
            {
                "id": f"flywheel_sft_{task_id}",
                "category": "tool_calling",
                "messages": messages,
                "metadata": {
                    "task_type": "failure_flywheel_repair",
                    "task_id": task_id,
                    "source_episode_ids": [
                        str(row.get("episode_id", row.get("id", "")))
                        for row in rows
                    ],
                    "failure_type_counts": dict(sorted(failure_counter.items())),
                    "source_rollout_count": len(rows),
                },
            }
        )
    return records


def build_dpo_records(
    failed_by_task: dict[str, list[dict[str, Any]]],
    registry: ToolRegistry,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task_id, rows in sorted(failed_by_task.items()):
        oracle_row = rows[0]
        initial, call, observation, final_answer = oracle_messages(oracle_row, registry)
        tool_rows = [
            row
            for row in rows
            if primary_failure(row) in TOOL_FAILURES
            and str(row.get("tool_output", "")).strip()
        ]
        if tool_rows:
            row = max(
                tool_rows,
                key=lambda item: float(item.get("reward", {}).get("total_reward", 0.0)),
            )
            rejected = str(row["tool_output"]).strip()
            if rejected != call:
                records.append(
                    {
                        "id": f"flywheel_dpo_{task_id}_tool",
                        "category": "tool_calling",
                        "prompt": initial,
                        "chosen": call,
                        "rejected": rejected,
                        "metadata": {
                            "task_id": task_id,
                            "stage": "tool_call",
                            "failure_types": failure_types(row),
                            "source_episode_id": row.get("episode_id", row.get("id")),
                            "source_reward": row.get("reward", {}),
                        },
                    }
                )
        final_rows = [
            row
            for row in rows
            if "final_answer_failure" in failure_types(row)
            and float(row.get("reward", {}).get("format_reward", 0.0)) >= 1.0
            and float(row.get("reward", {}).get("tool_selection_reward", 0.0)) >= 1.0
            and float(row.get("reward", {}).get("argument_reward", 0.0)) >= 1.0
            and float(row.get("reward", {}).get("execution_reward", 0.0)) >= 1.0
            and str(row.get("final_output") or "").strip()
        ]
        if final_rows:
            row = max(
                final_rows,
                key=lambda item: float(item.get("reward", {}).get("total_reward", 0.0)),
            )
            rejected = str(row["final_output"]).strip()
            final_prompt = initial + [
                {"role": "assistant", "content": call},
                observation,
            ]
            if rejected != final_answer:
                records.append(
                    {
                        "id": f"flywheel_dpo_{task_id}_final",
                        "category": "tool_calling",
                        "prompt": final_prompt,
                        "chosen": final_answer,
                        "rejected": rejected,
                        "metadata": {
                            "task_id": task_id,
                            "stage": "final_answer",
                            "failure_types": failure_types(row),
                            "source_episode_id": row.get("episode_id", row.get("id")),
                            "source_reward": row.get("reward", {}),
                        },
                    }
                )
    return records


def validate_sft_record(
    record: dict[str, Any],
    registry: ToolRegistry,
    parser: ToolCallParser,
) -> None:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{record.get('id')}: messages must be non-empty")
    calls = [
        message
        for message in messages
        if message.get("role") == "assistant"
        and "<tool_call>" in str(message.get("content", ""))
    ]
    observations = [message for message in messages if message.get("role") == "tool"]
    if len(calls) != 1 or len(observations) != 1:
        raise ValueError(f"{record.get('id')}: expected one tool call and observation")
    parsed = parser.parse(str(calls[0]["content"]))
    if parsed.kind != "tool_call" or parsed.tool_call is None:
        raise ValueError(f"{record.get('id')}: canonical call did not parse")
    execution = registry.execute(parsed.tool_call.name, parsed.tool_call.arguments)
    if not execution.ok:
        raise ValueError(f"{record.get('id')}: generated call failed")
    match = re.fullmatch(
        r"<tool_observation>\s*(.*?)\s*</tool_observation>",
        str(observations[0].get("content", "")).strip(),
        flags=re.DOTALL,
    )
    if not match or json.loads(match.group(1)) != execution.to_dict():
        raise ValueError(f"{record.get('id')}: observation mismatch")
    final_messages = [message for message in messages if message.get("role") == "assistant"]
    if len(final_messages) != 2 or not str(final_messages[-1].get("content", "")).strip():
        raise ValueError(f"{record.get('id')}: final answer missing")


def validate_dpo_record(record: dict[str, Any], parser: ToolCallParser) -> None:
    prompt = record.get("prompt")
    chosen = str(record.get("chosen", "")).strip()
    rejected = str(record.get("rejected", "")).strip()
    if not isinstance(prompt, list) or not prompt or not chosen or not rejected:
        raise ValueError(f"{record.get('id')}: malformed DPO pair")
    if chosen == rejected:
        raise ValueError(f"{record.get('id')}: chosen and rejected are identical")
    stage = str(record.get("metadata", {}).get("stage", ""))
    if stage == "tool_call":
        parsed = parser.parse(chosen)
        if parsed.kind != "tool_call" or parsed.tool_call is None:
            raise ValueError(f"{record.get('id')}: chosen tool call is invalid")


def split_grouped(
    records: list[dict[str, Any]],
    validation_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < validation_ratio < 0.5:
        raise ValueError("validation ratio must be between 0 and 0.5")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        task_id = str(record.get("metadata", {}).get("task_id", record["id"]))
        groups[task_id].append(record)
    if len(groups) < 2:
        raise ValueError("at least two task groups are required for train/valid split")
    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)
    valid_count = max(1, round(len(group_ids) * validation_ratio))
    valid_ids = set(group_ids[:valid_count])
    train = [record for task_id, items in groups.items() if task_id not in valid_ids for record in items]
    valid = [record for task_id, items in groups.items() if task_id in valid_ids for record in items]
    random.Random(seed + 1).shuffle(train)
    random.Random(seed + 1).shuffle(valid)
    return train, valid


def evaluation_hashes_and_overlap(
    rows: list[dict[str, Any]],
    eval_paths: list[str],
) -> tuple[dict[str, str], list[str]]:
    eval_prompts: set[str] = set()
    hashes: dict[str, str] = {}
    for path in eval_paths:
        hashes[path] = file_sha256(path)
        for record in read_jsonl(path):
            for message in record.get("messages", []):
                if message.get("role") == "user":
                    eval_prompts.add(normalize_text(str(message.get("content", ""))))
    source_prompts = {
        normalize_text(str(message.get("content", "")))
        for row in rows
        for message in row.get("initial_messages", [])
        if message.get("role") == "user"
    }
    return hashes, sorted(eval_prompts & source_prompts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SFT and DPO repair data from failed training rollouts."
    )
    parser.add_argument("--source", default="outputs/grpo/rollouts_epoch_1.jsonl")
    parser.add_argument("--task-data", default="data/agentic_rl/train.jsonl")
    parser.add_argument("--sft-train-output", default="data/failure_flywheel/sft_train.jsonl")
    parser.add_argument("--sft-valid-output", default="data/failure_flywheel/sft_valid.jsonl")
    parser.add_argument("--dpo-train-output", default="data/failure_flywheel/dpo_train.jsonl")
    parser.add_argument("--dpo-valid-output", default="data/failure_flywheel/dpo_valid.jsonl")
    parser.add_argument("--manifest-output", default="artifacts/failure_flywheel_manifest.json")
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--eval-dataset", action="append", default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.source)
    if not rows:
        raise ValueError(f"No rollout rows found in {args.source}")
    eval_paths = args.eval_dataset or [
        "data/evaluation/base_smoke.jsonl",
        "data/evaluation/tool_calling_smoke.jsonl",
        "data/agentic_rl/valid.jsonl",
    ]
    eval_hashes, overlap = evaluation_hashes_and_overlap(rows, eval_paths)
    if overlap:
        raise ValueError(f"Failure source overlaps evaluation prompts: {overlap}")

    task_records = {
        str(record["id"]): record for record in read_jsonl(args.task_data)
    }
    if not task_records:
        raise ValueError(f"No task records found in {args.task_data}")
    failed_rows: list[dict[str, Any]] = []
    for row in rows:
        if not failure_types(row):
            continue
        task_id = task_id_for(row)
        task = task_records.get(task_id)
        if task is None:
            raise ValueError(f"No task-data record found for rollout task {task_id}")
        enriched = dict(row)
        enriched["expected"] = task["expected"]
        if not enriched.get("initial_messages"):
            enriched["initial_messages"] = task["messages"]
        failed_rows.append(enriched)
    if not failed_rows:
        raise ValueError("No failed rollout rows were found")
    failed_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in failed_rows:
        failed_by_task[task_id_for(row)].append(row)

    registry = build_default_registry()
    parser_impl = ToolCallParser()
    sft_records = build_sft_records(failed_by_task, registry)
    dpo_records = build_dpo_records(failed_by_task, registry)
    for record in sft_records:
        validate_sft_record(record, registry, parser_impl)
    for record in dpo_records:
        validate_dpo_record(record, parser_impl)
    if len(sft_records) < 2 or len(dpo_records) < 2:
        raise ValueError("Failure flywheel needs at least two SFT and DPO task records")

    sft_train, sft_valid = split_grouped(sft_records, args.validation_ratio, args.seed)
    dpo_train, dpo_valid = split_grouped(dpo_records, args.validation_ratio, args.seed)
    write_jsonl(args.sft_train_output, sft_train)
    write_jsonl(args.sft_valid_output, sft_valid)
    write_jsonl(args.dpo_train_output, dpo_train)
    write_jsonl(args.dpo_valid_output, dpo_valid)
    failure_counts = Counter(
        failure for row in failed_rows for failure in failure_types(row)
    )
    primary_counts = Counter(primary_failure(row) for row in failed_rows)
    manifest = {
        "version": "failure_flywheel_v1",
        "source": args.source,
        "source_sha256": file_sha256(args.source),
        "task_data": args.task_data,
        "task_data_sha256": file_sha256(args.task_data),
        "source_rollout_count": len(rows),
        "failed_rollout_count": len(failed_rows),
        "failed_task_count": len(failed_by_task),
        "sft_count": len(sft_records),
        "sft_train_count": len(sft_train),
        "sft_validation_count": len(sft_valid),
        "dpo_count": len(dpo_records),
        "dpo_train_count": len(dpo_train),
        "dpo_validation_count": len(dpo_valid),
        "failure_type_counts": dict(sorted(failure_counts.items())),
        "primary_failure_counts": {
            str(key): value for key, value in sorted(primary_counts.items())
        },
        "evaluation_datasets": eval_hashes,
        "evaluation_prompt_overlap_count": len(overlap),
        "sft_outputs": {
            "train": args.sft_train_output,
            "valid": args.sft_valid_output,
        },
        "dpo_outputs": {
            "train": args.dpo_train_output,
            "valid": args.dpo_valid_output,
        },
        "contract": "failed training rollouts only; task-group split; validated oracle SFT and DPO pairs",
    }
    write_json(args.manifest_output, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
