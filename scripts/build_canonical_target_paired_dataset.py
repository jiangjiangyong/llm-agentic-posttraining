from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from llm_posttrain.agent.protocol import ToolCallParser
from llm_posttrain.tools.registry import build_default_registry

try:
    from scripts.data_access_policy import assert_no_final_test_input
except ModuleNotFoundError:
    from data_access_policy import assert_no_final_test_input


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
                raise ValueError(f"Expected object at {path}:{line_number}")
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


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def user_prompts(record: dict[str, Any]) -> list[str]:
    return [
        normalize(str(message.get("content", "")))
        for message in record.get("messages", [])
        if message.get("role") == "user"
    ]


def canonical_tool_call(call: Any) -> str:
    payload = {
        "name": call.name,
        "arguments": call.arguments,
    }
    compact = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"{TOOL_CALL_OPEN}{compact}{TOOL_CALL_CLOSE}"


def _validate_execution(record: dict[str, Any], call: Any) -> None:
    execution = build_default_registry().execute(call.name, call.arguments)
    if not execution.ok:
        raise ValueError(f"{record.get('id')}: source tool call does not execute: {execution.error}")


def pair_record(
    record: dict[str, Any],
    *,
    split: str,
    parser: ToolCallParser | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parser = parser or ToolCallParser()
    source = copy.deepcopy(record)
    result = copy.deepcopy(record)
    messages = result.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{record.get('id')}: messages must be a non-empty list")

    pairs: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        legacy_target = str(message.get("content", ""))
        if TOOL_CALL_OPEN not in legacy_target:
            continue
        parsed = parser.parse(legacy_target)
        if parsed.kind != "tool_call" or parsed.tool_call is None:
            raise ValueError(f"{record.get('id')}: tool target is not parseable")
        _validate_execution(record, parsed.tool_call)
        canonical_target = canonical_tool_call(parsed.tool_call)
        canonical_parsed = parser.parse(canonical_target)
        if canonical_parsed.tool_call != parsed.tool_call:
            raise AssertionError(f"{record.get('id')}: serialization changed tool semantics")
        message["content"] = canonical_target
        pairs.append(
            {
                "message_index": message_index,
                "legacy_target": legacy_target,
                "canonical_target": canonical_target,
                "tool_name": parsed.tool_call.name,
                "arguments": parsed.tool_call.arguments,
                "arguments_sha256": stable_hash(parsed.tool_call.arguments),
                "serialization_changed": legacy_target != canonical_target,
            }
        )

    metadata = result.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"{record.get('id')}: metadata must be an object")
    metadata.update(
        {
            "canonical_target_contract": "paired_serialization_v1",
            "source_split": split,
            "source_record_id": record.get("id"),
            "source_record_sha256": stable_hash(source),
            "target_protocol": "canonical_v1",
            "paired_tool_call_count": len(pairs),
            "serialization_changed_count": sum(
                int(pair["serialization_changed"]) for pair in pairs
            ),
            "paired_targets": pairs,
        }
    )
    audit = {
        "source_id": record.get("id"),
        "source_split": split,
        "source_record_sha256": stable_hash(source),
        "output_record_sha256": stable_hash(result),
        "tool_call_pairs": len(pairs),
        "serialization_changed": sum(int(pair["serialization_changed"]) for pair in pairs),
        "user_prompts": user_prompts(record),
    }
    return result, audit


def build_split(path: str | Path, *, split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assert_no_final_test_input(path, f"canonical-target {split} dataset build")
    parser = ToolCallParser()
    output: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    ids: set[str] = set()
    for record in read_jsonl(path):
        record_id = str(record.get("id", ""))
        if not record_id or record_id in ids:
            raise ValueError(f"{split}: missing or duplicate source id {record_id!r}")
        ids.add(record_id)
        paired, audit = pair_record(record, split=split, parser=parser)
        output.append(paired)
        audits.append(audit)
    return output, audits


def evaluate_overlap(
    records: list[dict[str, Any]],
    evaluation_paths: list[str | Path],
) -> dict[str, Any]:
    evaluation_prompts: set[str] = set()
    for path in evaluation_paths:
        assert_no_final_test_input(path, "canonical-target evaluation overlap audit")
        for record in read_jsonl(path):
            evaluation_prompts.update(user_prompts(record))
    overlap = sum(
        int(prompt in evaluation_prompts)
        for record in records
        for prompt in user_prompts(record)
    )
    return {
        "evaluation_paths": [str(path) for path in evaluation_paths],
        "evaluation_prompt_count": len(evaluation_prompts),
        "evaluation_prompt_overlap_count": overlap,
    }


def cross_split_audit(
    train_audits: list[dict[str, Any]],
    valid_audits: list[dict[str, Any]],
) -> dict[str, Any]:
    train_prompts = {prompt for row in train_audits for prompt in row["user_prompts"]}
    valid_prompts = {prompt for row in valid_audits for prompt in row["user_prompts"]}
    train_ids = {str(row["source_id"]) for row in train_audits}
    valid_ids = {str(row["source_id"]) for row in valid_audits}
    return {
        "source_id_overlap_count": len(train_ids & valid_ids),
        "normalized_user_prompt_overlap_count": len(train_prompts & valid_prompts),
        "train_unique_user_prompts": len(train_prompts),
        "valid_unique_user_prompts": len(valid_prompts),
    }


def drop_valid_prompt_overlaps(
    train_audits: list[dict[str, Any]],
    valid_records: list[dict[str, Any]],
    valid_audits: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    train_prompts = {prompt for row in train_audits for prompt in row["user_prompts"]}
    retained_records: list[dict[str, Any]] = []
    retained_audits: list[dict[str, Any]] = []
    excluded_ids: list[str] = []
    for record, audit in zip(valid_records, valid_audits):
        if train_prompts.intersection(audit["user_prompts"]):
            excluded_ids.append(str(audit["source_id"]))
            continue
        retained_records.append(record)
        retained_audits.append(audit)
    return retained_records, retained_audits, excluded_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a paired canonical-target SFT dataset.")
    parser.add_argument("--source-train", default="data/sft/train.jsonl")
    parser.add_argument("--source-valid", default="data/sft/valid.jsonl")
    parser.add_argument("--train-output", default="data/canonical_target_paired/train.jsonl")
    parser.add_argument("--valid-output", default="data/canonical_target_paired/valid.jsonl")
    parser.add_argument("--manifest-output", default="artifacts/canonical_target_paired_manifest.json")
    parser.add_argument(
        "--drop-overlapping-valid-records",
        action="store_true",
        help="Explicitly exclude valid records whose normalized prompt occurs in train.",
    )
    parser.add_argument(
        "--evaluation-path",
        action="append",
        default=[
            "data/evaluation/base_smoke.jsonl",
            "data/evaluation/tool_calling_smoke.jsonl",
            "data/agentic_rl/valid.jsonl",
        ],
    )
    args = parser.parse_args()

    train_records, train_audits = build_split(args.source_train, split="train")
    valid_records, valid_audits = build_split(args.source_valid, split="valid")
    original_valid_count = len(valid_records)
    excluded_valid_ids: list[str] = []
    if args.drop_overlapping_valid_records:
        valid_records, valid_audits, excluded_valid_ids = drop_valid_prompt_overlaps(
            train_audits,
            valid_records,
            valid_audits,
        )
    all_records = train_records + valid_records
    overlap = evaluate_overlap(all_records, args.evaluation_path)
    cross_split = cross_split_audit(train_audits, valid_audits)
    if overlap["evaluation_prompt_overlap_count"] != 0:
        raise ValueError("canonical-target source overlaps with evaluation prompts")
    if cross_split["source_id_overlap_count"] != 0:
        raise ValueError("canonical-target source id crosses train/valid split")
    if cross_split["normalized_user_prompt_overlap_count"] != 0:
        raise ValueError("canonical-target normalized user prompt crosses train/valid split")

    write_jsonl(args.train_output, train_records)
    write_jsonl(args.valid_output, valid_records)
    manifest = {
        "version": "canonical_target_paired_v1",
        "source": {
            "train": {"path": args.source_train, "count": len(train_records), "sha256": sha256(args.source_train)},
            "valid": {"path": args.source_valid, "count": original_valid_count, "sha256": sha256(args.source_valid)},
        },
        "outputs": {
            "train": {"path": args.train_output, "count": len(train_records), "sha256": sha256(args.train_output)},
            "valid": {"path": args.valid_output, "count": len(valid_records), "sha256": sha256(args.valid_output)},
        },
        "source_split_audit": {
            "original_valid_count": original_valid_count,
            "retained_valid_count": len(valid_records),
            "excluded_valid_overlap_ids": excluded_valid_ids,
            "overlap_exclusion_requested": bool(args.drop_overlapping_valid_records),
        },
        "serialization": {
            "total_tool_call_pairs": sum(row["tool_call_pairs"] for row in train_audits + valid_audits),
            "changed_tool_call_pairs": sum(row["serialization_changed"] for row in train_audits + valid_audits),
            "records_with_tool_call": sum(bool(row["tool_call_pairs"]) for row in train_audits + valid_audits),
            "records_without_tool_call_unchanged": sum(not row["tool_call_pairs"] for row in train_audits + valid_audits),
            "non_tool_messages_unchanged": True,
            "tool_name_and_arguments_unchanged": True,
            "observation_and_final_answer_unchanged": True,
        },
        "guards": {
            **overlap,
            **cross_split,
            "frozen_final_test_accessed": False,
            "failure_mining_dev_used": False,
            "validation_used_for_training": False,
        },
        "category_counts": dict(Counter(str(record.get("category", "unknown")) for record in all_records)),
        "contract": "same source task and semantic action; only assistant tool-call serialization changes to canonical_v1",
    }
    manifest_path = Path(args.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
