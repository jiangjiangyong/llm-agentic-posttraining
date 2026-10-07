from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from llm_posttrain.agent.environment import CalculatorEnvironment
from llm_posttrain.agent.protocol import ToolCallParser
from llm_posttrain.agent.protocol_normalizer import CanonicalProtocolNormalizer
from llm_posttrain.agent.runtime import AgentRuntime
from llm_posttrain.config import load_yaml
from llm_posttrain.models.adapter_runner import build_adapter_runner
from llm_posttrain.models.loader import render_chat_prompt
from llm_posttrain.tools.registry import build_default_registry


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
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


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def first_diff(left: list[int], right: list[int]) -> int | None:
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def initial_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for message in record["messages"]:
        if message.get("role") == "assistant":
            break
        messages.append(copy.deepcopy(message))
    return messages


def prompt_alignment_audit(
    tokenizer: Any,
    records: list[dict[str, Any]],
    runtime: AgentRuntime,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for record in records[:3]:
        training_messages = initial_messages(record)
        runtime_messages = runtime._prepare_messages(training_messages)
        training_prompt = render_chat_prompt(tokenizer, training_messages)
        runtime_prompt = render_chat_prompt(tokenizer, runtime_messages)
        training_ids = tokenizer(training_prompt, add_special_tokens=False)["input_ids"]
        runtime_ids = tokenizer(runtime_prompt, add_special_tokens=False)["input_ids"]
        rows.append(
            {
                "record_id": record["id"],
                "training_prompt_tokens": len(training_ids),
                "runtime_prompt_tokens": len(runtime_ids),
                "first_diff": first_diff(training_ids, runtime_ids),
                "text_equal": training_prompt == runtime_prompt,
                "token_sequence_equal": training_ids == runtime_ids,
                "system_count": sum(
                    message.get("role") == "system" for message in runtime_messages
                ),
                "protocol_instruction_count": runtime_messages[0]["content"].count(
                    "You are running inside Tool Calling Protocol v1."
                ),
            }
        )
    return {
        "records_sampled": len(rows),
        "all_text_equal": all(row["text_equal"] for row in rows),
        "all_token_sequences_equal": all(
            row["token_sequence_equal"] for row in rows
        ),
        "all_single_system": all(row["system_count"] == 1 for row in rows),
        "all_single_protocol_instruction": all(
            row["protocol_instruction_count"] == 1 for row in rows
        ),
        "rows": rows,
    }


def smoke_metrics(
    rows: list[dict[str, Any]],
    *,
    arm: str,
) -> dict[str, Any]:
    parser_rows = [row for row in rows if row["tool_call_present"]]
    final_reached = sum(row["final_output"] is not None for row in rows)
    return {
        "arm": arm,
        "episodes": len(rows),
        "raw_canonical": sum(row["raw_canonical"] for row in rows),
        "parsed_tool_calls": len(parser_rows),
        "tool_selection_correct": sum(row["tool_selection_correct"] for row in rows),
        "argument_correct": sum(row["arguments_correct"] for row in rows),
        "execution_success": sum(row["execution_success"] for row in rows),
        "final_exact": sum(row["final_exact"] for row in rows),
        "final_reached": final_reached,
        "semantic_success": sum(row["semantic_success"] for row in rows),
        "raw_strict_success": sum(row["strict_success"] for row in rows),
        "parser_recovered_noncanonical": sum(
            row["tool_call_present"] and not row["raw_canonical"] for row in rows
        ),
        "rates": {
            "parsed_tool_call": ratio(len(parser_rows), len(rows)),
            "semantic": ratio(
                sum(row["semantic_success"] for row in rows), len(rows)
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate runtime prompt alignment without training."
    )
    parser.add_argument("--model-config", default="configs/base_model.yaml")
    parser.add_argument(
        "--smoke-input",
        default="data/evaluation/tool_calling_smoke.jsonl",
    )
    parser.add_argument(
        "--alignment-input",
        default="data/canonical_target_paired/train.jsonl",
    )
    parser.add_argument(
        "--adapter",
        default="models/adapters/sft_qwen3_1.7b",
    )
    parser.add_argument(
        "--predictions-output",
        default="outputs/engineering_runtime_alignment/calculator_smoke.jsonl",
    )
    parser.add_argument(
        "--artifact-output",
        default="artifacts/engineering_runtime_alignment_regression_v1.json",
    )
    args = parser.parse_args()

    smoke_records = read_jsonl(args.smoke_input)
    if len(smoke_records) != 6:
        raise ValueError(f"Expected 6 calculator smoke records, got {len(smoke_records)}")
    alignment_records = read_jsonl(args.alignment_input)
    model_config = load_yaml(args.model_config)
    runner = build_adapter_runner(
        model_config,
        args.adapter,
        max_new_tokens=128,
        do_sample=False,
    )
    registry = build_default_registry()
    runtime = AgentRuntime(backend=runner, registry=registry, max_steps=3)
    environment = CalculatorEnvironment(registry)
    parser_impl = ToolCallParser()
    rows: list[dict[str, Any]] = []
    for sample in smoke_records:
        prepared = copy.deepcopy(sample)
        prepared["messages"] = runtime._prepare_messages(sample["messages"])
        result = environment.run(
            prepared,
            runner.generate,
            task_id=f"engineering-runtime-alignment:{sample['id']}",
        )
        first_output = result.steps[0]["model_output"]
        parsed = parser_impl.parse(first_output)
        call = parsed.tool_call
        expected = sample["expected"]
        rows.append(
            {
                "id": sample["id"],
                "raw_first_action": first_output,
                "raw_canonical": CanonicalProtocolNormalizer.is_canonical(first_output),
                "parsed_kind": parsed.kind,
                "tool_call_present": call is not None,
                "tool_selection_correct": call is not None and call.name == expected["tool_name"],
                "arguments_correct": call is not None and call.arguments == expected["arguments"],
                "execution_success": bool(
                    result.execution and result.execution.get("ok") is True
                ),
                "final_output": result.final_output,
                "final_exact": (
                    result.final_output is not None
                    and result.final_output.strip() == str(expected["final_answer"]).strip()
                ),
                "semantic_success": bool(result.semantic_success),
                "strict_success": bool(result.strict_success),
                "prompt_system_count": sum(
                    message.get("role") == "system"
                    for message in prepared["messages"]
                ),
                "protocol_instruction_count": prepared["messages"][0]["content"].count(
                    "You are running inside Tool Calling Protocol v1."
                ),
            }
        )

    historical = None
    historical_path = Path("artifacts/experiment_4_canonical_target_smoke_v1.json")
    if historical_path.exists():
        historical = json.loads(historical_path.read_text(encoding="utf-8")).get("C0")
    alignment = prompt_alignment_audit(runner.tokenizer, alignment_records, runtime)
    artifact = {
        "experiment": "engineering_runtime_prompt_alignment_regression",
        "scope": {
            "training": False,
            "reward_changed": False,
            "evaluator_changed": False,
            "data_changed": False,
            "rq1_adapter_changed": False,
            "dev_evaluated": False,
            "validation_evaluated": False,
            "frozen_final_test_accessed": False,
        },
        "runtime_fix": {
            "behavior": "preserve an existing Tool Calling Protocol system; inject the canonical instruction once only when absent",
            "system_prompt_alignment_checked": True,
        },
        "prompt_alignment": alignment,
        "smoke_input": {
            "path": args.smoke_input,
            "sha256": sha256(args.smoke_input),
            "episodes": len(smoke_records),
        },
        "adapter": args.adapter,
        "after_fix": smoke_metrics(rows, arm="stable_sft_after_runtime_alignment"),
        "historical_before_fix": historical,
        "predictions_path": args.predictions_output,
    }
    write_jsonl(args.predictions_output, rows)
    output = Path(args.artifact_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
