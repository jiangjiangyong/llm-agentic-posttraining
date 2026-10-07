from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import LogitsProcessorList

from llm_posttrain.agent.environment import CalculatorEnvironment
from llm_posttrain.agent.protocol import ToolCallParser
from llm_posttrain.agent.protocol_constraint import (
    CanonicalCalculatorProtocolLogitsProcessor,
)
from llm_posttrain.agent.protocol_normalizer import CanonicalProtocolNormalizer
from llm_posttrain.config import load_yaml
from llm_posttrain.models.adapter_runner import load_quantized_adapter
from llm_posttrain.models.loader import decode_generation, model_input_device, render_chat_prompt
from llm_posttrain.tools.registry import build_default_registry


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at line {line_number}: {path}")
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


class ConstrainedFirstActionRunner:
    """Use constrained decoding only before the first tool observation."""

    def __init__(
        self,
        base_model_path: str | Path,
        adapter_path: str | Path,
        *,
        tool_name: str = "calculator",
        max_new_tokens: int,
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.tool_name = tool_name
        self.tokenizer, self.model = load_quantized_adapter(
            base_model_path,
            adapter_path,
        )
        self.outputs: list[dict[str, Any]] = []

    def generate(self, messages: list[dict[str, Any]]) -> str:
        constrained = not any(message.get("role") == "tool" for message in messages)
        prompt = render_chat_prompt(self.tokenizer, messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = model_input_device(self.model)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        prompt_length = int(inputs["input_ids"].shape[1])
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if constrained:
            processor = CanonicalCalculatorProtocolLogitsProcessor(
                self.tokenizer,
                [prompt_length],
                tool_name=self.tool_name,
            )
            generation_kwargs["logits_processor"] = LogitsProcessorList([processor])

        started = time.perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_kwargs)
        elapsed_ms = (time.perf_counter() - started) * 1000
        new_tokens = output_ids[:, prompt_length:]
        text = decode_generation(self.tokenizer, new_tokens)
        self.outputs.append(
            {
                "constrained": constrained,
                "text": text,
                "completion_tokens": int(new_tokens.shape[1]),
                "latency_ms": elapsed_ms,
            }
        )
        return text


def control_summary(path: str | Path) -> dict[str, Any]:
    records = read_jsonl(path)
    raw_outputs: list[str] = []
    for record in records:
        trace = record.get("trace", {})
        for event in trace.get("events", []):
            if event.get("kind") == "model_output":
                raw_outputs.append(str(event.get("payload", {}).get("text", "")))
                break
    parser = ToolCallParser()
    raw_canonical = sum(CanonicalProtocolNormalizer.is_canonical(text) for text in raw_outputs)
    tool_present = sum(bool(record.get("tool_call_present")) for record in records)
    return {
        "input_path": str(path),
        "input_sha256": sha256(path),
        "episodes": len(records),
        "raw_first_action_outputs": len(raw_outputs),
        "raw_canonical_first_action": raw_canonical,
        "raw_canonical_rate": ratio(raw_canonical, len(raw_outputs)),
        "semantic_success": sum(bool(record.get("task_success")) for record in records),
        "semantic_success_rate": ratio(
            sum(bool(record.get("task_success")) for record in records), len(records)
        ),
        "tool_call_present": tool_present,
        "tool_call_parse_rate": ratio(tool_present, len(records)),
        "parser_kinds": {
            kind: sum(
                parser.parse(text).kind == kind
                for text in raw_outputs
            )
            for kind in ("tool_call", "final", "invalid")
        },
    }


def summarize(records: list[dict[str, Any]], input_path: str | Path) -> dict[str, Any]:
    parser = ToolCallParser()
    first_outputs = [record["steps"][0]["model_output"] for record in records]
    canonical = sum(CanonicalProtocolNormalizer.is_canonical(text) for text in first_outputs)
    parsed_calls = [parser.parse(text).tool_call for text in first_outputs]
    parsed_calls = [call for call in parsed_calls if call is not None]
    semantic = sum(bool(record["semantic_success"]) for record in records)
    strict = sum(bool(record["strict_success"]) for record in records)
    tool_selection = sum(
        call is not None and call.name == record["expected"]["tool_name"]
        for record, call in zip(records, [parser.parse(text).tool_call for text in first_outputs])
    )
    argument = sum(
        call is not None and call.arguments == record["expected"]["arguments"]
        for record, call in zip(records, [parser.parse(text).tool_call for text in first_outputs])
    )
    execution = sum(record["execution"] is not None and record["execution"].get("ok") is True for record in records)
    final = sum(
        str(record.get("final_output") or "").strip() == str(record["expected"]["final_answer"]).strip()
        for record in records
    )
    constrained_first = sum(bool(record["generation"][0]["constrained"]) for record in records)
    return {
        "input_path": str(input_path),
        "input_sha256": sha256(input_path),
        "episodes": len(records),
        "first_action_outputs": len(first_outputs),
        "constrained_first_action": constrained_first,
        "constrained_canonical_first_action": canonical,
        "constrained_canonical_rate": ratio(canonical, len(first_outputs)),
        "parsed_tool_calls": len(parsed_calls),
        "tool_selection_correct": tool_selection,
        "tool_selection_accuracy": ratio(tool_selection, len(records)),
        "argument_correct": argument,
        "argument_accuracy": ratio(argument, len(records)),
        "execution_success": execution,
        "execution_success_rate": ratio(execution, len(records)),
        "final_answer_exact": final,
        "final_answer_exact_rate": ratio(final, len(records)),
        "semantic_success": semantic,
        "semantic_success_rate": ratio(semantic, len(records)),
        "strict_success": strict,
        "strict_success_rate": ratio(strict, len(records)),
        "average_latency_ms": ratio(
            sum(float(record.get("elapsed_ms", 0.0)) for record in records), len(records)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolate canonical protocol constrained decoding.")
    parser.add_argument("--model-config", default="configs/base_model.yaml")
    parser.add_argument("--adapter", default="models/adapters/protocol_sft_qwen3_1.7b")
    parser.add_argument("--input", default="data/evaluation/tool_calling_smoke.jsonl")
    parser.add_argument(
        "--control-predictions",
        default="outputs/agent_benchmark_protocol_sft/sft/tool_calling_predictions.jsonl",
    )
    parser.add_argument("--output-predictions", default="outputs/experiment_3_protocol_constrained_isolation/predictions.jsonl")
    parser.add_argument("--summary-output", default="artifacts/experiment_3_protocol_constrained_isolation.json")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    model_config = load_yaml(args.model_config)
    runner = ConstrainedFirstActionRunner(
        model_config["model"]["local_path"],
        args.adapter,
        max_new_tokens=args.max_new_tokens,
    )
    environment = CalculatorEnvironment(build_default_registry())
    records: list[dict[str, Any]] = []
    for sample in read_jsonl(args.input):
        runner.outputs = []
        result = environment.run(
            sample,
            runner.generate,
            task_id=f"protocol-constrained:{sample['id']}",
        )
        row = result.to_dict()
        row["expected"] = sample["expected"]
        row["generation"] = list(runner.outputs)
        records.append(row)

    write_jsonl(args.output_predictions, records)
    summary = {
        "experiment": "experiment_3_protocol_constrained_isolation",
        "scope": {
            "new_model_inference": True,
            "training": False,
            "reward_changes": False,
            "frozen_final_test_access": False,
            "constraint_scope": "calculator tool name plus arithmetic-value canonical wrapper",
            "semantic_oracle": False,
        },
        "model_config": args.model_config,
        "adapter": args.adapter,
        "control": control_summary(args.control_predictions),
        "constrained": summarize(records, args.input),
        "predictions_path": args.output_predictions,
        "predictions_sha256": sha256(args.output_predictions),
    }
    output = Path(args.summary_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
