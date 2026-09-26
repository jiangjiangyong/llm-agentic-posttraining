from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from llm_posttrain.agent.runtime import AgentRuntime
from llm_posttrain.config import load_yaml
from llm_posttrain.models.adapter_runner import build_adapter_runner
from llm_posttrain.models.loader import build_runner
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
                raise ValueError(f"Expected an object at line {line_number}")
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


class ModelBackend:
    def __init__(
        self,
        config: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> None:
        self.runner = build_runner(config)
        if max_new_tokens is not None:
            self.runner.max_new_tokens = max_new_tokens

    def generate(self, messages: list[dict[str, Any]]) -> str:
        return self.runner.generate(messages)


class AdapterBackend:
    def __init__(
        self,
        config: dict[str, Any],
        adapter_path: str,
        max_new_tokens: int | None = None,
    ) -> None:
        self.runner = build_adapter_runner(
            config,
            adapter_path,
            max_new_tokens=max_new_tokens or 128,
        )

    def generate(self, messages: list[dict[str, Any]]) -> str:
        return self.runner.generate(messages)


class ScriptedBackend:
    def __init__(self, expected: dict[str, Any]) -> None:
        self.expected = expected
        self.calls = 0

    def generate(self, messages: list[dict[str, Any]]) -> str:
        del messages
        self.calls += 1
        if self.calls == 1:
            payload = {
                "type": "tool_call",
                "name": self.expected["tool_name"],
                "arguments": self.expected["arguments"],
            }
            return f"<tool_call>{json.dumps(payload, ensure_ascii=False)}</tool_call>"
        return str(self.expected["final_answer"])


def evaluate_sample(
    sample: dict[str, Any],
    runtime: AgentRuntime,
) -> dict[str, Any]:
    expected = sample["expected"]
    started = time.perf_counter()
    result = runtime.run(sample["messages"], task_id=str(sample["id"]))
    elapsed_ms = (time.perf_counter() - started) * 1000

    tool_call_events = [
        event for event in result.trace.events if event.kind == "tool_call"
    ]
    tool_result_events = [
        event for event in result.trace.events if event.kind == "tool_result"
    ]
    actual_call = tool_call_events[0].payload if tool_call_events else None
    actual_result = tool_result_events[0].payload if tool_result_events else None

    tool_call_present = actual_call is not None
    tool_name_correct = (
        tool_call_present and actual_call["name"] == expected["tool_name"]
    )
    arguments_correct = (
        tool_call_present and actual_call["arguments"] == expected["arguments"]
    )
    final_answer_exact = (
        result.final_answer is not None
        and result.final_answer.strip() == str(expected["final_answer"]).strip()
    )
    task_success = bool(
        result.success
        and tool_name_correct
        and arguments_correct
        and final_answer_exact
        and actual_result
        and actual_result.get("ok") is True
    )
    return {
        "id": sample["id"],
        "category": sample.get("category", "unknown"),
        "expected": expected,
        "actual_tool_call": actual_call,
        "actual_tool_result": actual_result,
        "final_answer": result.final_answer,
        "runtime_success": result.success,
        "tool_call_present": tool_call_present,
        "tool_name_correct": tool_name_correct,
        "arguments_correct": arguments_correct,
        "final_answer_exact": final_answer_exact,
        "task_success": task_success,
        "elapsed_ms": elapsed_ms,
        "trace": result.trace.to_dict(),
        "error": result.error,
    }


def aggregate(records: list[dict[str, Any]], backend: str) -> dict[str, Any]:
    total = len(records)
    count = lambda key: sum(bool(record[key]) for record in records)
    call_count = count("tool_call_present")
    return {
        "backend": backend,
        "total": total,
        "tool_call_present": call_count,
        "tool_call_parse_rate": call_count / total if total else 0.0,
        "tool_name_accuracy_given_call": (
            sum(
                bool(record["tool_name_correct"])
                for record in records
                if record["tool_call_present"]
            )
            / call_count
            if call_count
            else 0.0
        ),
        "argument_accuracy_given_call": (
            sum(
                bool(record["arguments_correct"])
                for record in records
                if record["tool_call_present"]
            )
            / call_count
            if call_count
            else 0.0
        ),
        "final_answer_exact_rate": (
            count("final_answer_exact") / total if total else 0.0
        ),
        "task_success_rate": count("task_success") / total if total else 0.0,
        "average_latency_ms": (
            sum(float(record["elapsed_ms"]) for record in records) / total
            if total
            else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/agent_runtime.yaml")
    parser.add_argument(
        "--backend",
        choices=("model", "adapter", "scripted"),
        default="model",
    )
    parser.add_argument(
        "--adapter-path",
        default="models/adapters/sft_qwen3_1.7b",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    samples = read_jsonl(config["evaluation"]["dataset_path"])
    print(f"Loaded {len(samples)} tool-calling evaluation samples")

    if args.backend == "model":
        model_config = load_yaml(config["model_config"])
        backend: ModelBackend | AdapterBackend | ScriptedBackend = ModelBackend(
            model_config,
            max_new_tokens=args.max_new_tokens,
        )
    elif args.backend == "adapter":
        model_config = load_yaml(config["model_config"])
        backend = AdapterBackend(
            model_config,
            args.adapter_path,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        backend = ScriptedBackend(samples[0]["expected"])

    records: list[dict[str, Any]] = []
    for sample in tqdm(samples, desc=f"Evaluating ({args.backend})"):
        if args.backend == "scripted":
            backend = ScriptedBackend(sample["expected"])
        runtime = AgentRuntime(
            backend=backend,
            registry=build_default_registry(),
            max_steps=int(config["runtime"].get("max_steps", 3)),
            observation_role=str(config["runtime"].get("observation_role", "tool")),
        )
        records.append(evaluate_sample(sample, runtime))

    metrics = aggregate(records, args.backend)
    output_dir = Path(
        args.output_dir or config["evaluation"]["output_dirs"][args.backend]
    )
    write_jsonl(output_dir / "predictions.jsonl", records)
    write_json(output_dir / "metrics.json", metrics)

    print("\n" + "=" * 60)
    print("TOOL CALLING EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Backend:   {args.backend}")
    print(f"Total:     {metrics['total']}")
    print(f"Parse:     {metrics['tool_call_parse_rate']:.2%}")
    print(f"Arguments: {metrics['argument_accuracy_given_call']:.2%}")
    print(f"Final:     {metrics['final_answer_exact_rate']:.2%}")
    print(f"Success:   {metrics['task_success_rate']:.2%}")
    print(f"Predictions: {output_dir / 'predictions.jsonl'}")
    print(f"Metrics:     {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
