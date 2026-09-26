from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from llm_posttrain.agent.environment import CalculatorEnvironment
from llm_posttrain.agent.runtime import AgentRuntime
from llm_posttrain.config import load_yaml
from llm_posttrain.evaluation.scorers import score_prediction
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


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def aggregate_general(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, dict[str, int]] = {}
    for record in records:
        category = str(record["category"])
        stats = by_category.setdefault(category, {"total": 0, "passed": 0})
        stats["total"] += 1
        stats["passed"] += int(bool(record["passed"]))
    category_metrics = {
        category: {
            **stats,
            "accuracy": _ratio(stats["passed"], stats["total"]),
        }
        for category, stats in sorted(by_category.items())
    }
    latencies = [float(record["latency_ms"]) for record in records]
    passed = sum(int(bool(record["passed"])) for record in records)
    return {
        "total": len(records),
        "passed": passed,
        "accuracy": _ratio(passed, len(records)),
        "average_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "by_category": category_metrics,
    }


def aggregate_tool(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    call_count = sum(bool(record["tool_call_present"]) for record in records)
    return {
        "total": total,
        "tool_call_present": call_count,
        "tool_call_parse_rate": _ratio(call_count, total),
        "tool_name_accuracy_given_call": _ratio(
            sum(
                bool(record["tool_name_correct"])
                for record in records
                if record["tool_call_present"]
            ),
            call_count,
        ),
        "argument_accuracy_given_call": _ratio(
            sum(
                bool(record["arguments_correct"])
                for record in records
                if record["tool_call_present"]
            ),
            call_count,
        ),
        "final_answer_exact_rate": _ratio(
            sum(bool(record["final_answer_exact"]) for record in records), total
        ),
        "task_success_rate": _ratio(
            sum(bool(record["task_success"]) for record in records), total
        ),
        "average_latency_ms": (
            sum(float(record["elapsed_ms"]) for record in records) / total
            if total
            else 0.0
        ),
    }


def aggregate_agent(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    reward_keys = (
        "format_reward",
        "canonical_format_reward",
        "tool_selection_reward",
        "argument_reward",
        "execution_reward",
        "final_answer_reward",
        "total_reward",
    )
    mean_components = {
        key: (
            sum(float(record["reward"][key]) for record in records) / total
            if total
            else 0.0
        )
        for key in reward_keys
    }
    semantic_successes = sum(
        bool(record.get("semantic_success", record.get("success", False)))
        for record in records
    )
    strict_successes = sum(
        bool(record.get("strict_success", record.get("success", False)))
        for record in records
    )
    return {
        "total": total,
        "semantic_success_rate": _ratio(semantic_successes, total),
        "strict_success_rate": _ratio(strict_successes, total),
        "mean_reward": mean_components["total_reward"],
        "mean_components": mean_components,
        "average_latency_ms": (
            sum(float(record["elapsed_ms"]) for record in records) / total
            if total
            else 0.0
        ),
    }


def evaluate_general(
    runner: Any,
    samples: list[dict[str, Any]],
    backend_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for sample in samples:
        result = runner.generate_with_stats(sample["messages"])
        score_result = score_prediction(
            scorer_name=sample["scorer"],
            prediction=result.text,
            expected=sample.get("expected"),
        )
        records.append(
            {
                "backend": backend_name,
                "id": sample["id"],
                "category": sample["category"],
                "prediction": result.text,
                "expected": sample.get("expected"),
                "scorer": sample["scorer"],
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "latency_ms": result.latency_ms,
                **score_result,
            }
        )
    return aggregate_general(records), records


class RunnerBackend:
    def __init__(self, runner: Any) -> None:
        self.runner = runner

    def generate(self, messages: list[dict[str, Any]]) -> str:
        return self.runner.generate(messages)


def evaluate_tool(
    runner: Any,
    samples: list[dict[str, Any]],
    backend_name: str,
    *,
    max_steps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    backend = RunnerBackend(runner)
    for sample in samples:
        expected = sample["expected"]
        started = time.perf_counter()
        result = AgentRuntime(
            backend=backend,
            registry=build_default_registry(),
            max_steps=max_steps,
        ).run(sample["messages"], task_id=str(sample["id"]))
        elapsed_ms = (time.perf_counter() - started) * 1000
        tool_calls = [
            event for event in result.trace.events if event.kind == "tool_call"
        ]
        tool_results = [
            event for event in result.trace.events if event.kind == "tool_result"
        ]
        actual_call = tool_calls[0].payload if tool_calls else None
        actual_result = tool_results[0].payload if tool_results else None
        present = actual_call is not None
        name_correct = present and actual_call["name"] == expected["tool_name"]
        arguments_correct = present and actual_call["arguments"] == expected["arguments"]
        final_exact = (
            result.final_answer is not None
            and result.final_answer.strip() == str(expected["final_answer"]).strip()
        )
        success = bool(
            result.success
            and name_correct
            and arguments_correct
            and final_exact
            and actual_result
            and actual_result.get("ok") is True
        )
        records.append(
            {
                "backend": backend_name,
                "id": sample["id"],
                "category": sample.get("category", "unknown"),
                "expected": expected,
                "actual_tool_call": actual_call,
                "actual_tool_result": actual_result,
                "final_answer": result.final_answer,
                "runtime_success": result.success,
                "tool_call_present": present,
                "tool_name_correct": name_correct,
                "arguments_correct": arguments_correct,
                "final_answer_exact": final_exact,
                "task_success": success,
                "elapsed_ms": elapsed_ms,
                "trace": result.trace.to_dict(),
                "error": result.error,
            }
        )
    return aggregate_tool(records), records


def evaluate_agent(
    runner: Any,
    samples: list[dict[str, Any]],
    backend_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    environment = CalculatorEnvironment(build_default_registry())
    records: list[dict[str, Any]] = []
    for sample in samples:
        episode = environment.run(
            sample,
            runner.generate,
            task_id=f"{backend_name}:{sample['id']}",
        )
        row = episode.to_dict()
        row["backend"] = backend_name
        records.append(row)
    return aggregate_agent(records), records


def build_backend(
    model_config: dict[str, Any],
    name: str,
    adapter_path: str | None,
    max_new_tokens: int,
) -> Any:
    if name == "base":
        runner = build_runner(model_config)
        runner.max_new_tokens = max_new_tokens
        return runner
    if not adapter_path:
        raise ValueError(f"Adapter path is required for backend {name}")
    return build_adapter_runner(
        model_config,
        adapter_path,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )


def parse_backend_names(value: str) -> list[str]:
    names = [item.strip().lower() for item in value.split(",") if item.strip()]
    allowed = {"base", "sft", "dpo", "grpo"}
    unknown = sorted(set(names) - allowed)
    if unknown:
        raise ValueError(f"Unknown backends: {unknown}")
    if not names:
        raise ValueError("At least one backend is required")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one comparable benchmark over Base, SFT, DPO and GRPO."
    )
    parser.add_argument("--model-config", default="configs/base_model.yaml")
    parser.add_argument(
        "--general-dataset", default="data/evaluation/base_smoke.jsonl"
    )
    parser.add_argument(
        "--tool-dataset", default="data/evaluation/tool_calling_smoke.jsonl"
    )
    parser.add_argument("--agent-dataset", default="data/agentic_rl/valid.jsonl")
    parser.add_argument("--backends", default="base,sft,dpo,grpo")
    parser.add_argument("--sft-adapter", default="models/adapters/sft_qwen3_1.7b")
    parser.add_argument(
        "--dpo-adapter", default="models/adapters/dpo_qwen3_1.7b_v2"
    )
    parser.add_argument("--grpo-adapter", default="models/adapters/grpo_qwen3_1.7b")
    parser.add_argument("--output-dir", default="outputs/agent_benchmark")
    parser.add_argument(
        "--manifest-output", default="artifacts/agent_benchmark_manifest.json"
    )
    parser.add_argument(
        "--summary-output", default="artifacts/agent_benchmark_summary.json"
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--max-general-samples", type=int, default=None)
    parser.add_argument("--max-tool-samples", type=int, default=None)
    parser.add_argument("--max-agent-samples", type=int, default=None)
    args = parser.parse_args()

    model_config = load_yaml(args.model_config)
    general_samples = read_jsonl(args.general_dataset)
    tool_samples = read_jsonl(args.tool_dataset)
    agent_samples = read_jsonl(args.agent_dataset)
    if args.max_general_samples is not None:
        general_samples = general_samples[: args.max_general_samples]
    if args.max_tool_samples is not None:
        tool_samples = tool_samples[: args.max_tool_samples]
    if args.max_agent_samples is not None:
        agent_samples = agent_samples[: args.max_agent_samples]
    if not general_samples or not tool_samples or not agent_samples:
        raise ValueError("All benchmark datasets must contain at least one sample")

    backend_names = parse_backend_names(args.backends)
    adapter_paths = {
        "sft": args.sft_adapter,
        "dpo": args.dpo_adapter,
        "grpo": args.grpo_adapter,
    }
    datasets = {
        "general": args.general_dataset,
        "tool_calling": args.tool_dataset,
        "agent_environment": args.agent_dataset,
    }
    manifest = {
        "version": "agent_benchmark_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "model_config": args.model_config,
        "datasets": {
            name: {
                "path": path,
                "count": len(
                    {"general": general_samples, "tool_calling": tool_samples, "agent_environment": agent_samples}[name]
                ),
                "sha256": file_sha256(path),
            }
            for name, path in datasets.items()
        },
        "backends": {
            name: {"kind": "base" if name == "base" else "adapter", "adapter_path": adapter_paths.get(name)}
            for name in backend_names
        },
        "max_new_tokens": args.max_new_tokens,
        "max_steps": args.max_steps,
    }
    write_json(args.manifest_output, manifest)

    benchmark_summary: dict[str, Any] = {
        "version": "agent_benchmark_v1",
        "manifest": args.manifest_output,
        "output_dir": args.output_dir,
        "backends": {},
    }
    output_root = Path(args.output_dir)
    for backend_name in backend_names:
        print(f"Running benchmark backend: {backend_name}")
        runner = build_backend(
            model_config,
            backend_name,
            adapter_paths.get(backend_name),
            args.max_new_tokens,
        )
        backend_dir = output_root / backend_name
        try:
            general_metrics, general_records = evaluate_general(
                runner, general_samples, backend_name
            )
            tool_metrics, tool_records = evaluate_tool(
                runner,
                tool_samples,
                backend_name,
                max_steps=args.max_steps,
            )
            agent_metrics, agent_records = evaluate_agent(
                runner, agent_samples, backend_name
            )
            write_jsonl(backend_dir / "general_predictions.jsonl", general_records)
            write_jsonl(backend_dir / "tool_calling_predictions.jsonl", tool_records)
            write_jsonl(backend_dir / "agent_predictions.jsonl", agent_records)
            write_json(
                backend_dir / "metrics.json",
                {
                    "backend": backend_name,
                    "general": general_metrics,
                    "tool_calling": tool_metrics,
                    "agent_environment": agent_metrics,
                },
            )
            benchmark_summary["backends"][backend_name] = {
                "general": general_metrics,
                "tool_calling": tool_metrics,
                "agent_environment": agent_metrics,
                "paths": {
                    "general": str(backend_dir / "general_predictions.jsonl"),
                    "tool_calling": str(backend_dir / "tool_calling_predictions.jsonl"),
                    "agent_environment": str(backend_dir / "agent_predictions.jsonl"),
                    "metrics": str(backend_dir / "metrics.json"),
                },
            }
        finally:
            del runner
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_json(args.summary_output, benchmark_summary)
    print("\nBackend comparison")
    print("backend | general | tool success | agent mean reward | agent semantic | agent strict")
    for backend_name in backend_names:
        result = benchmark_summary["backends"][backend_name]
        print(
            f"{backend_name} | "
            f"{result['general']['accuracy']:.2%} | "
            f"{result['tool_calling']['task_success_rate']:.2%} | "
            f"{result['agent_environment']['mean_reward']:.4f} | "
            f"{result['agent_environment']['semantic_success_rate']:.2%} | "
            f"{result['agent_environment']['strict_success_rate']:.2%}"
        )
    print(f"Manifest: {args.manifest_output}")
    print(f"Summary:  {args.summary_output}")


if __name__ == "__main__":
    main()
