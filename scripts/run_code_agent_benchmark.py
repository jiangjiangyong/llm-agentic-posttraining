from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

from llm_posttrain.agent.code_environment import CodeAgentEnvironment
from llm_posttrain.config import load_yaml
from llm_posttrain.models.adapter_runner import build_adapter_runner
from llm_posttrain.models.loader import build_runner


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


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def ratio(value: int, total: int) -> float:
    return value / total if total else 0.0


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    reward_keys = (
        "format_reward",
        "canonical_format_reward",
        "tool_selection_reward",
        "argument_reward",
        "execution_reward",
        "unit_test_reward",
        "final_answer_reward",
        "efficiency_reward",
        "total_reward",
    )
    categories: dict[str, dict[str, Any]] = {}
    for record in records:
        category = str(record.get("category", "unknown"))
        bucket = categories.setdefault(
            category,
            {
                "total": 0,
                "semantic_success": 0,
                "strict_success": 0,
                "mean_reward": 0.0,
            },
        )
        bucket["total"] += 1
        bucket["semantic_success"] += int(record["semantic_success"])
        bucket["strict_success"] += int(record["strict_success"])
        bucket["mean_reward"] += float(record["reward"]["total_reward"])
    for bucket in categories.values():
        bucket["semantic_success_rate"] = ratio(
            bucket["semantic_success"], bucket["total"]
        )
        bucket["strict_success_rate"] = ratio(
            bucket["strict_success"], bucket["total"]
        )
        bucket["mean_reward"] /= bucket["total"]

    return {
        "total": total,
        "semantic_success_rate": ratio(
            sum(int(record["semantic_success"]) for record in records),
            total,
        ),
        "strict_success_rate": ratio(
            sum(int(record["strict_success"]) for record in records),
            total,
        ),
        "runtime_success_rate": ratio(
            sum(int(record["runtime_success"]) for record in records),
            total,
        ),
        "mean_reward": (
            sum(float(record["reward"]["total_reward"]) for record in records)
            / total
            if total
            else 0.0
        ),
        "mean_components": {
            key: (
                sum(float(record["reward"][key]) for record in records) / total
                if total
                else 0.0
            )
            for key in reward_keys
        },
        "tool_calls_per_episode": (
            sum(len(record["tool_calls"]) for record in records) / total
            if total
            else 0.0
        ),
        "unit_test_pass_rate": ratio(
            sum(
                int(
                    any(
                        result.get("tool_name") == "run_unit_tests"
                        and isinstance(result.get("output"), dict)
                        and result["output"].get("passed") is True
                        for result in record["tool_results"]
                    )
                )
                for record in records
            ),
            total,
        ),
        "by_category": categories,
    }


def build_backend(
    model_config: dict[str, Any],
    name: str,
    args: argparse.Namespace,
) -> Any:
    if name == "base":
        runner = build_runner(model_config)
        runner.max_new_tokens = args.max_new_tokens
        return runner
    adapter_paths = {
        "code_sft": args.sft_adapter,
        "code_dpo": args.dpo_adapter,
        "code_grpo": args.grpo_adapter,
    }
    adapter_path = adapter_paths.get(name)
    if not adapter_path:
        raise ValueError(f"missing adapter path for backend: {name}")
    runner = build_adapter_runner(
        model_config,
        adapter_path,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    return runner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/base_model.yaml")
    parser.add_argument(
        "--dataset",
        default="data/code_agent/benchmark.jsonl",
    )
    parser.add_argument(
        "--backends",
        default="base,code_sft,code_dpo,code_grpo",
    )
    parser.add_argument(
        "--sft-adapter",
        default="models/adapters/code_agent_sft_qwen3_1.7b",
    )
    parser.add_argument(
        "--dpo-adapter",
        default="models/adapters/code_agent_dpo_qwen3_1.7b",
    )
    parser.add_argument(
        "--grpo-adapter",
        default="models/adapters/code_agent_grpo_qwen3_1.7b",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/code_agent_benchmark",
    )
    parser.add_argument(
        "--summary-output",
        default="artifacts/code_agent_benchmark_summary.json",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    model_config = load_yaml(args.model_config)
    samples = read_jsonl(args.dataset)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    if not samples:
        raise ValueError("benchmark dataset is empty")
    environment = CodeAgentEnvironment(max_steps=args.max_steps)
    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "samples": len(samples),
        "categories": sorted({str(sample["category"]) for sample in samples}),
        "backends": {},
    }

    for name in [value.strip() for value in args.backends.split(",") if value.strip()]:
        print(f"Evaluating backend: {name}")
        runner = build_backend(model_config, name, args)
        records: list[dict[str, Any]] = []
        for sample in samples:
            episode = environment.run(
                sample,
                runner.generate,
                task_id=f"{name}:{sample['id']}",
            )
            row = episode.to_dict()
            row["backend"] = name
            row["category"] = sample["category"]
            records.append(row)
        backend_dir = Path(args.output_dir) / name
        write_jsonl(backend_dir / "episodes.jsonl", records)
        metrics = aggregate(records)
        write_json(backend_dir / "metrics.json", metrics)
        summary["backends"][name] = metrics
        print(json.dumps({name: metrics}, ensure_ascii=False))
        del runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
