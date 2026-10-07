"""Evaluate protocol-aware adapter routing on disjoint validation splits.

Rich v2 tasks use the task-matched Rich SFT adapter. Tool v1 tasks use the
protocol SFT adapter behind the loss-aware normalizer. Each adapter is loaded
sequentially, so this is a validation matrix rather than a multi-adapter
production serving claim.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from llm_posttrain.agent.adapter_router import ProtocolAdapterRouter
from llm_posttrain.agent.code_environment import CodeAgentEnvironment
from llm_posttrain.agent.protocol import ToolCallParser
from llm_posttrain.agent.protocol_normalizer import (
    CanonicalProtocolNormalizer,
    ProtocolNormalizingBackend,
)
from llm_posttrain.agent.runtime import AgentRuntime
from llm_posttrain.config import load_yaml
from llm_posttrain.models.adapter_runner import build_adapter_runner
from llm_posttrain.tools.registry import build_default_registry


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def run_rich_task(
    runner: Any,
    task: dict[str, Any],
    *,
    max_steps: int,
) -> dict[str, Any]:
    result = CodeAgentEnvironment(max_steps=max_steps).run(
        task, runner.generate, task_id=f"route:rich_v2:{task['id']}"
    )
    return {
        "id": task["id"],
        "route": "rich_v2",
        "semantic_success": result.semantic_success,
        "strict_success": result.strict_success,
        "runtime_success": result.runtime_success,
        "mean_reward": result.reward.total_reward,
        "tool_calls": len(result.tool_calls),
    }


def run_tool_task(
    runner: Any,
    task: dict[str, Any],
    *,
    normalizer: CanonicalProtocolNormalizer,
) -> dict[str, Any]:
    backend = ProtocolNormalizingBackend(runner, normalizer)
    result = AgentRuntime(
        backend=backend,
        registry=build_default_registry(),
        max_steps=3,
    ).run(task["messages"], task_id=f"route:tool_v1:{task['id']}")
    decisions = backend.records
    first = decisions[0] if decisions else None
    parser = ToolCallParser()
    raw_first = first.raw_text if first is not None else ""
    raw_canonical = normalizer.is_canonical(raw_first)
    raw_call = parser.parse(raw_first).tool_call
    tool_events = [
        event for event in result.trace.events if event.kind == "tool_call"
    ]
    tool_results = [
        event for event in result.trace.events if event.kind == "tool_result"
    ]
    actual_call = tool_events[0].payload if tool_events else None
    actual_result = tool_results[0].payload if tool_results else None
    expected = task["expected"]
    semantic = bool(
        actual_call is not None
        and actual_call.get("name") == expected["tool_name"]
        and actual_call.get("arguments") == expected["arguments"]
        and actual_result is not None
        and actual_result.get("ok") is True
        and (result.final_answer or "").strip()
        == str(expected["final_answer"]).strip()
    )
    normalized_strict = bool(
        semantic
        and first is not None
        and first.normalized_text is not None
        and normalizer.is_canonical(first.normalized_text)
    )
    return {
        "id": task["id"],
        "route": "tool_v1",
        "semantic_success": semantic,
        "strict_success": normalized_strict,
        "runtime_success": result.success,
        "mean_reward": None,
        "tool_calls": len(tool_events),
        "raw_canonical": raw_canonical,
        "raw_parseable_action": raw_call is not None,
        "normalized_canonical": bool(
            first is not None
            and first.normalized_text is not None
            and normalizer.is_canonical(first.normalized_text)
        ),
        "normalization_status": first.status if first is not None else None,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    route = records[0]["route"] if records else "unknown"
    result: dict[str, Any] = {
        "route": route,
        "total": total,
        "semantic_success": sum(bool(row["semantic_success"]) for row in records),
        "strict_success": sum(bool(row["strict_success"]) for row in records),
        "runtime_success": sum(bool(row["runtime_success"]) for row in records),
        "route_records": len(records),
    }
    result["semantic_success_rate"] = ratio(result["semantic_success"], total)
    result["strict_success_rate"] = ratio(result["strict_success"], total)
    result["runtime_success_rate"] = ratio(result["runtime_success"], total)
    rewards = [row["mean_reward"] for row in records if row["mean_reward"] is not None]
    if rewards:
        result["mean_reward"] = sum(float(value) for value in rewards) / len(rewards)
    if route == "tool_v1":
        result.update(
            {
                "raw_canonical": sum(bool(row["raw_canonical"]) for row in records),
                "raw_parseable_action": sum(
                    bool(row["raw_parseable_action"]) for row in records
                ),
                "normalized_canonical": sum(
                    bool(row["normalized_canonical"]) for row in records
                ),
                "normalization_status_counts": dict(
                    Counter(row["normalization_status"] for row in records)
                ),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/base_model.yaml")
    parser.add_argument("--rich-dataset", default="data/code_agent_rich/valid.jsonl")
    parser.add_argument("--tool-dataset", default="data/agentic_rl/valid.jsonl")
    parser.add_argument(
        "--rich-adapter",
        default="models/adapters/code_agent_rich_sft_from_v2_qwen3_1.7b",
    )
    parser.add_argument(
        "--tool-adapter", default="models/adapters/protocol_sft_qwen3_1.7b"
    )
    parser.add_argument("--fallback-adapter", default="models/adapters/sft_qwen3_1.7b")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-rich-steps", type=int, default=5)
    parser.add_argument(
        "--output", default="artifacts/adapter_route_validation_matrix_v1.json"
    )
    parser.add_argument(
        "--report", default="docs/adapter_route_validation_matrix_v1.md"
    )
    args = parser.parse_args()

    rich_path = ROOT / args.rich_dataset
    tool_path = ROOT / args.tool_dataset
    rich_tasks = read_jsonl(rich_path)
    tool_tasks = read_jsonl(tool_path)
    tasks = rich_tasks + tool_tasks
    router = ProtocolAdapterRouter(
        rich_adapter=args.rich_adapter,
        protocol_adapter=args.tool_adapter,
        fallback_adapter=args.fallback_adapter,
    )
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    decisions: dict[str, dict[str, Any]] = {}
    for task in tasks:
        route = router.route(task.get("messages", []))
        groups[route.name].append(task)
        decisions[str(task["id"])] = route.to_dict()

    model_config = load_yaml(ROOT / args.model_config)
    records: list[dict[str, Any]] = []
    route_summaries: dict[str, dict[str, Any]] = {}
    for route_name in ("rich_v2", "tool_v1", "stable_fallback"):
        route_tasks = groups.get(route_name, [])
        if not route_tasks:
            route_summaries[route_name] = {
                "route": route_name,
                "total": 0,
                "semantic_success": 0,
                "strict_success": 0,
                "runtime_success": 0,
                "route_records": 0,
                "semantic_success_rate": 0.0,
                "strict_success_rate": 0.0,
                "runtime_success_rate": 0.0,
                "skipped": True,
            }
            continue
        adapter_path = router.routes[route_name].adapter_path
        runner = build_adapter_runner(
            model_config,
            adapter_path,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
        route_records: list[dict[str, Any]] = []
        try:
            for task in route_tasks:
                if route_name == "rich_v2":
                    route_records.append(
                        run_rich_task(
                            runner,
                            task,
                            max_steps=args.max_rich_steps,
                        )
                    )
                elif route_name == "tool_v1":
                    route_records.append(
                        run_tool_task(
                            runner,
                            task,
                            normalizer=CanonicalProtocolNormalizer(),
                        )
                    )
                else:
                    raise ValueError(
                        "fallback tasks require a separate explicitly registered validation split"
                    )
        finally:
            del runner
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        records.extend(route_records)
        route_summaries[route_name] = summarize(route_records)

    artifact = {
        "schema": "llm_agentic_protocol_adapter_route_validation_v1",
        "datasets": {
            "rich": {
                "path": args.rich_dataset,
                "count": len(rich_tasks),
                "sha256": sha256(rich_path),
            },
            "tool": {
                "path": args.tool_dataset,
                "count": len(tool_tasks),
                "sha256": sha256(tool_path),
            },
        },
        "scope": {
            "validation_only": True,
            "training_used": False,
            "holdout_used": False,
            "feedback_used": False,
            "expected_action_used_for_routing": False,
            "route_source": "system protocol marker only",
            "adapters_loaded_sequentially": True,
        },
        "route_policy": {
            name: route.to_dict() for name, route in router.routes.items()
        },
        "route_counts": {name: len(items) for name, items in groups.items()},
        "route_summaries": route_summaries,
        "records": records,
        "decisions": decisions,
        "claim_boundary": [
            "This is a validation matrix for capability isolation, not a production multi-adapter serving benchmark.",
            "Rich v2 and Tool v1 use disjoint protocol markers and adapters; no expected action or holdout content is used for routing.",
            "Tool v1 strict success is after the explicit protocol normalizer; Rich v2 strict success is raw CodeAgentEnvironment accounting.",
            "The matrix does not prove unrestricted generalization or a successful joint-training solution.",
        ],
    }
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Protocol-aware adapter route validation matrix",
        "",
        "Validation-only capability isolation: route from system protocol markers, then load adapters sequentially.",
        "",
        "| Route | Dataset rows | Semantic | Strict | Runtime | Mean reward |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for route_name in ("rich_v2", "tool_v1", "stable_fallback"):
        value = route_summaries[route_name]
        total = value["total"]
        reward = value.get("mean_reward")
        report.append(
            f"| {route_name} | {total} | {value['semantic_success']}/{total} | "
            f"{value['strict_success']}/{total} | {value['runtime_success']}/{total} | "
            f"{'n/a' if reward is None else f'{reward:.4f}'} |"
        )
    report.extend(
        [
            "",
            "Routing reads only the system protocol marker. It does not inspect expected actions, answers, categories, or holdout data.",
            "Tool v1 strict is normalized executor-facing strictness; it must not be presented as native model canonical-format learning.",
            f"Artifact: `{args.output}`",
        ]
    )
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact": args.output,
                "report": args.report,
                "route_counts": artifact["route_counts"],
                "route_summaries": route_summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
