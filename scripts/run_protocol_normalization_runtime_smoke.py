"""Evaluate protocol normalization inside the real calculator runtime.

This is a validation-only inference smoke. The model's raw output is recorded
before a loss-aware normalizer turns an already parseable tool call into the
canonical wrapper consumed by the executor. Raw canonical success and
normalized runtime success are reported separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run protocol normalization inside CalculatorEnvironment"
    )
    parser.add_argument(
        "--dataset", default="data/evaluation/tool_calling_smoke.jsonl"
    )
    parser.add_argument("--model-config", default="configs/base_model.yaml")
    parser.add_argument(
        "--adapter", default="models/adapters/protocol_sft_qwen3_1.7b"
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--output", default="artifacts/protocol_normalization_runtime_smoke_v1.json"
    )
    parser.add_argument(
        "--report", default="docs/protocol_normalization_runtime_smoke_v1.md"
    )
    args = parser.parse_args()

    dataset_path = ROOT / args.dataset
    model_config = load_yaml(ROOT / args.model_config)
    runner = build_adapter_runner(
        model_config,
        args.adapter,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    backend = ProtocolNormalizingBackend(runner)
    normalizer = CanonicalProtocolNormalizer()
    parser_for_raw = ToolCallParser()
    registry = build_default_registry()
    tasks = read_jsonl(dataset_path)
    records: list[dict[str, Any]] = []

    for sample in tasks:
        start = len(backend.records)
        episode = AgentRuntime(
            backend=backend,
            registry=registry,
            max_steps=3,
        ).run(
            sample["messages"], task_id=f"protocol-normalized:{sample['id']}"
        )
        decisions = backend.records[start:]
        first = decisions[0] if decisions else None
        raw_first = first.raw_text if first is not None else ""
        parsed_raw = parser_for_raw.parse(raw_first)
        raw_call = parsed_raw.tool_call
        expected = sample["expected"]
        raw_action_correct = bool(
            raw_call is not None
            and raw_call.name == expected["tool_name"]
            and raw_call.arguments == expected["arguments"]
        )
        tool_calls = [
            event for event in episode.trace.events if event.kind == "tool_call"
        ]
        tool_results = [
            event for event in episode.trace.events if event.kind == "tool_result"
        ]
        actual_call = tool_calls[0].payload if tool_calls else None
        actual_result = tool_results[0].payload if tool_results else None
        runtime_semantic = bool(
            episode.success
            and actual_call is not None
            and actual_call.get("name") == expected["tool_name"]
            and actual_call.get("arguments") == expected["arguments"]
            and actual_result is not None
            and actual_result.get("ok") is True
            and (episode.final_answer or "").strip()
            == str(expected["final_answer"]).strip()
        )
        runtime_normalized_strict = bool(
            runtime_semantic
            and first is not None
            and first.normalized_text is not None
            and normalizer.is_canonical(first.normalized_text)
        )
        records.append(
            {
                "id": sample["id"],
                "raw_first_action": raw_first,
                "raw_first_status": first.status if first is not None else None,
                "raw_first_canonical": normalizer.is_canonical(raw_first),
                "raw_first_parseable_tool_call": raw_call is not None,
                "raw_first_action_correct": raw_action_correct,
                "normalized_first_action": (
                    first.normalized_text if first is not None else None
                ),
                "normalized_first_canonical": bool(
                    first is not None
                    and first.normalized_text is not None
                    and normalizer.is_canonical(first.normalized_text)
                ),
                "runtime_semantic_success": runtime_semantic,
                "runtime_normalized_strict_success": runtime_normalized_strict,
                "runtime_execution_success": bool(
                    actual_result is not None and actual_result.get("ok") is True
                ),
                "final_output": episode.final_answer,
            }
        )

    total = len(records)
    raw_canonical = sum(int(row["raw_first_canonical"]) for row in records)
    raw_parseable = sum(
        int(row["raw_first_parseable_tool_call"]) for row in records
    )
    raw_action_correct = sum(int(row["raw_first_action_correct"]) for row in records)
    normalized_canonical = sum(
        int(row["normalized_first_canonical"]) for row in records
    )
    semantic = sum(int(row["runtime_semantic_success"]) for row in records)
    strict = sum(int(row["runtime_normalized_strict_success"]) for row in records)
    execution = sum(int(row["runtime_execution_success"]) for row in records)
    status_counts = Counter(
        item.status for item in backend.records if item.status is not None
    )
    artifact = {
        "schema": "llm_agentic_protocol_normalization_runtime_smoke_v1",
        "dataset": args.dataset,
        "dataset_sha256": sha256(dataset_path),
        "adapter": args.adapter,
        "split": "tool_calling_validation_smoke_only",
        "scope": {
            "new_model_inference": True,
            "training_used": False,
            "holdout_used": False,
            "feedback_used": False,
            "frozen_final_test_access": False,
            "reward_changes": False,
            "raw_and_normalized_separated": True,
        },
        "raw_model": {
            "episodes": total,
            "raw_canonical_first_action": raw_canonical,
            "raw_canonical_rate": ratio(raw_canonical, total),
            "parseable_tool_call": raw_parseable,
            "parseable_tool_call_rate": ratio(raw_parseable, total),
            "raw_action_correct": raw_action_correct,
            "raw_action_correct_rate": ratio(raw_action_correct, total),
        },
        "normalized_runtime": {
            "normalized_canonical_first_action": normalized_canonical,
            "normalized_canonical_rate": ratio(normalized_canonical, total),
            "semantic_success": semantic,
            "semantic_success_rate": ratio(semantic, total),
            "strict_success_after_normalization": strict,
            "strict_success_after_normalization_rate": ratio(strict, total),
            "execution_success": execution,
            "execution_success_rate": ratio(execution, total),
        },
        "normalization_status_counts": dict(sorted(status_counts.items())),
        "records": records,
        "claim_boundary": [
            "Raw canonical success measures the model output before normalization.",
            "Normalized strict success is executor-facing protocol reliability, not native model format learning.",
            "The normalizer only wraps a parser-recognized tool call and does not invent a tool or arguments.",
            "This is a six-row validation smoke, not Rich generalization or production evidence.",
        ],
    }
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_lines = [
        "# Protocol normalization runtime smoke",
        "",
        "A real adapter run through `CalculatorEnvironment` with an explicit, loss-aware protocol adapter.",
        "",
        "| Boundary | Result |",
        "|---|---:|",
        f"| Raw canonical first action | {raw_canonical}/{total} |",
        f"| Raw parser-recognized tool call | {raw_parseable}/{total} |",
        f"| Normalized canonical first action | {normalized_canonical}/{total} |",
        f"| Runtime semantic success after normalization | {semantic}/{total} |",
        f"| Runtime strict success after normalization | {strict}/{total} |",
        f"| Runtime execution success | {execution}/{total} |",
        "",
        "The normalized result is an executor-facing protocol guarantee. It does not change the raw model metric and cannot be reported as native canonical-format learning.",
        "",
        f"Artifact: `{args.output}`",
    ]
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact": args.output,
                "report": args.report,
                "raw_canonical": f"{raw_canonical}/{total}",
                "normalized_strict": f"{strict}/{total}",
                "semantic": f"{semantic}/{total}",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
