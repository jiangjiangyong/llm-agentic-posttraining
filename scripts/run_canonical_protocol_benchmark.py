"""Evaluate raw canonical tool-call behavior on the held-out validation split.

This benchmark deliberately uses the validation portion of the canonical
protocol dataset, not training rows or the Rich holdout. It reports semantic
success and strict canonical success separately so parser recovery cannot be
mistaken for the model emitting the required <tool_call> wrapper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from llm_posttrain.config import load_yaml
from scripts.run_agent_benchmark import build_backend, evaluate_agent, read_jsonl


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_tasks(
    path: Path,
    limit: int | None = None,
    start: int = 0,
) -> list[dict[str, Any]]:
    records = read_jsonl(path)
    records = records[start:]
    if limit is not None:
        records = records[:limit]
    tasks: list[dict[str, Any]] = []
    for record in records:
        messages = record.get("messages", [])
        metadata = record.get("metadata", {})
        if len(messages) < 2 or messages[0].get("role") != "system" or messages[1].get("role") != "user":
            raise ValueError(f"{record.get('id')}: canonical validation record must start system/user")
        if metadata.get("canonical_format") is not True:
            raise ValueError(f"{record.get('id')}: record is not marked canonical_format")
        tasks.append(
            {
                "id": str(record["id"]),
                "category": str(record.get("category", "tool_calling_protocol")),
                "messages": [messages[0], messages[1]],
                "expected": {
                    "tool_name": "calculator",
                    "arguments": {"expression": metadata["expression"]},
                    "final_answer": str(metadata["answer"]),
                },
            }
        )
    return tasks


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    semantic = sum(bool(row.get("semantic_success")) for row in records)
    strict = sum(bool(row.get("strict_success")) for row in records)
    canonical_reward = sum(
        float(row.get("reward", {}).get("canonical_format_reward", 0.0))
        for row in records
    )
    return {
        "total": total,
        "semantic_success": semantic,
        "semantic_success_rate": semantic / total if total else 0.0,
        "strict_canonical_success": strict,
        "strict_canonical_success_rate": strict / total if total else 0.0,
        "mean_canonical_format_reward": canonical_reward / total if total else 0.0,
    }


def run_backend(
    model_config: dict[str, Any],
    backend_name: str,
    adapter_path: str | None,
    tasks: list[dict[str, Any]],
    max_new_tokens: int,
) -> dict[str, Any]:
    runner = build_backend(model_config, backend_name, adapter_path, max_new_tokens)
    _, records = evaluate_agent(runner, tasks, backend_name)
    return {
        "summary": summarize(records),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit raw canonical tool-call behavior.")
    parser.add_argument("--dataset", default="data/protocol_alignment/valid.jsonl")
    parser.add_argument("--model-config", default="configs/base_model.yaml")
    parser.add_argument("--backends", default="base,protocol_sft")
    parser.add_argument("--protocol-sft-adapter", default="models/adapters/protocol_sft_qwen3_1.7b")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="artifacts/canonical_protocol_benchmark_v1.json")
    parser.add_argument("--report", default="docs/canonical_protocol_benchmark_v1.md")
    args = parser.parse_args()

    dataset_path = ROOT / args.dataset
    tasks = canonical_tasks(dataset_path, args.limit, args.start)
    model_config = load_yaml(ROOT / args.model_config)
    adapter_by_backend = {
        "base": None,
        "protocol_sft": args.protocol_sft_adapter,
    }
    backend_names = [item.strip() for item in args.backends.split(",") if item.strip()]
    unknown = sorted(set(backend_names) - set(adapter_by_backend))
    if unknown:
        raise ValueError(f"Unsupported backends: {unknown}")

    results = {
        name: run_backend(
            model_config,
            name,
            adapter_by_backend[name],
            tasks,
            args.max_new_tokens,
        )
        for name in backend_names
    }
    artifact = {
        "schema": "llm_agentic_canonical_protocol_benchmark_v1",
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "split": "protocol_alignment_validation_only",
        "start": args.start,
        "limit": args.limit,
        "holdout_used": False,
        "training_used": False,
        "claim_boundary": [
            "Strict canonical success measures the raw model output before parser recovery.",
            "Validation rows are separate from protocol_alignment train rows.",
            "This benchmark does not establish Rich benchmark or unrestricted Agent capability.",
        ],
        "backends": {name: {"summary": value["summary"]} for name, value in results.items()},
        "records": {name: value["records"] for name, value in results.items()},
    }
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_lines = [
        "# Canonical protocol benchmark",
        "",
        "This benchmark uses only `data/protocol_alignment/valid.jsonl` and reports raw canonical output separately from semantic parser recovery.",
        "",
        "| Backend | Semantic | Strict canonical | Mean canonical reward |",
        "|---|---:|---:|---:|",
    ]
    for name, value in results.items():
        summary = value["summary"]
        report_lines.append(
            f"| {name} | {summary['semantic_success']}/{summary['total']} | "
            f"{summary['strict_canonical_success']}/{summary['total']} | "
            f"{summary['mean_canonical_format_reward']:.3f} |"
        )
    report_lines.extend(
        [
            "",
            "Strict canonical success is not inferred from parser recovery; raw `<tool_call>` delimiters and the exact `{name, arguments}` payload are required.",
            "",
            f"Artifact: `{args.output}`",
        ]
    )
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": args.output, "report": args.report, "backends": artifact["backends"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
