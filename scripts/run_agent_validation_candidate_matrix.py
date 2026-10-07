"""Compare Stable C0 and controlled experiment adapters on one fixed Agent validation smoke.

This is an evaluation-only matrix. It does not train, read final-test targets, or feed
validation outputs back into any adapter. The matrix is intentionally separate from the
AL/AM/AN structured content_ref benchmark so regression/no-harm is visible on the
calculator Agent environment as well.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from llm_posttrain.config import load_yaml
from scripts.run_agent_benchmark import build_backend, evaluate_agent, read_jsonl


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_candidates(values: list[str] | None) -> list[tuple[str, str]]:
    raw = values or [
        "stable_c0=models/adapters/sft_qwen3_1.7b",
        "AL=models/adapters/experiment_5e_al_balanced_residual_weighting_seed20261014",
        "AM=models/adapters/experiment_5e_am_balanced_residual_weighting_seed20261015",
        "AN=models/adapters/experiment_5e_an_balanced_residual_weighting_seed20261016",
    ]
    parsed: list[tuple[str, str]] = []
    for item in raw:
        if "=" not in item:
            raise ValueError(f"Candidate must use name=adapter_path: {item}")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Candidate must use non-empty name=adapter_path: {item}")
        parsed.append((name, path))
    if len({name for name, _ in parsed}) != len(parsed):
        raise ValueError("Candidate names must be unique")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate candidate adapters on fixed Agent validation smoke")
    parser.add_argument("--model-config", default="configs/base_model.yaml")
    parser.add_argument("--dataset", default="data/agentic_rl/valid.jsonl")
    parser.add_argument("--candidate", action="append", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", default="artifacts/agent_validation_candidate_matrix_v1.json")
    parser.add_argument("--report", default="docs/agent_validation_candidate_matrix_v1.md")
    args = parser.parse_args()

    dataset_path = ROOT / args.dataset
    model_config = load_yaml(ROOT / args.model_config)
    samples = read_jsonl(dataset_path)
    if not samples:
        raise ValueError("Agent validation dataset must be non-empty")
    candidates = parse_candidates(args.candidate)

    summaries: dict[str, dict[str, Any]] = {}
    records: dict[str, list[dict[str, Any]]] = {}
    for name, adapter_path in candidates:
        adapter = ROOT / adapter_path
        if not (adapter / "adapter_model.safetensors").exists():
            raise FileNotFoundError(f"Missing adapter for {name}: {adapter}")
        print(f"[candidate-matrix] Loading {name}: {adapter_path}")
        runner = build_backend(model_config, "sft", adapter_path, args.max_new_tokens)
        try:
            metrics, candidate_records = evaluate_agent(runner, samples, name)
            summaries[name] = {
                "adapter_path": adapter_path,
                "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
                **metrics,
            }
            records[name] = candidate_records
        finally:
            del runner
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    artifact = {
        "schema": "llm_agentic_agent_validation_candidate_matrix_v1",
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "dataset_count": len(samples),
        "split": "agent_validation_only",
        "holdout_used": False,
        "training_used": False,
        "feedback_used": False,
        "claim_boundary": [
            "This is a fixed validation smoke on the calculator Agent environment.",
            "It measures regression/no-harm for candidate adapters; it is not a Rich benchmark or production claim.",
            "Validation outputs are not fed back into training or adapter selection.",
        ],
        "candidates": summaries,
        "records": records,
    }
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report_lines = [
        "# Agent validation candidate matrix",
        "",
        "Evaluation-only comparison on `data/agentic_rl/valid.jsonl`; no training or feedback is performed.",
        "",
        "| Candidate | Semantic | Strict | Mean reward |",
        "|---|---:|---:|---:|",
    ]
    for name, summary in summaries.items():
        report_lines.append(
            f"| {name} | {summary['semantic_success_rate']:.2%} | "
            f"{summary['strict_success_rate']:.2%} | {summary['mean_reward']:.4f} |"
        )
    report_lines.extend(
        [
            "",
            "This smoke is a no-harm/regression check for the calculator Agent environment.",
            "It does not establish Rich benchmark generalization or unrestricted production Agent ability.",
            f"Artifact: `{args.output}`",
        ]
    )
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": args.output, "report": args.report, "candidates": summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
