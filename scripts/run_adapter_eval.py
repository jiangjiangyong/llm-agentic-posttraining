from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from llm_posttrain.config import load_yaml
from llm_posttrain.evaluation.scorers import score_prediction
from llm_posttrain.models.adapter_runner import build_adapter_runner


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at line {line_number}")
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


def aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, dict[str, int]] = {}
    for record in records:
        category = str(record["category"])
        stats = by_category.setdefault(category, {"total": 0, "passed": 0})
        stats["total"] += 1
        stats["passed"] += int(bool(record["passed"]))

    category_metrics = {}
    for category, stats in sorted(by_category.items()):
        category_metrics[category] = {
            **stats,
            "accuracy": stats["passed"] / stats["total"] if stats["total"] else 0.0,
        }
    total = len(records)
    passed = sum(int(bool(record["passed"])) for record in records)
    latencies = [float(record["latency_ms"]) for record in records]
    return {
        "total": total,
        "passed": passed,
        "accuracy": passed / total if total else 0.0,
        "average_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "by_category": category_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/base_model.yaml")
    parser.add_argument(
        "--adapter-path",
        default="models/adapters/sft_qwen3_1.7b",
    )
    parser.add_argument("--dataset", default="data/evaluation/base_smoke.jsonl")
    parser.add_argument("--output-dir", default="outputs/sft_eval/base_smoke")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--display-name", default="SFT Adapter")
    args = parser.parse_args()

    model_config = load_yaml(args.model_config)
    samples = read_jsonl(args.dataset)
    print(f"Loaded {len(samples)} evaluation samples")
    runner = build_adapter_runner(
        model_config,
        args.adapter_path,
        max_new_tokens=args.max_new_tokens,
    )

    prediction_records: list[dict[str, Any]] = []
    display_name = args.display_name
    for sample in tqdm(samples, desc=f"Evaluating ({display_name.lower()})"):
        result = runner.generate_with_stats(sample["messages"])
        score_result = score_prediction(
            scorer_name=sample["scorer"],
            prediction=result.text,
            expected=sample.get("expected"),
        )
        prediction_records.append(
            {
                "id": sample["id"],
                "category": sample["category"],
                "messages": sample["messages"],
                "prediction": result.text,
                "expected": sample.get("expected"),
                "scorer": sample["scorer"],
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "latency_ms": result.latency_ms,
                **score_result,
            }
        )

    metrics = aggregate_metrics(prediction_records)
    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "predictions.jsonl", prediction_records)
    write_json(output_dir / "metrics.json", metrics)
    print("\n" + "=" * 60)
    print(f"{display_name.upper()} EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Total:    {metrics['total']}")
    print(f"Passed:   {metrics['passed']}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Latency:  {metrics['average_latency_ms']:.1f} ms/sample")
    for category, stats in metrics["by_category"].items():
        print(f"{category:12s} {stats['passed']}/{stats['total']} ({stats['accuracy']:.2%})")
    print(f"Predictions: {output_dir / 'predictions.jsonl'}")
    print(f"Metrics:     {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
