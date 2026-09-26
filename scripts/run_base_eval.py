from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from llm_posttrain.config import load_yaml
from llm_posttrain.evaluation.scorers import score_prediction
from llm_posttrain.models.loader import build_runner


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc
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
    output_path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "passed": 0})
    for record in records:
        stats = by_category[str(record["category"])]
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
    latency_values = [float(record["latency_ms"]) for record in records]
    return {
        "total": total,
        "passed": passed,
        "accuracy": passed / total if total else 0.0,
        "average_latency_ms": sum(latency_values) / len(latency_values) if latency_values else 0.0,
        "by_category": category_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_model.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    evaluation_config = config["evaluation"]
    samples = read_jsonl(evaluation_config["dataset_path"])
    print(f"Loaded {len(samples)} evaluation samples")

    runner = build_runner(config)
    prediction_records: list[dict[str, Any]] = []
    for sample in tqdm(samples, desc="Evaluating"):
        started = time.perf_counter()
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
                "runner_wall_time_ms": (time.perf_counter() - started) * 1000,
                **score_result,
            }
        )

    metrics = aggregate_metrics(prediction_records)
    write_jsonl(evaluation_config["predictions_path"], prediction_records)
    write_json(evaluation_config["metrics_path"], metrics)

    print("\n" + "=" * 60)
    print("BASELINE EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Total:    {metrics['total']}")
    print(f"Passed:   {metrics['passed']}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Latency:  {metrics['average_latency_ms']:.1f} ms/sample")
    for category, stats in metrics["by_category"].items():
        print(f"{category:12s} {stats['passed']}/{stats['total']} ({stats['accuracy']:.2%})")
    print(f"Predictions: {evaluation_config['predictions_path']}")
    print(f"Metrics:     {evaluation_config['metrics_path']}")


if __name__ == "__main__":
    main()
