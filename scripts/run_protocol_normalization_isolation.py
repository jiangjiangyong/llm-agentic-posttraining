from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from llm_posttrain.agent.protocol_normalizer import CanonicalProtocolNormalizer

try:
    from scripts.data_access_policy import assert_no_final_test_input
except ModuleNotFoundError:
    from data_access_policy import assert_no_final_test_input


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_outputs(record: dict[str, Any]) -> Iterable[str]:
    steps = record.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if isinstance(step, dict) and step.get("model_output") is not None:
                yield str(step["model_output"])
        return

    trace = record.get("trace")
    if isinstance(trace, dict):
        for event in trace.get("events", []):
            if not isinstance(event, dict) or event.get("kind") != "model_output":
                continue
            payload = event.get("payload", {})
            if isinstance(payload, dict) and payload.get("text") is not None:
                yield str(payload["text"])

    if record.get("model_output") is not None:
        yield str(record["model_output"])


def episode_semantic_success(record: dict[str, Any]) -> bool:
    if "semantic_success" in record:
        return bool(record["semantic_success"])
    if "success" in record:
        return bool(record["success"])
    trace = record.get("trace")
    if isinstance(trace, dict) and "success" in trace:
        return bool(trace["success"])
    return False


def analyze(records: list[dict[str, Any]], input_path: str | Path) -> dict[str, Any]:
    normalizer = CanonicalProtocolNormalizer()
    status_counts: Counter[str] = Counter()
    model_output_count = 0
    parsed_tool_output_count = 0
    raw_canonical_count = 0
    normalization_applied_count = 0
    normalized_canonical_count = 0
    episode_with_tool_output = 0
    raw_strict_episode_success = 0
    normalized_strict_episode_success = 0
    semantic_success_count = 0
    per_episode: list[dict[str, Any]] = []

    for record in records:
        outputs = list(model_outputs(record))
        results = [normalizer.normalize(output) for output in outputs]
        tool_results = [result for result in results if result.parsed_kind == "tool_call"]
        semantic_success = episode_semantic_success(record)
        semantic_success_count += int(semantic_success)
        model_output_count += len(results)
        parsed_tool_output_count += len(tool_results)
        raw_canonical_count += sum(
            int(normalizer.is_canonical(result.raw_text)) for result in tool_results
        )
        normalization_applied_count += sum(int(result.applied) for result in tool_results)
        normalized_canonical_count += sum(
            int(result.canonical_available) for result in tool_results
        )
        status_counts.update(result.status for result in results)

        has_tool_output = bool(tool_results)
        if has_tool_output:
            episode_with_tool_output += 1
            raw_strict = all(normalizer.is_canonical(result.raw_text) for result in tool_results)
            normalized_strict = all(result.canonical_available for result in tool_results)
            raw_strict_episode_success += int(semantic_success and raw_strict)
            normalized_strict_episode_success += int(semantic_success and normalized_strict)
        per_episode.append(
            {
                "id": record.get("task_id", record.get("episode_id", record.get("id"))),
                "semantic_success": semantic_success,
                "model_output_count": len(results),
                "parsed_tool_output_count": len(tool_results),
                "raw_canonical": all(normalizer.is_canonical(result.raw_text) for result in tool_results)
                if tool_results
                else False,
                "normalized_canonical": all(result.canonical_available for result in tool_results)
                if tool_results
                else False,
                "statuses": [result.status for result in results],
            }
        )

    def rate(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    return {
        "version": "protocol_normalization_isolation_v1",
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "model_inference": False,
        "final_test_model_inference": False,
        "episodes": len(records),
        "model_output_count": model_output_count,
        "parsed_tool_output_count": parsed_tool_output_count,
        "raw_canonical": {"n": raw_canonical_count, "denominator": parsed_tool_output_count, "rate": rate(raw_canonical_count, parsed_tool_output_count)},
        "normalization_applied": {"n": normalization_applied_count, "denominator": parsed_tool_output_count, "rate": rate(normalization_applied_count, parsed_tool_output_count)},
        "normalized_canonical": {"n": normalized_canonical_count, "denominator": parsed_tool_output_count, "rate": rate(normalized_canonical_count, parsed_tool_output_count)},
        "semantic_success": {"n": semantic_success_count, "denominator": len(records), "rate": rate(semantic_success_count, len(records))},
        "episode_with_tool_output": episode_with_tool_output,
        "raw_strict_episode_success": {"n": raw_strict_episode_success, "denominator": episode_with_tool_output, "rate": rate(raw_strict_episode_success, episode_with_tool_output)},
        "normalized_strict_episode_success": {"n": normalized_strict_episode_success, "denominator": episode_with_tool_output, "rate": rate(normalized_strict_episode_success, episode_with_tool_output)},
        "status_counts": dict(sorted(status_counts.items())),
        "episodes_detail": per_episode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure protocol normalization without model inference.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert_no_final_test_input(args.input, "protocol normalization isolation")
    summary = analyze(read_jsonl(args.input), args.input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "episodes_detail"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
