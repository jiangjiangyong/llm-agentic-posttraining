from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from llm_posttrain.agent.protocol import ToolCallParser
from llm_posttrain.evaluation.scorers import score_prediction


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


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_json(text: str) -> Any:
    return json.loads(text.strip())


def score_candidate(source: dict[str, Any], text: str) -> dict[str, Any]:
    category = source["category"]
    gold = source["gold"]
    if category == "tool_calling":
        parsed = ToolCallParser().parse(text)
        if parsed.kind != "tool_call" or parsed.tool_call is None:
            return {"passed": False, "score": 0.0, "reason": parsed.error or "not a tool call"}
        expected = gold["tool_call"]
        actual = parsed.tool_call.to_dict()
        if actual != expected:
            return {"passed": False, "score": 0.0, "reason": f"tool call mismatch: {actual}"}
        return {"passed": True, "score": 1.0, "reason": "canonical tool call and arguments"}
    if category == "code":
        result = score_prediction("python_syntax", text, gold["expected"])
        return {"passed": bool(result["passed"]), "score": result["score"], "reason": result["detail"]}
    if category == "json":
        try:
            actual = strict_json(text)
        except (TypeError, json.JSONDecodeError) as exc:
            return {"passed": False, "score": 0.0, "reason": f"invalid JSON: {exc}"}
        passed = actual == gold["expected"]
        return {
            "passed": passed,
            "score": 1.0 if passed else 0.0,
            "reason": "exact JSON object" if passed else f"JSON mismatch: {actual}",
        }
    if category == "math":
        passed = text.strip() == str(gold["expected"]).strip()
        return {"passed": passed, "score": 1.0 if passed else 0.0, "reason": "exact number" if passed else "number mismatch"}
    raise ValueError(f"Unknown preference category: {category}")


def make_rule_negative(source: dict[str, Any], index: int) -> str:
    category = source["category"]
    gold = source["gold"]
    if category == "tool_calling":
        payload = json.dumps(gold["tool_call"], ensure_ascii=False, separators=(",", ":"))
        if index % 2 == 0:
            return f"<tool_call>{payload[:-1]}</tool_call>"
        wrong = dict(gold["tool_call"]["arguments"])
        expression = str(wrong["expression"])
        wrong["expression"] = f"({expression}) + 1"
        bad_call = {"name": gold["tool_call"]["name"], "arguments": wrong}
        return f"<tool_call>{json.dumps(bad_call, ensure_ascii=False, separators=(',', ':'))}</tool_call>"
    if category == "code":
        name = str(gold["expected"]["required_symbols"][0])
        return f"def unrelated_{name}():\n    return None"
    if category == "json":
        if index % 2 == 0:
            return json.dumps(gold["expected"], ensure_ascii=False, separators=(",", ":"))[:-1]
        bad = dict(gold["expected"])
        priority = str(bad.get("priority", "medium"))
        bad["priority"] = {"low": "medium", "medium": "high", "high": "low"}.get(priority, "low")
        return json.dumps(bad, ensure_ascii=False, separators=(",", ":"))
    if category == "math":
        expected = str(gold["expected"]).strip()
        return "1" if expected == "0" else "0"
    raise ValueError(f"Unknown preference category: {category}")


def load_candidates(paths: list[str]) -> dict[str, list[dict[str, Any]]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        for candidate in read_jsonl(path):
            source_id = str(candidate.get("source_id", ""))
            text = candidate.get("candidate")
            if not source_id or not isinstance(text, str):
                raise ValueError(f"Malformed candidate in {path}: {candidate}")
            candidate["candidate_file"] = path
            by_source[source_id].append(candidate)
    return by_source


def choose_rejected(
    source: dict[str, Any],
    candidates: list[dict[str, Any]],
    index: int,
) -> tuple[str, dict[str, Any]]:
    for candidate in candidates:
        quality = score_candidate(source, str(candidate["candidate"]))
        if not quality["passed"] and str(candidate["candidate"]).strip():
            return str(candidate["candidate"]), {
                "source": candidate.get("backend", "model"),
                "candidate_file": candidate.get("candidate_file"),
                "quality": quality,
                "latency_ms": candidate.get("latency_ms"),
            }
    negative = make_rule_negative(source, index)
    quality = score_candidate(source, negative)
    if quality["passed"]:
        raise ValueError(f"Rule negative unexpectedly passed for {source['id']}")
    return negative, {"source": "rule_negative", "quality": quality}


def split_stratified(records: list[dict[str, Any]], ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < ratio < 0.5:
        raise ValueError("validation ratio must be between 0 and 0.5")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["category"])].append(record)
    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    for category in sorted(groups):
        group = list(groups[category])
        rng.shuffle(group)
        valid_count = max(1, round(len(group) * ratio))
        valid.extend(group[:valid_count])
        train.extend(group[valid_count:])
    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def check_eval_overlap(records: list[dict[str, Any]], eval_paths: list[str]) -> dict[str, str]:
    eval_prompts: set[str] = set()
    hashes: dict[str, str] = {}
    for path in eval_paths:
        hashes[path] = file_sha256(path)
        for item in read_jsonl(path):
            for message in item.get("messages", []):
                if message.get("role") == "user":
                    eval_prompts.add(normalize_text(str(message.get("content", ""))))
    source_prompts = {
        normalize_text(str(message.get("content", "")))
        for record in records
        for message in record.get("messages", [])
        if message.get("role") == "user"
    }
    overlap = sorted(eval_prompts & source_prompts)
    if overlap:
        raise ValueError(f"Preference source overlaps frozen evaluation prompts: {overlap}")
    return hashes


def build_pairs(source_records: list[dict[str, Any]], candidate_map: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for index, source in enumerate(source_records):
        chosen = str(source["gold"]["response"])
        chosen_quality = score_candidate(source, chosen)
        if not chosen_quality["passed"]:
            raise ValueError(f"Gold response failed validation for {source['id']}: {chosen_quality}")
        rejected, rejection_meta = choose_rejected(source, candidate_map.get(str(source["id"]), []), index)
        rejected_quality = score_candidate(source, rejected)
        if rejected_quality["passed"]:
            raise ValueError(f"Rejected response passed validation for {source['id']}")
        pairs.append(
            {
                "id": f"dpo_{source['id']}",
                "category": source["category"],
                "prompt": source["messages"],
                "chosen": chosen,
                "rejected": rejected,
                "metadata": {
                    "source_id": source["id"],
                    "chosen_source": "verified_oracle",
                    "chosen_quality": chosen_quality,
                    "rejected_source": rejection_meta["source"],
                    "rejected_quality": rejected_quality,
                    "candidate_file": rejection_meta.get("candidate_file"),
                },
            }
        )
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build validated DPO preference pairs.")
    parser.add_argument("--source", default="data/preference/source.jsonl")
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--train-output", default="data/preference/train.jsonl")
    parser.add_argument("--valid-output", default="data/preference/valid.jsonl")
    parser.add_argument("--manifest-output", default="artifacts/preference_manifest.json")
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument(
        "--eval-dataset",
        action="append",
        default=["data/evaluation/base_smoke.jsonl", "data/evaluation/tool_calling_smoke.jsonl"],
    )
    args = parser.parse_args()
    source_records = read_jsonl(args.source)
    if not source_records:
        raise ValueError(f"No source records found in {args.source}")
    eval_hashes = check_eval_overlap(source_records, args.eval_dataset)
    candidate_map = load_candidates(args.candidate)
    pairs = build_pairs(source_records, candidate_map)
    train, valid = split_stratified(pairs, args.validation_ratio, args.seed)
    write_jsonl(args.train_output, train)
    write_jsonl(args.valid_output, valid)
    rejection_sources = Counter(str(pair["metadata"]["rejected_source"]) for pair in pairs)
    manifest = {
        "version": "preference_pairs_v1",
        "seed": args.seed,
        "source": args.source,
        "source_count": len(source_records),
        "pair_count": len(pairs),
        "train_count": len(train),
        "validation_count": len(valid),
        "category_counts": dict(sorted(Counter(pair["category"] for pair in pairs).items())),
        "rejection_source_counts": dict(sorted(rejection_sources.items())),
        "candidate_files": args.candidate,
        "train_output": args.train_output,
        "validation_output": args.valid_output,
        "evaluation_datasets": eval_hashes,
        "evaluation_prompt_overlap_count": 0,
        "contract": "prompt/chosen/rejected; chosen passes task validator and rejected fails it",
    }
    output_path = Path(args.manifest_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
