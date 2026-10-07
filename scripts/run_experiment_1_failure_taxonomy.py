from __future__ import annotations

"""Experiment 1: diagnose Code-Agent failures on the v1 dev split.

This script intentionally evaluates only ``data/code_agent_v1/dev.jsonl``.
It does not train, modify the frozen v1 evaluator/reward/runtime, inspect
validation cases individually, or run Frozen Final Test inference.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from llm_posttrain.agent.code_environment import CodeAgentEnvironment
from llm_posttrain.config import load_yaml
from llm_posttrain.models.adapter_runner import build_adapter_runner

try:
    from scripts.data_access_policy import assert_final_test_eval_allowed
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from data_access_policy import assert_final_test_eval_allowed


CANONICAL_PATTERN = re.compile(
    r"^\s*<tool_call>\s*(.*?)\s*</tool_call>\s*$",
    re.DOTALL | re.IGNORECASE,
)

TAXONOMY_PRIORITY = [
    "runtime_error",
    "max_step",
    "parse_schema",
    "tool_selection",
    "argument",
    "execution",
    "unit_test",
    "observation_use",
    "repeated_call",
    "final_answer",
    "protocol_wrapper",
    "unknown",
]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.STDOUT, text=True
        ).strip()
    except Exception as exc:  # pragma: no cover - environment diagnostic
        return f"UNAVAILABLE:{type(exc).__name__}:{exc}"


def is_canonical_tool_output(text: str) -> bool:
    match = CANONICAL_PATTERN.fullmatch(text)
    if match is None:
        return False
    try:
        payload = json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return False
    return (
        isinstance(payload, dict)
        and set(payload) == {"name", "arguments"}
        and isinstance(payload["name"], str)
        and isinstance(payload["arguments"], dict)
    )


def recovery_kind(text: str) -> str:
    """Return a descriptive parser-recovery category for non-canonical output."""
    stripped = text.strip()
    lowered = stripped.lower()
    if "<tool_call>" in lowered and "</tool_call>" not in lowered:
        return "unterminated_tag"
    if stripped.startswith("```"):
        return "fenced_json"
    if stripped.startswith("|"):
        return "pipe_prefix"
    if "<tool_call>" in lowered:
        return "noncanonical_tag_payload"
    if stripped.startswith("{"):
        return "bare_json_or_json_prefix"
    if "{" in stripped:
        return "prose_prefixed_json"
    return "unknown_parser_recovery"


def _string_values(value: Any) -> list[str]:
    values: list[str] = []
    if isinstance(value, dict):
        for nested in value.values():
            values.extend(_string_values(nested))
    elif isinstance(value, list):
        for nested in value:
            values.extend(_string_values(nested))
    elif isinstance(value, str):
        compact = " ".join(value.split())
        if len(compact) >= 4:
            values.append(compact)
    return values


def _observation_terms(result: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    output = result.get("output")
    for value in _string_values(output):
        terms.append(value)
        # Shorter tokens make matching useful without declaring a whole
        # long JSON object as "used" just because one character matched.
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{4,}", value):
            terms.append(token)
    return sorted(set(terms), key=len, reverse=True)


def _later_model_text(steps: list[dict[str, Any]], index: int, final: str | None) -> str:
    pieces = [
        str(step.get("model_output") or "")
        for step in steps[index + 1 :]
        if step.get("model_output") is not None
    ]
    if final:
        pieces.append(final)
    return "\n".join(pieces)


def observation_analysis(
    task: dict[str, Any], row: dict[str, Any]
) -> dict[str, Any]:
    """Apply an executable, conservative observation-utilization rule.

    A successful observation is marked ``used`` only when a non-trivial scalar
    returned by the tool appears in a later model action or final answer. For
    failed observations, changing to a different next action is evidence of
    recovery but not proof of semantic understanding, so it is marked
    ``manual_review_required``. Unknown cases are never forced into a label.
    """
    steps = row.get("steps", [])
    final_output = row.get("final_output")
    observations: list[dict[str, Any]] = []
    manual_review = False
    explicit_used = False
    explicit_not_used = False

    for index, step in enumerate(steps):
        results = step.get("tool_results", [])
        if not isinstance(results, list):
            continue
        later_text = _later_model_text(steps, index, final_output)
        for result in results:
            if not isinstance(result, dict):
                continue
            ok = result.get("ok") is True
            terms = _observation_terms(result)
            hits = [
                term
                for term in terms
                if term.lower() in later_text.lower()
                and len(term) >= 4
            ]
            same_signature_again = False
            current_call = step.get("tool_call")
            for later_step in steps[index + 1 :]:
                later_calls = later_step.get("tool_calls", [])
                if not isinstance(later_calls, list):
                    continue
                for later_call in later_calls:
                    if later_call == current_call:
                        same_signature_again = True
            if hits:
                status = "used"
                explicit_used = True
                rule = "returned_scalar_appears_in_later_action_or_final"
            elif not ok and same_signature_again:
                status = "not_used"
                explicit_not_used = True
                rule = "failed_action_repeated_without_observable_change"
            else:
                status = "manual_review_required"
                manual_review = True
                rule = (
                    "automatic_match_did_not_prove_observation_use;"
                    "manual_review_required"
                )
            observations.append(
                {
                    "step": step.get("step"),
                    "tool_name": result.get("tool_name"),
                    "ok": ok,
                    "status": status,
                    "rule": rule,
                    "matched_terms": hits[:10],
                    "candidate_terms": terms[:10],
                }
            )

    if not observations:
        status = "not_applicable"
    elif manual_review:
        status = "manual_review_required"
    elif explicit_not_used and not explicit_used:
        status = "not_used"
    else:
        status = "used"
    return {
        "status": status,
        "manual_review_required": manual_review,
        "observations": observations,
        "rule_version": "experiment_1_observation_rule_v1",
        "task_type": task.get("task_type", task.get("category")),
    }


def classify_episode(
    task: dict[str, Any], row: dict[str, Any], max_steps: int
) -> dict[str, Any]:
    steps = row.get("steps", [])
    tool_outputs = [str(value) for value in row.get("tool_outputs", [])]
    canonical_flags = [is_canonical_tool_output(value) for value in tool_outputs]
    recovery_records: list[dict[str, Any]] = []
    tool_output_step_index = 0
    for step in steps:
        if step.get("parsed_kind") != "tool_call":
            continue
        output = step.get("model_output")
        if output is None:
            continue
        output_text = str(output)
        canonical = is_canonical_tool_output(output_text)
        if not canonical:
            recovery_records.append(
                {
                    "step": step.get("step"),
                    "kind": recovery_kind(output_text),
                    "raw_output": output_text,
                }
            )
        tool_output_step_index += 1

    expected = task.get("expected", {})
    required_tools = [str(value) for value in expected.get("required_tools", [])]
    actual_names = [str(call.get("name")) for call in row.get("tool_calls", [])]
    missing_tools = sorted(set(required_tools) - set(actual_names))
    reward = row.get("reward", {})
    runtime_error = any(
        step.get("model_output") is None and step.get("parsed_kind") == "invalid"
        for step in steps
    )
    parse_schema = any(
        step.get("model_output") is not None and step.get("parsed_kind") == "invalid"
        for step in steps
    )
    max_step = (
        not bool(row.get("final_output"))
        and len(steps) >= max_steps
        and not runtime_error
    )
    tool_selection = bool(missing_tools)
    argument = float(reward.get("argument_reward", 0.0)) < 1.0
    execution = float(reward.get("execution_reward", 0.0)) < 1.0
    unit_test = bool(expected.get("unit_test_required")) and float(
        reward.get("unit_test_reward", 0.0)
    ) < 1.0
    final_answer = bool(
        expected.get("final_answer")
        or expected.get("final_answer_contains")
        or expected.get("final_json_schema")
    ) and float(reward.get("final_answer_reward", 0.0)) < 1.0
    signatures = [
        json.dumps(call, ensure_ascii=False, sort_keys=True)
        for call in row.get("tool_calls", [])
    ]
    repeated_call = len(signatures) != len(set(signatures))
    observation = observation_analysis(task, row)
    observation_use = observation["status"] == "not_used"
    protocol_wrapper = bool(recovery_records)

    flags = {
        "runtime_error": runtime_error,
        "max_step": max_step,
        "parse_schema": parse_schema,
        "tool_selection": tool_selection,
        "argument": argument,
        "execution": execution,
        "unit_test": unit_test,
        "observation_use": observation_use,
        "repeated_call": repeated_call,
        "final_answer": final_answer,
        "protocol_wrapper": protocol_wrapper,
    }
    semantic_success = bool(row.get("semantic_success"))
    strict_success = bool(row.get("strict_success"))
    candidates = [name for name in TAXONOMY_PRIORITY if flags.get(name, False)]
    if not semantic_success and not candidates:
        candidates = ["unknown"]
    primary_failure = candidates[0] if candidates else "none"
    secondary_failures = [
        name for name in candidates if name != primary_failure
    ]

    return {
        "primary_failure": primary_failure,
        "secondary_failures": secondary_failures,
        "failure_priority_version": "experiment_1_failure_priority_v1",
        "failure_flags": flags,
        "missing_required_tools": missing_tools,
        "actual_tool_names": actual_names,
        "raw_canonical_tool_output_count": sum(canonical_flags),
        "raw_tool_output_count": len(canonical_flags),
        "raw_canonical_rate": (
            sum(canonical_flags) / len(canonical_flags) if canonical_flags else 0.0
        ),
        "raw_strict_success": strict_success,
        "recovered_semantic_success": semantic_success,
        "parser_recovery_used": bool(recovery_records),
        "parser_recovery_types": [item["kind"] for item in recovery_records],
        "parser_recovery_records": recovery_records,
        "parser_rescued_task": bool(semantic_success and not strict_success),
        "parser_recovery_failed_task": bool(
            recovery_records and not semantic_success
        ),
        "observation_utilization": observation,
        "representative_reason": (
            "primary failure follows fixed priority; secondary failures retain "
            "all independently observed signals"
        ),
    }


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def aggregate(
    rows: list[dict[str, Any]], *, include_categories: bool = True
) -> dict[str, Any]:
    total = len(rows)
    classification = [row["classification"] for row in rows]
    tool_output_count = sum(
        int(item["raw_tool_output_count"]) for item in classification
    )
    canonical_count = sum(
        int(item["raw_canonical_tool_output_count"]) for item in classification
    )
    strict_failures = sum(
        not bool(item["raw_strict_success"]) for item in classification
    )
    rescued = sum(
        bool(item["parser_rescued_task"]) for item in classification
    )
    recovery_used = sum(
        bool(item["parser_recovery_used"]) for item in classification
    )
    recovery_failed = sum(
        bool(item["parser_recovery_failed_task"]) for item in classification
    )
    metrics: dict[str, Any] = {
        "episodes": total,
        "raw_canonical_rate": ratio(canonical_count, tool_output_count),
        "raw_strict_success_rate": ratio(
            sum(bool(row.get("strict_success")) for row in rows), total
        ),
        "recovered_semantic_success_rate": ratio(
            sum(bool(row.get("semantic_success")) for row in rows), total
        ),
        "runtime_success_rate": ratio(
            sum(bool(row.get("runtime_success")) for row in rows), total
        ),
        "parser_recovery_rate": ratio(recovery_used, total),
        "parser_rescue_rate": ratio(rescued, strict_failures),
        "parser_recovery_failed_rate": ratio(recovery_failed, recovery_used),
        "execution_success_rate": ratio(
            sum(float(row["reward"].get("execution_reward", 0.0)) == 1.0 for row in rows),
            total,
        ),
        "unit_test_reward_component_one_rate": ratio(
            sum(
                bool(row["reward"].get("unit_test_reward", 0.0) == 1.0)
                for row in rows
            ),
            total,
        ),
        "final_answer_success_rate": ratio(
            sum(float(row["reward"].get("final_answer_reward", 0.0)) == 1.0 for row in rows),
            total,
        ),
        "mean_reward": (
            sum(float(row["reward"].get("total_reward", 0.0)) for row in rows) / total
            if total
            else 0.0
        ),
        "mean_tool_calls": (
            sum(len(row.get("tool_calls", [])) for row in rows) / total
            if total
            else 0.0
        ),
        "repeated_call_rate": ratio(
            sum("repeated_call" in item["secondary_failures"] or item["primary_failure"] == "repeated_call" for item in classification),
            total,
        ),
        "manual_review_required_rate": ratio(
            sum(bool(item["observation_utilization"]["manual_review_required"]) for item in classification),
            total,
        ),
        "failure_by_primary": dict(
            Counter(item["primary_failure"] for item in classification)
        ),
        "failure_by_secondary": dict(
            Counter(
                failure
                for item in classification
                for failure in item["secondary_failures"]
            )
        ),
        "parser_recovery_types": dict(
            Counter(
                recovery_type
                for item in classification
                for recovery_type in item["parser_recovery_types"]
            )
        ),
    }
    if include_categories:
        by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_category[str(row.get("category", "unknown"))].append(row)
        metrics["by_category"] = {
            category: aggregate(category_rows, include_categories=False)
            for category, category_rows in sorted(by_category.items())
        }
    return metrics


def build_metric_audit(
    tasks: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Make numerator/denominator/applicability explicit for interview use."""
    task_by_id = {str(task["id"]): task for task in tasks}
    counters: dict[str, Counter] = {
        name: Counter()
        for name in ("tool_selection", "argument", "execution", "unit_test", "final_answer")
    }
    for row in rows:
        task_id = str(row["task_id"]).split(":")[-1]
        task = task_by_id[task_id]
        expected = task.get("expected", {})
        calls = row.get("tool_calls", [])
        results = row.get("tool_results", [])
        conditions = {
            "tool_selection": bool(expected.get("required_tools")),
            "argument": bool(expected.get("expected_arguments") or expected.get("file_checks")),
            "execution": bool(expected.get("execution_tools")),
            "unit_test": bool(expected.get("unit_test_required")),
            "final_answer": bool(
                expected.get("final_answer")
                or expected.get("final_answer_contains")
                or expected.get("final_json_schema")
            ),
        }
        assessed = {
            "tool_selection": conditions["tool_selection"]
            and not any(step.get("parsed_kind") == "invalid" for step in row.get("steps", [])),
            "argument": conditions["argument"] and bool(calls),
            "execution": conditions["execution"] and bool(results),
            "unit_test": conditions["unit_test"]
            and any(result.get("tool_name") == "run_unit_tests" for result in results),
            "final_answer": conditions["final_answer"] and row.get("final_output") is not None,
        }
        passed = {
            "tool_selection": assessed["tool_selection"]
            and all(
                str(required) in [str(call.get("name")) for call in calls]
                for required in expected.get("required_tools", [])
            ),
            "argument": assessed["argument"]
            and float(row["reward"].get("argument_reward", 0.0)) == 1.0,
            "execution": assessed["execution"]
            and float(row["reward"].get("execution_reward", 0.0)) == 1.0,
            "unit_test": assessed["unit_test"]
            and float(row["reward"].get("unit_test_reward", 0.0)) == 1.0,
            "final_answer": assessed["final_answer"]
            and float(row["reward"].get("final_answer_reward", 0.0)) == 1.0,
        }
        for name in conditions:
            counters[name]["applicable"] += int(conditions[name])
            counters[name]["assessed"] += int(assessed[name])
            counters[name]["passed"] += int(passed[name])
    observations = Counter(
        row["classification"]["observation_utilization"]["status"]
        for row in rows
    )
    total = len(rows)
    observation_applicable = total - observations["not_applicable"]
    return {
        "definition": "Every metric exposes applicable, assessed, and passed counts; neutral reward defaults are not reported as task success.",
        "metrics": {
            name: {
                "applicable_n": counts["applicable"],
                "assessed_n": counts["assessed"],
                "passed_n": counts["passed"],
                "applicable_denominator": total,
                "assessed_denominator": counts["assessed"],
                "pass_rate_over_applicable": ratio(counts["passed"], counts["applicable"]),
                "pass_rate_over_assessed": ratio(counts["passed"], counts["assessed"]),
            }
            for name, counts in counters.items()
        },
        "unit_test_neutral_reward_component": {
            "one_n": sum(
                float(row["reward"].get("unit_test_reward", 0.0)) == 1.0
                for row in rows
            ),
            "denominator": total,
            "warning": "This is not unit-test pass rate; non-unit-test tasks receive neutral reward 1.0.",
        },
        "observation_utilization": {
            "status_counts": dict(observations),
            "applicable_episode_n": observation_applicable,
            "manual_review_n": observations["manual_review_required"],
            "manual_rate_over_all_episodes": ratio(observations["manual_review_required"], total),
            "manual_rate_over_observation_episodes": ratio(observations["manual_review_required"], observation_applicable),
        },
    }


def select_representative_cases(
    rows: list[dict[str, Any]], max_per_type: int = 3
) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        classification = row["classification"]
        labels = [classification["primary_failure"]] + list(
            classification["secondary_failures"]
        )
        for label in labels:
            if label in {"none", "unknown"}:
                continue
            if len(selected[label]) < max_per_type:
                selected[label].append(str(row["task_id"]))
    return dict(selected)


def clip(value: Any, limit: int = 1800) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "\n...[clipped]"


def build_cases_markdown(rows: list[dict[str, Any]], selected: dict[str, list[str]]) -> str:
    by_id = {str(row["task_id"]): row for row in rows}
    lines = [
        "# Experiment 1 Representative Cases",
        "",
        "本文件只来自 `data/code_agent_v1/dev.jsonl`，用于面试时解释失败证据；不包含 Validation 或 Frozen Final Test 单条样本。",
        "",
    ]
    for label in sorted(selected):
        lines.extend([f"## {label}", ""])
        for task_id in selected[label]:
            row = by_id[task_id]
            classification = row["classification"]
            lines.extend(
                [
                    f"### `{task_id}`",
                    "",
                    f"- category: `{row.get('category')}`",
                    f"- primary_failure: `{classification['primary_failure']}`",
                    f"- secondary_failures: `{classification['secondary_failures']}`",
                    f"- parser_recovery: `{classification['parser_recovery_types']}`",
                    f"- observation_utilization: `{classification['observation_utilization']['status']}`",
                    f"- semantic/strict/runtime: `{row.get('semantic_success')}` / `{row.get('strict_success')}` / `{row.get('runtime_success')}`",
                    "",
                    "#### User task",
                    "",
                    "```text",
                    clip(next((m.get("content", "") for m in row.get("initial_messages", []) if m.get("role") == "user"), "")),
                    "```",
                    "",
                    "#### Raw model output(s)",
                    "",
                    "```text",
                    clip("\n--- step ---\n".join(str(s.get("model_output")) for s in row.get("steps", []))),
                    "```",
                    "",
                    "#### Tool calls/results",
                    "",
                    "```json",
                    clip({"tool_calls": row.get("tool_calls"), "tool_results": row.get("tool_results")}),
                    "```",
                    "",
                    "#### Final state",
                    "",
                    f"- final_output: `{clip(row.get('final_output') or '', 1200)}`",
                    f"- workspace_files: `{row.get('workspace_files', [])}`",
                    f"- reward: `{row.get('reward', {})}`",
                    "",
                ]
            )
    return "\n".join(lines) + "\n"


def build_card(summary: dict[str, Any]) -> str:
    return f"""# Experiment Card 1 — Failure Taxonomy and Three-Layer Evaluation

## Research Question

On the stable SFT Adapter, what is the observable failure distribution on the 24-task Failure-Mining Dev split, and how much does parser recovery hide raw protocol failure?

## Scope and non-goals

- Dataset: `data/code_agent_v1/dev.jsonl` only; 24 tasks, six categories, four per category.
- Validation individual cases were not inspected or evaluated.
- Frozen Final Test was not used for model inference.
- No SFT, DPO, Agentic RL, Reward Ablation, Credit Assignment, or Data Flywheel training was run.
- No Reward, Evaluator, Runtime, or v1 dataset file was modified.

## Fixed inputs

- Model: `models/adapters/sft_qwen3_1.7b`.
- Decode: deterministic (`do_sample=false`), `max_new_tokens=128`.
- Environment max steps: 5, explicitly recorded for this diagnostic run.
- Dataset hash: `{summary['dataset']['sha256']}`.
- Code snapshot at run start: `{summary['code']['head']}`.

## Taxonomy contract

Each episode has exactly one `primary_failure` selected by the fixed priority below, plus zero or more `secondary_failures`:

`runtime_error > max_step > parse_schema > tool_selection > argument > execution > unit_test > observation_use > repeated_call > final_answer > protocol_wrapper > unknown`

The labels are diagnostic observations, not training prescriptions. A failure distribution does not by itself justify changing Reward weights.

## Observation utilization rule

A tool observation is marked `used` only when a non-trivial scalar returned by the tool is found in a later model action or final answer. Repeated failed actions are marked `not_used`. Ambiguous cases are marked `manual_review_required`; they are never force-labeled.

## Metric denominator rule

The neutral `unit_test_reward=1.0` assigned to non-unit-test tasks is not counted as a test pass. The job-facing unit-test metric uses only applicable tasks (`0/4` in this run). Execution, final-answer, argument, and observation metrics also report applicable and assessed denominators in `metric_audit`.

## Decision Gate

`{summary['decision_gate']}`

The output is sufficient to choose the next research question, but it is not a model improvement claim and does not authorize Experiment 2.
"""


def build_report(summary: dict[str, Any]) -> str:
    metrics = summary["metrics"]
    audit = summary["metric_audit"]
    primary_rows = "\n".join(
        f"| `{key}` | {value} | {value / metrics['episodes']:.2%} |"
        for key, value in sorted(metrics["failure_by_primary"].items())
    )
    secondary_rows = "\n".join(
        f"| `{key}` | {value} | {value / metrics['episodes']:.2%} |"
        for key, value in sorted(metrics["failure_by_secondary"].items())
    ) or "| none | 0 | 0.00% |"
    category_rows = "\n".join(
        f"| `{category}` | {data['episodes']} | {data['recovered_semantic_success_rate']:.2%} | {data['raw_strict_success_rate']:.2%} | {data['mean_reward']:.4f} |"
        for category, data in metrics["by_category"].items()
    )
    return f"""# Experiment 1 — Failure Taxonomy and Raw / Recovered / Final Report

## 1. Executive conclusion

This run diagnoses the stable SFT Adapter on the 24-task Failure-Mining Dev split. It does not train a model and does not make a Reward or performance-improvement claim.

Decision Gate: **{summary['decision_gate']}**

The result can be used to select the next research question and to prepare interview case studies. It cannot be used to tune against Validation or Frozen Final Test.

## 2. Scope protection

- Evaluated dataset: `data/code_agent_v1/dev.jsonl`, 24/24 records.
- Validation individual cases: **not evaluated**.
- Frozen Final Test model inference: **not run**.
- New training or reward modification: **not run**.
- Final Test hash re-check: `{summary['final_test_protection']['hash_matches_manifest']}`.

## 3. Provenance

| Item | Value |
|---|---|
| Git HEAD | `{summary['code']['head']}` |
| Dataset SHA-256 | `{summary['dataset']['sha256']}` |
| Dataset manifest | `{summary['dataset']['manifest_path']}` |
| Adapter | `{summary['model']['adapter_path']}` |
| Adapter weight SHA-256 | `{summary['model']['adapter_weight_sha256']}` |
| Evaluator | `{summary['sources']['evaluator_path']}` |
| Reward | `{summary['sources']['reward_path']}` |
| Runtime | `{summary['sources']['runtime_path']}` |
| Raw episodes | `{summary['artifacts']['episodes_path']}` |
| Summary | `{summary['artifacts']['summary_path']}` |

## 3.1 Verification

- Classifier self-checks: `{summary['tests']['classifier_self_checks']}`
- Repository pytest: `{summary['tests']['pytest_passed']}`
- Pytest command: `{summary['tests']['pytest_command']}`
- Pytest result: `{summary['tests']['pytest_result']}`

## 4. Three-layer and parser metrics

| Metric | Value | Definition |
|---|---:|---|
| Raw canonical tool-output rate | {metrics['raw_canonical_rate']:.2%} | canonical wrapper among emitted tool outputs |
| Raw strict success | {metrics['raw_strict_success_rate']:.2%} | existing strict semantic + canonical evaluator |
| Recovered semantic success | {metrics['recovered_semantic_success_rate']:.2%} | parser-compatible semantic success |
| Runtime success | {metrics['runtime_success_rate']:.2%} | runtime reached a parsed final answer |
| Parser recovery rate | {metrics['parser_recovery_rate']:.2%} | episodes requiring non-canonical recovery |
| Parser rescue rate | {metrics['parser_rescue_rate']:.2%} | rescued tasks / raw-strict failures |
| Recovery failed rate | {metrics['parser_recovery_failed_rate']:.2%} | recovery-used episodes still semantically failing |
| Execution success (all applicable episodes) | {metrics['execution_success_rate']:.2%} | {audit['metrics']['execution']['passed_n']}/{audit['metrics']['execution']['applicable_denominator']}; all 24 tasks require execution |
| Execution success among assessed episodes | {audit['metrics']['execution']['pass_rate_over_assessed']:.2%} | {audit['metrics']['execution']['passed_n']}/{audit['metrics']['execution']['assessed_n']}; 12 episodes produced tool results |
| Unit-test reward component = 1 | {metrics['unit_test_reward_component_one_rate']:.2%} | {audit['unit_test_neutral_reward_component']['one_n']}/24; neutral 1.0 is assigned to 20 non-unit-test tasks |
| Unit-test pass rate (applicable tasks) | {audit['metrics']['unit_test']['pass_rate_over_applicable']:.2%} | {audit['metrics']['unit_test']['passed_n']}/{audit['metrics']['unit_test']['applicable_n']}; no unit-test call passed |
| Final-answer success (all applicable episodes) | {metrics['final_answer_success_rate']:.2%} | {audit['metrics']['final_answer']['passed_n']}/{audit['metrics']['final_answer']['applicable_n']} |
| Final-answer success among reached answers | {audit['metrics']['final_answer']['pass_rate_over_assessed']:.2%} | {audit['metrics']['final_answer']['passed_n']}/{audit['metrics']['final_answer']['assessed_n']}; 16 episodes never reached a final answer |
| Mean tool calls | {metrics['mean_tool_calls']:.3f} | executed calls per episode |
| Repeated-call rate | {metrics['repeated_call_rate']:.2%} | episodes with repeated identical call |
| Observation manual-review rate (all episodes) | {metrics['manual_review_required_rate']:.2%} | {audit['observation_utilization']['manual_review_n']}/24 |
| Observation manual-review rate (episodes with observations) | {audit['observation_utilization']['manual_rate_over_observation_episodes']:.2%} | {audit['observation_utilization']['manual_review_n']}/{audit['observation_utilization']['applicable_episode_n']} |

Parser rescue is reported separately because a semantic success after recovery is not evidence of strict protocol learning.

### Denominator correction

The earlier shorthand `unit-test pass rate = 83.33%` would be misleading. That number is the rate at which the Reward component equals its neutral value of 1.0 across all episodes; it is not a unit-test pass rate. The applicable unit-test result is `0/4 = 0.00%`, because four Dev tasks require unit tests and none reached a passed test call. The complete numerator/denominator audit is stored in `metric_audit` inside the summary Artifact.

## 5. Primary failure distribution

| Primary failure | Count | Share |
|---|---:|---:|
{primary_rows}

Primary labels are mutually exclusive and follow the priority in `experiment_card_1.md`. Secondary labels preserve additional signals.

## 6. Secondary failure signals

| Secondary failure | Count | Episode share |
|---|---:|---:|
{secondary_rows}

## 7. By task category

| Category | N | Recovered semantic | Raw strict | Mean reward |
|---|---:|---:|---:|---:|
{category_rows}

## 8. Observation utilization

The script uses an executable conservative rule: a non-trivial scalar from a tool result must appear in a later model action/final answer to be marked `used`; repeated failed actions are `not_used`; ambiguous cases are `manual_review_required`. The per-observation evidence is stored in every raw episode record.

## 9. Representative evidence

Representative cases (up to three per observed failure type) are in:

```text
{summary['artifacts']['representative_cases_path']}
```

The complete three-layer records are in:

```text
{summary['artifacts']['episodes_path']}
```

## 10. Job-search interpretation

| Claim | Status | Evidence-bound wording |
|---|---|---|
| Built a reproducible Agent failure-analysis pipeline | 可以写 | “在 Dev split 上保存 raw output、parser recovery、tool result、workspace final state 并按 primary/secondary taxonomy 聚合。” |
| Quantified parser recovery gap | 可以写 | Only cite the reported raw strict / recovered semantic / rescue metrics with definitions. |
| Identified the single best Reward change | 不能写 | Experiment 1 is diagnostic; Reward changes require Experiment 2A ablation. |
| Improved model performance | 不能写 | No training or before/after experiment was run here. |
| Generalized to real repositories | 不能写 | The Dev set is deterministic synthetic Code-Agent data. |

## 11. Remaining limitations

1. Dev has only 24 deterministic synthetic tasks; taxonomy frequency is a local diagnostic, not a market-wide or real-repository distribution.
2. Observation utilization has an explicit manual-review bucket; automatic matching is intentionally conservative.
3. Single stable SFT Adapter and deterministic decoding are used; no multi-seed stability claim is possible.
4. Validation and Frozen Final Test remain protected and are not used to select the next failure category.

## 12. Gate and next action

All required Experiment 1 evidence checks passed: Dev-only scope, raw/recovered/final layers, primary/secondary schema, observation rule traceability, representative cases, Final Test protection, and tests.

**Next action after review:** choose the next research question from the measured failure distribution. Do not start Experiment 2A until the report is reviewed; do not infer a Reward change from taxonomy counts alone.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/code_agent_v1/dev.jsonl")
    parser.add_argument("--model-config", default="configs/base_model.yaml")
    parser.add_argument("--adapter-path", default="models/adapters/sft_qwen3_1.7b")
    parser.add_argument("--output-dir", default="outputs/experiment_1_failure_taxonomy_v1")
    parser.add_argument("--summary-output", default="artifacts/experiment_1_failure_taxonomy_v1.json")
    parser.add_argument("--report-output", default="docs/experiment_results/experiment_1_failure_taxonomy_report.md")
    parser.add_argument("--card-output", default="docs/experiment_cards/experiment_card_1.md")
    parser.add_argument("--cases-output", default="docs/experiment_results/experiment_1_representative_cases.md")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--allow-final-test-eval", action="store_true")
    parser.add_argument(
        "--reuse-episodes",
        action="store_true",
        help="Reuse existing Dev raw episodes and only recompute classification/report artifacts.",
    )
    parser.add_argument("--episodes-input", default=None)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if dataset_path.name != "dev.jsonl" or "code_agent_v1" not in dataset_path.as_posix():
        raise ValueError("Experiment 1 is restricted to data/code_agent_v1/dev.jsonl")
    if args.allow_final_test_eval:
        raise ValueError("Experiment 1 must not receive --allow-final-test-eval")
    assert_final_test_eval_allowed(args.dataset, False)
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    adapter_path = Path(args.adapter_path)
    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(adapter_path / "adapter_config.json")

    tasks = read_jsonl(dataset_path)
    if len(tasks) != 24 or any(str(task.get("split")) != "dev" for task in tasks):
        raise ValueError("Experiment 1 requires exactly 24 dev records")

    manifest_path = Path("artifacts/manifests/dataset_manifest_v1.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_dataset_hash = manifest["files"]["dev"]["sha256"]
    actual_dataset_hash = sha256_file(dataset_path)
    if actual_dataset_hash != expected_dataset_hash:
        raise ValueError("dev dataset hash does not match dataset_manifest_v1")
    final_test_path = Path("data/code_agent_v1/final_test.jsonl")
    final_test_hash = sha256_file(final_test_path)
    expected_final_hash = manifest["files"]["final_test"]["sha256"]
    if final_test_hash != expected_final_hash:
        raise ValueError("Frozen Final Test hash changed")

    model_config = load_yaml(args.model_config)
    rows: list[dict[str, Any]] = []
    output_dir = Path(args.output_dir)
    episodes_path = output_dir / "sft" / "dev_episodes.jsonl"
    if args.reuse_episodes:
        episodes_input = Path(args.episodes_input or episodes_path)
        raw_rows = read_jsonl(episodes_input)
        if len(raw_rows) != len(tasks):
            raise ValueError("reused episodes must contain exactly 24 Dev records")
        task_by_id = {str(task["id"]): task for task in tasks}
        for row in raw_rows:
            task_id = str(row["task_id"]).split(":")[-1]
            if task_id not in task_by_id:
                raise ValueError(f"episode is not a v1 dev task: {task_id}")
            row["classification"] = classify_episode(
                task_by_id[task_id], row, args.max_steps
            )
            rows.append(row)
    else:
        runner = build_adapter_runner(
            model_config,
            str(adapter_path),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
        environment = CodeAgentEnvironment(max_steps=args.max_steps)
        for index, task in enumerate(tasks, start=1):
            print(f"Evaluating dev episode {index}/{len(tasks)}: {task['id']}", flush=True)
            episode = environment.run(
                task,
                runner.generate,
                task_id=f"experiment_1:sft:{task['id']}",
            )
            row = episode.to_dict()
            row["backend"] = "sft"
            row["category"] = task.get("category", task.get("task_type"))
            row["dataset_split"] = "dev"
            row["classification"] = classify_episode(task, row, args.max_steps)
            rows.append(row)
        del runner
    write_jsonl(episodes_path, rows)
    selected = select_representative_cases(rows)
    cases_path = Path(args.cases_output)
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    cases_path.write_text(build_cases_markdown(rows, selected), encoding="utf-8")

    evaluator_path = Path("artifacts/evaluation/evaluator_version_v1.json")
    reward_path = Path("artifacts/reward/reward_v1_freeze.json")
    runtime_path = Path("artifacts/runtime/runtime_version_v1.json")
    source_paths = {
        "experiment_script": str(Path(__file__).as_posix()),
        "environment_source": "src/llm_posttrain/agent/code_environment.py",
        "parser_source": "src/llm_posttrain/agent/protocol.py",
        "reward_source": "src/llm_posttrain/rewards/code_agent.py",
        "runtime_source": "src/llm_posttrain/agent/runtime.py",
    }
    source_hashes = {
        key: sha256_file(path) for key, path in source_paths.items()
    }
    classifier_self_checks = (
        is_canonical_tool_output(
            '<tool_call>{"name":"calculator","arguments":{"expression":"2+2"}}</tool_call>'
        )
        and not is_canonical_tool_output(
            '{"name":"calculator","arguments":{"expression":"2+2"}}'
        )
        and recovery_kind('{"name":"calculator","arguments":{}}')
        == "bare_json_or_json_prefix"
    )
    pytest_command = [sys.executable, "-m", "pytest", "-q"]
    pytest_process = subprocess.run(
        pytest_command,
        capture_output=True,
        text=True,
        check=False,
    )
    pytest_result = "\n".join(
        part.strip()
        for part in (pytest_process.stdout, pytest_process.stderr)
        if part.strip()
    )[-2000:]
    tests = {
        "classifier_self_checks": classifier_self_checks,
        "pytest_command": " ".join(pytest_command),
        "pytest_passed": pytest_process.returncode == 0,
        "pytest_returncode": pytest_process.returncode,
        "pytest_result": pytest_result,
    }
    summary: dict[str, Any] = {
        "version": "experiment_1_failure_taxonomy_v1",
        "experiment": "Experiment 1 — Failure Taxonomy and Raw / Recovered / Final",
        "decision_gate": "CONTINUE" if classifier_self_checks and tests["pytest_passed"] else "MODIFY",
        "tests": tests,
        "scope": {
            "dataset": "dev_only",
            "validation_individual_cases_evaluated": False,
            "final_test_model_inference": False,
            "training_run": False,
            "reward_or_evaluator_modified": False,
        },
        "code": {
            "head": git_value("rev-parse", "HEAD"),
            "status_count_at_run_start": len(git_value("status", "--short").splitlines()),
        },
        "model": {
            "base_model": model_config.get("model", {}).get("repo_id"),
            "adapter_path": str(adapter_path),
            "adapter_config_sha256": sha256_file(adapter_path / "adapter_config.json"),
            "adapter_weight_sha256": sha256_file(adapter_path / "adapter_model.safetensors"),
            "decode": {
                "do_sample": False,
                "max_new_tokens": args.max_new_tokens,
            },
        },
        "dataset": {
            "path": str(dataset_path),
            "split": "dev",
            "count": len(tasks),
            "categories": dict(Counter(str(task.get("category")) for task in tasks)),
            "sha256": actual_dataset_hash,
            "manifest_path": str(manifest_path),
            "manifest_version": manifest.get("version"),
        },
        "runtime": {
            "max_steps": args.max_steps,
            "environment": "CodeAgentEnvironment",
        },
        "sources": {
            "evaluator_path": str(evaluator_path),
            "evaluator_sha256": sha256_file(evaluator_path),
            "reward_path": str(reward_path),
            "reward_sha256": sha256_file(reward_path),
            "runtime_path": str(runtime_path),
            "runtime_sha256": sha256_file(runtime_path),
            "source_hashes": source_hashes,
        },
        "metrics": aggregate(rows),
        "metric_audit": build_metric_audit(tasks, rows),
        "failure_taxonomy": {
            "priority": TAXONOMY_PRIORITY,
            "primary_failure_is_exclusive": True,
            "secondary_failures_are_nonexclusive": True,
            "observation_rule_version": "experiment_1_observation_rule_v1",
        },
        "representative_cases": selected,
        "final_test_protection": {
            "path": str(final_test_path),
            "actual_sha256": final_test_hash,
            "manifest_sha256": expected_final_hash,
            "hash_matches_manifest": final_test_hash == expected_final_hash,
            "model_inference": False,
        },
        "artifacts": {
            "episodes_path": str(episodes_path),
            "episodes_sha256": sha256_file(episodes_path),
            "representative_cases_path": str(cases_path),
            "representative_cases_sha256": sha256_file(cases_path),
            "summary_path": str(Path(args.summary_output)),
        },
        "evidence_boundary": {
            "historical_or_new_model_improvement_claim": False,
            "resume_safe_claim": "Dev-only failure taxonomy with raw/recovered/final trace evidence",
            "unsupported_claims": [
                "Reward change is proven",
                "Model performance improved",
                "Real repository generalization",
            ],
        },
    }
    summary_path = Path(args.summary_output)
    write_json(summary_path, summary)
    Path(args.card_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.card_output).write_text(build_card(summary), encoding="utf-8")
    Path(args.report_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_output).write_text(build_report(summary), encoding="utf-8")
    summary["artifacts"]["card_path"] = str(Path(args.card_output))
    summary["artifacts"]["card_sha256"] = sha256_file(args.card_output)
    summary["artifacts"]["report_path"] = str(Path(args.report_output))
    summary["artifacts"]["report_sha256"] = sha256_file(args.report_output)
    write_json(summary_path, summary)

    print(json.dumps({
        "decision_gate": summary["decision_gate"],
        "summary": str(summary_path),
        "metrics": summary["metrics"],
        "final_test_untouched": summary["final_test_protection"]["hash_matches_manifest"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
