from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any



def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def canonical_call(name: str, arguments: dict[str, Any]) -> str:
    payload = {"name": name, "arguments": arguments}
    return (
        "<tool_call>"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "</tool_call>"
    )


def observation(name: str, result: dict[str, Any]) -> str:
    return (
        "<tool_observation>\n"
        + json.dumps(
            {"tool_name": name, "ok": True, "output": result, "error": None},
            ensure_ascii=False,
        )
        + "\n</tool_observation>"
    )


def system_message() -> str:
    return (
        "You are CodeToolAgent Protocol v2. "
        "Use the available tools to solve the task. "
        "Tool calls must be exactly one canonical block with only name and "
        "arguments: <tool_call>{\"name\":\"tool\",\"arguments\":{}}</tool_call>. "
        "After an observation, continue with another tool call or give the final "
        "answer in plain text. Never use Markdown fences for tool calls.\n"
        "Available tools and required arguments:\n"
        "calculator(expression); "
        "python_executor(code, stdin optional); "
        "validate_json(json_text, schema); "
        "mock_search(query, top_k optional); "
        "write_file(path, content); "
        "read_file(path); "
        "list_files(); "
        "run_unit_tests(path, tests)."
    )


def task_record(
    *,
    task_id: str,
    category: str,
    user: str,
    assistant_turns: list[tuple[str, dict[str, Any], dict[str, Any]]],
    final_answer: str,
    expected: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_message()},
        {"role": "user", "content": user},
    ]
    for name, arguments, result in assistant_turns:
        messages.append(
            {
                "role": "assistant",
                "content": canonical_call(name, arguments),
            }
        )
        messages.append(
            {
                "role": "tool",
                "name": name,
                "content": observation(name, result),
            }
        )
    messages.append({"role": "assistant", "content": final_answer})
    return {
        "id": task_id,
        "category": category,
        "messages": messages,
        "prompt_messages": messages[:2],
        "expected": expected,
        "metadata": metadata,
    }


def make_code_task(index: int, split: str) -> dict[str, Any]:
    function_name = f"reverse_words_{split}_{index:03d}"
    text = f"alpha {index} beta gamma"
    reversed_text = " ".join(reversed(text.split()))
    code = (
        f"def {function_name}(text):\n"
        "    return ' '.join(reversed(text.split()))\n"
    )
    tests = (
        f"assert {function_name}({text!r}) == {reversed_text!r}\n"
        f"assert {function_name}('one two') == 'two one'\n"
        f"assert {function_name}('') == ''\n"
    )
    user = (
        f"Implement {function_name}(text) in solution.py. "
        "Write the file, run the provided unit tests, and only after the tests "
        "pass report the verification result."
    )
    write_result = {
        "path": "solution.py",
        "chars": len(code),
        "bytes": len(code.encode("utf-8")),
    }
    test_result = {
        "passed": True,
        "status": "ok",
        "path": "solution.py",
        "test_count_hint": tests.count("assert "),
        "stdout": "",
        "stderr": "",
        "return_code": 0,
        "timed_out": False,
    }
    expected = {
        "required_tools": ["write_file", "run_unit_tests"],
        "expected_tool_calls": 2,
        "expected_arguments": {
            "write_file": {"path": "solution.py"},
            "run_unit_tests": {"path": "solution.py"},
        },
        "execution_tools": ["write_file", "run_unit_tests"],
        "unit_test_required": True,
        "file_checks": [{"path": "solution.py", "contains": [f"def {function_name}"]}],
        "final_answer_contains": [f"PASS: {function_name}"],
    }
    return task_record(
        task_id=f"code_{split}_{index:03d}",
        category="code_unit_test",
        user=user + "\nUnit tests:\n" + tests,
        assistant_turns=[
            ("write_file", {"path": "solution.py", "content": code}, write_result),
            ("run_unit_tests", {"path": "solution.py", "tests": tests}, test_result),
        ],
        final_answer=f"PASS: {function_name}",
        expected=expected,
        metadata={
            "task_type": "write_file_then_unit_test",
            "function_name": function_name,
            "source_code": code,
            "tests": tests,
            "answer": reversed_text,
        },
    )


def make_executor_task(index: int, split: str) -> dict[str, Any]:
    values = [index + 2, index + 5, index + 9]
    result = sum(value * value for value in values)
    code = (
        f"values = {values!r}\n"
        "print(sum(value * value for value in values))\n"
    )
    user = (
        "Use python_executor to run a short offline Python program. "
        f"The program must calculate the sum of squares of {values} and print it. "
        "After the execution succeeds, report the result."
    )
    execution_result = {
        "ok": True,
        "status": "ok",
        "return_code": 0,
        "stdout": f"{result}\n",
        "stderr": "",
        "timed_out": False,
        "result_text": str(result),
    }
    expected = {
        "required_tools": ["python_executor"],
        "expected_tool_calls": 1,
        "execution_tools": ["python_executor"],
        "expected_tool_outputs": {
            "python_executor": {"result_text": str(result)},
        },
        "final_answer": f"RESULT: {result}",
    }
    return task_record(
        task_id=f"executor_{split}_{index:03d}",
        category="python_execution",
        user=user,
        assistant_turns=[
            ("python_executor", {"code": code}, execution_result),
        ],
        final_answer=f"RESULT: {result}",
        expected=expected,
        metadata={
            "task_type": "python_executor",
            "values": values,
            "answer": result,
        },
    )


def make_json_task(index: int, split: str) -> dict[str, Any]:
    value = {
        "ticket_id": f"T-{split[:1].upper()}{index:03d}",
        "priority": "high" if index % 2 else "medium",
        "resolved": bool(index % 2),
    }
    schema = {
        "type": "object",
        "properties": {
            "ticket_id": {"type": "string"},
            "priority": {"type": "string", "enum": ["low", "medium", "high"]},
            "resolved": {"type": "boolean"},
        },
        "required": ["ticket_id", "priority", "resolved"],
        "additionalProperties": False,
    }
    json_text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    user = (
        "Create the requested structured JSON object, validate it with "
        "validate_json, then return the JSON object as the final answer. "
        f"The ticket id is {value['ticket_id']}, priority is {value['priority']}, "
        f"and resolved is {str(value['resolved']).lower()}."
    )
    validate_result = {"valid": True, "value": value, "errors": []}
    expected = {
        "required_tools": ["validate_json"],
        "expected_tool_calls": 1,
        "execution_tools": ["validate_json"],
        "expected_tool_outputs": {"validate_json": {"valid": True}},
        "final_json_schema": schema,
    }
    return task_record(
        task_id=f"json_{split}_{index:03d}",
        category="structured_json",
        user=user,
        assistant_turns=[
            (
                "validate_json",
                {"json_text": json_text, "schema": schema},
                validate_result,
            ),
        ],
        final_answer=json_text,
        expected=expected,
        metadata={"task_type": "json_schema_validation", "value": value},
    )


def make_file_task(index: int, split: str) -> dict[str, Any]:
    config = {
        "service": f"worker-{split}-{index}",
        "workers": index % 4 + 1,
        "enabled": True,
    }
    content = json.dumps(config, ensure_ascii=False, indent=2)
    user = (
        "Create config.json with the requested service configuration, read it "
        "back to verify it exists, and then report CONFIG READY. "
        f"Use service worker-{split}-{index}, workers {config['workers']}, enabled true."
    )
    expected = {
        "required_tools": ["write_file", "read_file"],
        "expected_tool_calls": 2,
        "expected_arguments": {
            "write_file": {"path": "config.json"},
            "read_file": {"path": "config.json"},
        },
        "execution_tools": ["write_file", "read_file"],
        "file_checks": [{"path": "config.json", "contains": [config["service"]]}],
        "final_answer_contains": ["CONFIG READY"],
    }
    return task_record(
        task_id=f"file_{split}_{index:03d}",
        category="workspace_file",
        user=user,
        assistant_turns=[
            (
                "write_file",
                {"path": "config.json", "content": content},
                {
                    "path": "config.json",
                    "chars": len(content),
                    "bytes": len(content.encode("utf-8")),
                },
            ),
            (
                "read_file",
                {"path": "config.json"},
                {"path": "config.json", "content": content, "truncated": False},
            ),
        ],
        final_answer="CONFIG READY",
        expected=expected,
        metadata={"task_type": "write_then_read_file", "config": config},
    )


_SEARCH_CASES = (
    {
        "slug": "unit_test_reward",
        "query": "unit test reward",
        "question": "the definition of unit test reward",
        "answer": "Unit test reward is one when all required assertions pass.",
        "doc_id": "doc-reward-unit-test",
        "title": "Unit Test Reward Definition",
        "snippet": (
            "Unit test reward is one when all required assertions pass "
            "in the isolated episode workspace."
        ),
        "score": 3,
    },
    {
        "slug": "python_timeout",
        "query": "python executor timeout",
        "question": "the Python Executor timeout policy",
        "answer": (
            "Python tasks run in an offline subprocess with a three second "
            "timeout and bounded stdout."
        ),
        "doc_id": "doc-python-timeout",
        "title": "Python Executor Timeout Policy",
        "snippet": (
            "Python tasks run in an offline subprocess with a three second "
            "timeout and bounded stdout."
        ),
        "score": 3,
    },
    {
        "slug": "tool_protocol",
        "query": "tool calling protocol",
        "question": "the canonical tool calling protocol",
        "answer": (
            "A canonical tool call contains exactly name and arguments inside "
            "one tool_call wrapper."
        ),
        "doc_id": "doc-tool-protocol",
        "title": "Tool Calling Protocol",
        "snippet": (
            "A canonical tool call contains exactly name and arguments inside "
            "one tool_call wrapper."
        ),
        "score": 3,
    },
    {
        "slug": "qlora",
        "query": "qlora adapter training",
        "question": "how the QLoRA experiment updates the model",
        "answer": (
            "QLoRA keeps the base model quantized and updates low rank adapter "
            "parameters during supervised fine tuning."
        ),
        "doc_id": "doc-qlora",
        "title": "QLoRA Experiment Note",
        "snippet": (
            "QLoRA keeps the base model quantized and updates low rank adapter "
            "parameters during supervised fine tuning."
        ),
        "score": 3,
    },
    {
        "slug": "failure_flywheel",
        "query": "failure data flywheel",
        "question": "how failed trajectories are reused for training",
        "answer": (
            "Failed trajectories are grouped by task and converted into repair "
            "SFT samples or preference pairs."
        ),
        "doc_id": "doc-flywheel",
        "title": "Failure Data Flywheel",
        "snippet": (
            "Failed trajectories are grouped by task and converted into repair "
            "SFT samples or preference pairs."
        ),
        "score": 3,
    },
)


def _search_result(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "query": case["query"],
        "results": [
            {
                "id": case["doc_id"],
                "title": case["title"],
                "snippet": case["snippet"],
                "score": case["score"],
            }
        ],
    }


def make_search_task(index: int, split: str) -> dict[str, Any]:
    case = _SEARCH_CASES[index % len(_SEARCH_CASES)]
    query = str(case["query"])
    answer = str(case["answer"])
    if index % 2:
        user = (
            "First call mock_search. Search the offline project knowledge base "
            f"for {case['question']}. Use the returned observation and answer "
            "with one sentence containing the key fact."
        )
    else:
        user = (
            "Use the mock_search tool before answering. Look up "
            f"{case['question']} in the offline project knowledge base, then "
            "use the result instead of guessing. Return one sentence."
        )
    expected = {
        "required_tools": ["mock_search"],
        "expected_tool_calls": 1,
        "expected_arguments": {"mock_search": {"query": query}},
        "execution_tools": ["mock_search"],
        "expected_tool_outputs": {"mock_search": {"query": query}},
        "final_answer_contains": [answer],
    }
    return task_record(
        task_id=f"search_{split}_{index:03d}",
        category="mock_search",
        user=user,
        assistant_turns=[
            ("mock_search", {"query": query, "top_k": 3}, _search_result(case)),
        ],
        final_answer=answer,
        expected=expected,
        metadata={
            "task_type": "mock_search",
            "query": query,
            "document_id": case["doc_id"],
        },
    )


def make_research_to_file_task(index: int, split: str) -> dict[str, Any]:
    case = _SEARCH_CASES[index % len(_SEARCH_CASES)]
    query = str(case["query"])
    answer = str(case["answer"])
    content = answer + "\n"
    user = (
        "Use mock_search to look up the requested fact in the offline project "
        f"knowledge base ({case['question']}). Then write the key fact from "
        "the search observation to finding.md, read the file back, and only "
        "after verification report NOTE READY."
    )
    expected = {
        "required_tools": ["mock_search", "write_file", "read_file"],
        "expected_tool_calls": 3,
        "expected_arguments": {
            "mock_search": {"query": query},
            "write_file": {"path": "finding.md"},
            "read_file": {"path": "finding.md"},
        },
        "execution_tools": ["mock_search", "write_file", "read_file"],
        "expected_tool_outputs": {
            "mock_search": {"query": query},
            "write_file": {"path": "finding.md"},
            "read_file": {"path": "finding.md"},
        },
        "file_checks": [{"path": "finding.md", "contains": [answer]}],
        "final_answer_contains": ["NOTE READY", answer],
    }
    return task_record(
        task_id=f"research_file_{split}_{index:03d}",
        category="research_to_file",
        user=user,
        assistant_turns=[
            ("mock_search", {"query": query, "top_k": 3}, _search_result(case)),
            (
                "write_file",
                {"path": "finding.md", "content": content},
                {
                    "path": "finding.md",
                    "chars": len(content),
                    "bytes": len(content.encode("utf-8")),
                },
            ),
            (
                "read_file",
                {"path": "finding.md"},
                {"path": "finding.md", "content": content, "truncated": False},
            ),
        ],
        final_answer=f"NOTE READY: {answer}",
        expected=expected,
        metadata={
            "task_type": "search_write_read",
            "query": query,
            "document_id": case["doc_id"],
            "answer": answer,
        },
    )


def build_records(count: int, split: str, seed: int) -> list[dict[str, Any]]:
    makers = (
        make_code_task,
        make_executor_task,
        make_json_task,
        make_file_task,
        make_search_task,
        make_research_to_file_task,
    )
    records = [makers[index % len(makers)](index, split) for index in range(count)]
    random.Random(seed).shuffle(records)
    return records


def validate_records(records: list[dict[str, Any]]) -> None:
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task ids")
    for record in records:
        messages = record["messages"]
        if messages[0]["role"] != "system" or messages[1]["role"] != "user":
            raise ValueError(f"{record['id']}: invalid initial messages")
        if not any(message["role"] == "assistant" for message in messages[2:]):
            raise ValueError(f"{record['id']}: missing assistant supervision")
        expected = record.get("expected", {})
        if not expected.get("required_tools"):
            raise ValueError(f"{record['id']}: missing required_tools")


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-output", default="data/code_agent/train.jsonl")
    parser.add_argument("--valid-output", default="data/code_agent/valid.jsonl")
    parser.add_argument("--benchmark-output", default="data/code_agent/benchmark.jsonl")
    parser.add_argument("--train-count", type=int, default=120)
    parser.add_argument("--valid-count", type=int, default=30)
    parser.add_argument("--benchmark-count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()

    train = build_records(args.train_count, "train", args.seed)
    valid = build_records(args.valid_count, "valid", args.seed + 1_000)
    benchmark = build_records(args.benchmark_count, "benchmark", args.seed + 2_000)
    for records in (train, valid, benchmark):
        validate_records(records)
    write_jsonl(args.train_output, train)
    write_jsonl(args.valid_output, valid)
    write_jsonl(args.benchmark_output, benchmark)
    summary = {
        "train": len(train),
        "valid": len(valid),
        "benchmark": len(benchmark),
        "categories": sorted({record["category"] for record in train}),
        "sha256": {
            "train": sha256(args.train_output),
            "valid": sha256(args.valid_output),
            "benchmark": sha256(args.benchmark_output),
        },
        "evaluation_overlap": sorted(
            set(record["id"] for record in train + valid)
            & set(record["id"] for record in benchmark)
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
