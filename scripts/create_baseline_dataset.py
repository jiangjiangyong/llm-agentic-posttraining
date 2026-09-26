from __future__ import annotations

import json
from pathlib import Path


SAMPLES = [
    {
        "id": "math_001",
        "category": "math",
        "messages": [
            {"role": "system", "content": "You are a precise math assistant. Output only the final integer."},
            {"role": "user", "content": "Calculate 125 * 36."},
        ],
        "scorer": "exact",
        "expected": "4500",
    },
    {
        "id": "math_002",
        "category": "math",
        "messages": [
            {"role": "system", "content": "You are a precise math assistant. Output only the final integer."},
            {"role": "user", "content": "Calculate (18 + 24) * 3."},
        ],
        "scorer": "exact",
        "expected": "126",
    },
    {
        "id": "math_003",
        "category": "math",
        "messages": [
            {"role": "system", "content": "You are a precise math assistant. Output only the final integer."},
            {"role": "user", "content": "What is 15 percent of 240?"},
        ],
        "scorer": "exact",
        "expected": "36",
    },
    {
        "id": "math_004",
        "category": "math",
        "messages": [
            {"role": "system", "content": "You are a precise math assistant. Output only the final integer."},
            {"role": "user", "content": "Calculate 2 to the power of 10."},
        ],
        "scorer": "exact",
        "expected": "1024",
    },
    {
        "id": "json_001",
        "category": "json",
        "messages": [
            {"role": "system", "content": "Return valid JSON only with category and priority."},
            {"role": "user", "content": "Classify: login failed after three wrong passwords. category=account, priority=medium."},
        ],
        "scorer": "json_contains",
        "expected": {"category": "account", "priority": "medium"},
    },
    {
        "id": "json_002",
        "category": "json",
        "messages": [
            {"role": "system", "content": "Return valid JSON only with category and priority."},
            {"role": "user", "content": "Classify: the order was paid but not shipped. category=order, priority=high."},
        ],
        "scorer": "json_contains",
        "expected": {"category": "order", "priority": "high"},
    },
    {
        "id": "json_003",
        "category": "json",
        "messages": [
            {"role": "system", "content": "Return valid JSON only with category and priority."},
            {"role": "user", "content": "Classify: change my avatar. category=profile, priority=low."},
        ],
        "scorer": "json_contains",
        "expected": {"category": "profile", "priority": "low"},
    },
    {
        "id": "json_004",
        "category": "json",
        "messages": [
            {"role": "system", "content": "Return valid JSON only with category and priority."},
            {"role": "user", "content": "Classify: cannot export the monthly sales report. category=report, priority=medium."},
        ],
        "scorer": "json_contains",
        "expected": {"category": "report", "priority": "medium"},
    },
    {
        "id": "code_001",
        "category": "code",
        "messages": [
            {"role": "system", "content": "Output Python code only. Do not use Markdown fences."},
            {"role": "user", "content": "Implement add(a, b), returning the sum of two values."},
        ],
        "scorer": "python_syntax",
        "expected": {"required_symbols": ["add"]},
    },
    {
        "id": "code_002",
        "category": "code",
        "messages": [
            {"role": "system", "content": "Output Python code only. Do not use Markdown fences."},
            {"role": "user", "content": "Implement is_even(n), returning whether n is even."},
        ],
        "scorer": "python_syntax",
        "expected": {"required_symbols": ["is_even"]},
    },
    {
        "id": "code_003",
        "category": "code",
        "messages": [
            {"role": "system", "content": "Output Python code only. Do not use Markdown fences."},
            {"role": "user", "content": "Implement square_list(xs), returning the square of every number."},
        ],
        "scorer": "python_syntax",
        "expected": {"required_symbols": ["square_list"]},
    },
    {
        "id": "code_004",
        "category": "code",
        "messages": [
            {"role": "system", "content": "Output Python code only. Do not use Markdown fences."},
            {"role": "user", "content": "Implement reverse_string(s), returning the reversed string."},
        ],
        "scorer": "python_syntax",
        "expected": {"required_symbols": ["reverse_string"]},
    },
]


def main() -> None:
    output_path = Path("data/evaluation/base_smoke.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in SAMPLES:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Wrote {len(SAMPLES)} samples to {output_path}")


if __name__ == "__main__":
    main()
