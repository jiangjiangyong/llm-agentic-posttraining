from __future__ import annotations

import re
from typing import Any

from .schema import ToolSpec


MOCK_DOCUMENTS = [
    {
        "id": "doc-python-timeout",
        "title": "Python Executor Timeout Policy",
        "content": (
            "Python tasks run in an offline subprocess with a three second "
            "timeout and bounded stdout."
        ),
        "tags": ["python", "executor", "timeout", "sandbox"],
    },
    {
        "id": "doc-reward-unit-test",
        "title": "Unit Test Reward Definition",
        "content": (
            "Unit test reward is one when all required assertions pass in the "
            "isolated episode workspace."
        ),
        "tags": ["reward", "unit test", "evaluation"],
    },
    {
        "id": "doc-tool-protocol",
        "title": "Tool Calling Protocol",
        "content": (
            "A canonical tool call contains exactly name and arguments inside "
            "one tool_call wrapper."
        ),
        "tags": ["tool calling", "protocol", "json"],
    },
    {
        "id": "doc-qlora",
        "title": "QLoRA Experiment Note",
        "content": (
            "QLoRA keeps the base model quantized and updates low rank adapter "
            "parameters during supervised fine tuning."
        ),
        "tags": ["qlora", "sft", "training"],
    },
    {
        "id": "doc-flywheel",
        "title": "Failure Data Flywheel",
        "content": (
            "Failed trajectories are grouped by task and converted into "
            "repair SFT samples or preference pairs."
        ),
        "tags": ["failure", "trajectory", "sft", "dpo"],
    },
]


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 1}


def mock_search(arguments: dict[str, Any]) -> dict[str, Any]:
    query = arguments.get("query")
    top_k = arguments.get("top_k", 3)
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise ValueError("top_k must be an integer")
    top_k = max(1, min(top_k, 5))
    query_tokens = _tokens(query)
    ranked = []
    for document in MOCK_DOCUMENTS:
        haystack = _tokens(
            " ".join(
                [document["title"], document["content"], *document["tags"]]
            )
        )
        score = len(query_tokens & haystack)
        if score:
            ranked.append((score, document))
    ranked.sort(key=lambda item: (-item[0], item[1]["id"]))
    return {
        "query": query,
        "results": [
            {
                "id": document["id"],
                "title": document["title"],
                "snippet": document["content"],
                "score": score,
            }
            for score, document in ranked[:top_k]
        ],
    }


def mock_search_tool_spec() -> ToolSpec:
    return ToolSpec(
        name="mock_search",
        description="Search a deterministic offline project knowledge corpus.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        handler=mock_search,
    )

