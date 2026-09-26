from __future__ import annotations

import ast
import json
import re
from typing import Any


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    match = re.fullmatch(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def _parse_json(text: str) -> Any:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    return json.loads(text)


def _result(passed: bool, detail: str, **extra: Any) -> dict[str, Any]:
    return {"score": 1.0 if passed else 0.0, "passed": passed, "detail": detail, **extra}


def score_exact(prediction: str, expected: Any) -> dict[str, Any]:
    passed = prediction.strip() == str(expected).strip()
    return _result(passed, "exact match" if passed else "exact mismatch")


def score_contains_all(prediction: str, expected: Any) -> dict[str, Any]:
    terms = expected if isinstance(expected, list) else [expected]
    lowered = prediction.lower()
    missing = [str(term) for term in terms if str(term).lower() not in lowered]
    return _result(not missing, "all terms present" if not missing else f"missing: {missing}")


def score_json_contains(prediction: str, expected: Any) -> dict[str, Any]:
    if not isinstance(expected, dict):
        return _result(False, "expected must be an object")
    try:
        actual = _parse_json(prediction)
    except (TypeError, json.JSONDecodeError) as exc:
        return _result(False, f"invalid JSON: {exc}")
    if not isinstance(actual, dict):
        return _result(False, "prediction JSON is not an object")
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    return _result(not mismatches, "JSON fields match" if not mismatches else f"mismatches: {mismatches}")


def score_python_syntax(prediction: str, expected: Any) -> dict[str, Any]:
    code = _strip_code_fence(prediction)
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return _result(False, f"syntax error: {exc}")

    required_symbols = []
    if isinstance(expected, dict):
        required_symbols = [str(item) for item in expected.get("required_symbols", [])]
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    missing = [name for name in required_symbols if name not in defined]
    return _result(not missing, "AST and required symbols valid" if not missing else f"missing symbols: {missing}")


SCORERS = {
    "exact": score_exact,
    "contains_all": score_contains_all,
    "json_contains": score_json_contains,
    "python_syntax": score_python_syntax,
}


def score_prediction(
    scorer_name: str,
    prediction: str,
    expected: Any,
) -> dict[str, Any]:
    try:
        scorer = SCORERS[scorer_name]
    except KeyError as exc:
        raise KeyError(f"Unknown scorer: {scorer_name}") from exc
    return scorer(prediction, expected)
