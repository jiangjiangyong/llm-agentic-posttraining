from __future__ import annotations

import ast
import math
import operator
from typing import Any

from .schema import ToolSpec


class CalculatorError(ValueError):
    """Raised when an expression violates the calculator contract."""


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _guard_number(value: int | float) -> int | float:
    if isinstance(value, float) and not math.isfinite(value):
        raise CalculatorError("result must be finite")
    if abs(value) > 1e100:
        raise CalculatorError("result is too large")
    return value


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalculatorError("only integer and floating-point literals are allowed")
        return _guard_number(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _guard_number(_UNARY_OPERATORS[type(node.op)](_evaluate(node.operand)))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise CalculatorError("exponent magnitude must be at most 100")
        return _guard_number(_BINARY_OPERATORS[type(node.op)](left, right))
    raise CalculatorError(
        f"unsupported expression node: {type(node).__name__}"
    )


def _format_number(value: int | float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def calculate(arguments: dict[str, Any]) -> dict[str, Any]:
    expression = arguments.get("expression")
    if not isinstance(expression, str) or not expression.strip():
        raise CalculatorError("expression must be a non-empty string")
    if len(expression) > 200:
        raise CalculatorError("expression is too long")
    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate(tree)
    except (SyntaxError, ZeroDivisionError, OverflowError) as exc:
        raise CalculatorError(str(exc)) from exc
    return {
        "expression": expression,
        "result": result,
        "result_text": _format_number(result),
    }


def calculator_tool_spec() -> ToolSpec:
    return ToolSpec(
        name="calculator",
        description="Evaluate a simple arithmetic expression safely.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression using numbers and + - * / // % **.",
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
        handler=calculate,
    )
