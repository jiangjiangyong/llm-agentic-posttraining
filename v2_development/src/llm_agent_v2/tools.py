"""Typed tool registry with deterministic CPU-only tools."""

from __future__ import annotations

import ast
import json
import operator
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    reversible: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "reversible": self.reversible,
        }


@dataclass
class ToolResult:
    success: bool
    output: str = ""
    error_type: str | None = None
    state_patch: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(
        cls,
        output: str,
        state_patch: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(
            success=True,
            output=output,
            state_patch=dict(state_patch or {}),
            metadata=dict(metadata or {}),
        )

    @classmethod
    def fail(
        cls,
        error_type: str,
        output: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(
            success=False,
            output=output,
            error_type=error_type,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error_type": self.error_type,
            "state_patch": self.state_patch,
            "metadata": self.metadata,
        }


ToolHandler = Callable[[dict[str, Any], Mapping[str, Any]], ToolResult]


class ToolRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, tuple[ToolSpec, ToolHandler]] = {}

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        if spec.name in self._handlers:
            raise ValueError(f"tool already registered: {spec.name}")
        self._handlers[spec.name] = (spec, handler)

    def specs(self) -> list[ToolSpec]:
        return [spec for spec, _ in self._handlers.values()]

    def invoke(
        self,
        name: str,
        arguments: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> ToolResult:
        entry = self._handlers.get(name)
        if entry is None:
            return ToolResult.fail("unknown_tool", f"tool is not registered: {name}")
        spec, handler = entry
        if not isinstance(arguments, Mapping):
            return ToolResult.fail("invalid_argument", "tool arguments must be an object")
        validation_error = _validate_arguments(spec.input_schema, arguments)
        if validation_error:
            return ToolResult.fail("invalid_argument", validation_error)
        try:
            return handler(dict(arguments), context or {})
        except Exception as exc:  # defensive boundary around tool code
            return ToolResult.fail("tool_execution", f"tool raised {type(exc).__name__}: {exc}")


def _validate_arguments(schema: Mapping[str, Any], arguments: Mapping[str, Any]) -> str | None:
    required = schema.get("required", [])
    for key in required:
        if key not in arguments:
            return f"missing required argument: {key}"
    properties = schema.get("properties", {})
    for key, value in arguments.items():
        expected = properties.get(key, {}).get("type")
        if expected == "string" and not isinstance(value, str):
            return f"argument {key} must be a string"
        if expected == "number" and not isinstance(value, (int, float)):
            return f"argument {key} must be a number"
    return None


_BINOPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_calculate(expression: str) -> Any:
    tree = ast.parse(expression, mode="eval")

    def evaluate(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
            return _UNARYOPS[type(node.op)](evaluate(node.operand))
        raise ValueError("only numeric arithmetic is allowed")

    result = evaluate(tree)
    if isinstance(result, float) and not result.is_integer():
        return round(result, 12)
    return result


def build_cpu_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="calculator",
            description="Evaluate a numeric arithmetic expression without names or function calls.",
            input_schema={
                "type": "object",
                "required": ["expression"],
                "properties": {"expression": {"type": "string"}},
            },
        ),
        lambda args, _context: ToolResult.ok(str(_safe_calculate(args["expression"]))),
    )

    def workspace_set(args: dict[str, Any], _context: Mapping[str, Any]) -> ToolResult:
        key = args["key"]
        value = args["value"]
        return ToolResult.ok(
            json.dumps({"key": key, "value": value}, ensure_ascii=False, sort_keys=True),
            state_patch={"workspace_state": {key: value}},
        )

    registry.register(
        ToolSpec(
            name="workspace_set",
            description="Set a JSON-compatible value in the typed workspace state.",
            input_schema={
                "type": "object",
                "required": ["key", "value"],
                "properties": {"key": {"type": "string"}},
            },
        ),
        workspace_set,
    )

    def workspace_get(args: dict[str, Any], context: Mapping[str, Any]) -> ToolResult:
        key = args["key"]
        workspace = context.get("workspace_state", {})
        if key not in workspace:
            return ToolResult.fail("observation_error", f"workspace key does not exist: {key}")
        return ToolResult.ok(json.dumps(workspace[key], ensure_ascii=False, sort_keys=True))

    registry.register(
        ToolSpec(
            name="workspace_get",
            description="Read a value from the current typed workspace state.",
            input_schema={
                "type": "object",
                "required": ["key"],
                "properties": {"key": {"type": "string"}},
            },
        ),
        workspace_get,
    )
    return registry
