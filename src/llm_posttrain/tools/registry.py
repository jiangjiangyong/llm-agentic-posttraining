from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .calculator import calculator_tool_spec
from .schema import ToolSpec


class ToolValidationError(ValueError):
    """Raised when a tool call does not satisfy its schema."""


@dataclass(frozen=True)
class ToolExecutionResult:
    tool_name: str
    ok: bool
    output: Any = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "ok": self.ok,
            "output": self.output,
            "error": self.error,
        }


class ToolRegistry:
    def __init__(self, specs: list[ToolSpec] | None = None) -> None:
        self._specs: dict[str, ToolSpec] = {}
        for spec in specs or []:
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        if not spec.name or spec.name in self._specs:
            raise ValueError(f"tool name is empty or already registered: {spec.name!r}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise ToolValidationError(f"unknown tool: {name}") from exc

    def schemas(self) -> list[dict[str, Any]]:
        return [spec.to_openai_schema() for spec in self._specs.values()]

    def _validate_arguments(self, spec: ToolSpec, arguments: Any) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            raise ToolValidationError("tool arguments must be a JSON object")
        properties = spec.parameters.get("properties", {})
        required = spec.parameters.get("required", [])
        missing = [name for name in required if name not in arguments]
        if missing:
            raise ToolValidationError(f"missing required arguments: {missing}")
        if spec.parameters.get("additionalProperties") is False:
            extras = [name for name in arguments if name not in properties]
            if extras:
                raise ToolValidationError(f"unexpected arguments: {extras}")
        return arguments

    def execute(self, name: str, arguments: Any) -> ToolExecutionResult:
        try:
            spec = self.get(name)
            validated = self._validate_arguments(spec, arguments)
            output = spec.handler(validated)
            return ToolExecutionResult(tool_name=name, ok=True, output=output)
        except Exception as exc:
            return ToolExecutionResult(
                tool_name=name,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )


def build_default_registry() -> ToolRegistry:
    return ToolRegistry([calculator_tool_spec()])
