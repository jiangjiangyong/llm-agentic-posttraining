from __future__ import annotations

import json
from typing import Any

from .schema import ToolSpec


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def validate_json_value(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    errors: list[str] = []
    expected_type = schema.get("type")
    if isinstance(expected_type, str) and not _type_matches(value, expected_type):
        errors.append(f"{path}: expected {expected_type}")
        return errors
    if isinstance(value, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                errors.append(f"{path}.{key}: required field is missing")
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for key, child_schema in properties.items():
                if key in value and isinstance(child_schema, dict):
                    errors.extend(
                        validate_json_value(value[key], child_schema, f"{path}.{key}")
                    )
        if schema.get("additionalProperties") is False and isinstance(properties, dict):
            extras = sorted(set(value) - set(properties))
            errors.extend(f"{path}: unexpected field {key}" for key in extras)
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            errors.extend(
                validate_json_value(item, schema["items"], f"{path}[{index}]")
            )
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value is not in enum")
    return errors


def validate_json(arguments: dict[str, Any]) -> dict[str, Any]:
    json_text = arguments.get("json_text")
    schema = arguments.get("schema")
    if not isinstance(json_text, str):
        raise ValueError("json_text must be a string")
    if not isinstance(schema, dict):
        raise ValueError("schema must be an object")
    try:
        value = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return {
            "valid": False,
            "value": None,
            "errors": [f"invalid JSON: {exc}"],
        }
    errors = validate_json_value(value, schema)
    return {
        "valid": not errors,
        "value": value if not errors else None,
        "errors": errors,
    }


def json_validator_tool_spec() -> ToolSpec:
    return ToolSpec(
        name="validate_json",
        description="Validate a JSON string against a small JSON Schema subset.",
        parameters={
            "type": "object",
            "properties": {
                "json_text": {"type": "string"},
                "schema": {"type": "object"},
            },
            "required": ["json_text", "schema"],
            "additionalProperties": False,
        },
        handler=validate_json,
    )

