from __future__ import annotations

from pathlib import Path
from typing import Any

from .schema import ToolSpec


class WorkspacePathError(ValueError):
    """Raised when a tool tries to escape the episode workspace."""


_ALLOWED_SUFFIXES = {".json", ".md", ".py", ".txt", ".yaml", ".yml"}
_MAX_FILE_CHARS = 128_000


def safe_workspace_path(
    workspace: str | Path,
    relative_path: str,
) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise WorkspacePathError("path must be a non-empty string")
    candidate = Path(relative_path)
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        raise WorkspacePathError("path must be relative and cannot contain '..'")
    root = Path(workspace).resolve()
    root.mkdir(parents=True, exist_ok=True)
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise WorkspacePathError("path escapes the workspace") from exc
    return resolved


def _display_path(workspace: str | Path, path: Path) -> str:
    return path.resolve().relative_to(Path(workspace).resolve()).as_posix()


def write_file(arguments: dict[str, Any], *, workspace: str | Path) -> dict[str, Any]:
    path = arguments.get("path")
    content = arguments.get("content")
    if not isinstance(content, str):
        raise WorkspacePathError("content must be a string")
    if len(content) > _MAX_FILE_CHARS:
        raise WorkspacePathError("content is too large")
    target = safe_workspace_path(workspace, path)
    if target.suffix and target.suffix.lower() not in _ALLOWED_SUFFIXES:
        raise WorkspacePathError(f"file suffix is not allowed: {target.suffix}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {
        "path": _display_path(workspace, target),
        "chars": len(content),
        "bytes": len(content.encode("utf-8")),
    }


def read_file(arguments: dict[str, Any], *, workspace: str | Path) -> dict[str, Any]:
    path = arguments.get("path")
    target = safe_workspace_path(workspace, path)
    if not target.exists() or not target.is_file():
        raise WorkspacePathError(f"file does not exist: {path}")
    content = target.read_text(encoding="utf-8")
    return {
        "path": _display_path(workspace, target),
        "content": content[:_MAX_FILE_CHARS],
        "truncated": len(content) > _MAX_FILE_CHARS,
    }


def list_files(arguments: dict[str, Any], *, workspace: str | Path) -> dict[str, Any]:
    del arguments
    root = Path(workspace).resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "__agent_main__.py"
    )
    return {"files": paths[:200], "count": len(paths)}


def make_file_tools(workspace: str | Path) -> list[ToolSpec]:
    return [
        ToolSpec(
            name="write_file",
            description="Write UTF-8 text to a relative path in the episode workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            handler=lambda arguments: write_file(arguments, workspace=workspace),
        ),
        ToolSpec(
            name="read_file",
            description="Read UTF-8 text from a relative path in the episode workspace.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=lambda arguments: read_file(arguments, workspace=workspace),
        ),
        ToolSpec(
            name="list_files",
            description="List files currently created in the episode workspace.",
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=lambda arguments: list_files(arguments, workspace=workspace),
        ),
    ]

