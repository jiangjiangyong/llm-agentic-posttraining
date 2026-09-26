from __future__ import annotations

import ast
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import ToolSpec

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class CodeSafetyError(ValueError):
    """Raised when generated Python violates the execution policy."""


@dataclass(frozen=True)
class ExecutionPolicy:
    timeout_seconds: float = 3.0
    max_output_chars: int = 8000
    max_code_chars: int = 16000
    max_file_size_bytes: int = 128_000


_ALLOWED_IMPORTS = {
    "collections",
    "datetime",
    "json",
    "math",
    "random",
    "re",
    "statistics",
    "string",
    "unittest",
}
_BLOCKED_NAMES = {
    "__builtins__",
    "__import__",
    "breakpoint",
    "compile",
    "dir",
    "eval",
    "exec",
    "exit",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "quit",
    "setattr",
    "vars",
}
_BLOCKED_MODULES = {
    "builtins",
    "ctypes",
    "importlib",
    "multiprocessing",
    "os",
    "pathlib",
    "pickle",
    "resource",
    "shutil",
    "signal",
    "socket",
    "subprocess",
    "sys",
    "threading",
    "traceback",
}


def _module_root(name: str) -> str:
    return name.split(".", 1)[0].strip()


def validate_python_source(
    source: str,
    *,
    policy: ExecutionPolicy | None = None,
) -> ast.AST:
    policy = policy or ExecutionPolicy()
    if not isinstance(source, str) or not source.strip():
        raise CodeSafetyError("code must be a non-empty string")
    if len(source) > policy.max_code_chars:
        raise CodeSafetyError(
            f"code exceeds max_code_chars={policy.max_code_chars}"
        )
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise CodeSafetyError(f"syntax error: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _BLOCKED_NAMES:
            raise CodeSafetyError(f"blocked name: {node.id}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in _BLOCKED_NAMES:
                raise CodeSafetyError(f"blocked attribute: {node.attr}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = _module_root(alias.name)
                if root not in _ALLOWED_IMPORTS:
                    raise CodeSafetyError(f"import is not allowed: {alias.name}")
        if isinstance(node, ast.ImportFrom):
            if node.level != 0:
                raise CodeSafetyError("relative imports are not allowed")
            root = _module_root(node.module or "")
            if root not in _ALLOWED_IMPORTS:
                raise CodeSafetyError(
                    f"import is not allowed: {node.module or '<empty>'}"
                )
            if any(alias.name == "*" for alias in node.names):
                raise CodeSafetyError("star imports are not allowed")
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise CodeSafetyError("global and nonlocal statements are not allowed")

    return tree


def _limited_preexec(policy: ExecutionPolicy):
    if os.name != "posix":
        return None

    def limit() -> None:
        try:
            import resource

            cpu_seconds = max(1, math.ceil(policy.timeout_seconds) + 1)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            resource.setrlimit(
                resource.RLIMIT_FSIZE,
                (policy.max_file_size_bytes, policy.max_file_size_bytes),
            )
            resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
        except (ImportError, OSError, ValueError):
            # subprocess timeout remains the hard fallback on unsupported hosts.
            pass

    return limit


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    omitted = len(value) - limit
    return value[:limit] + f"\n...[truncated {omitted} chars]"


def execute_python(
    source: str,
    *,
    stdin: str = "",
    workdir: str | Path | None = None,
    policy: ExecutionPolicy | None = None,
) -> dict[str, Any]:
    policy = policy or ExecutionPolicy()
    started = time.perf_counter()
    try:
        validate_python_source(source, policy=policy)
    except CodeSafetyError as exc:
        return {
            "ok": False,
            "status": "policy_error",
            "return_code": None,
            "stdout": "",
            "stderr": str(exc),
            "timed_out": False,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
        }

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if workdir is None:
        temporary = tempfile.TemporaryDirectory(prefix="llm_code_exec_")
        root = Path(temporary.name)
    else:
        root = Path(workdir).resolve()
        root.mkdir(parents=True, exist_ok=True)

    script_path = root / "__agent_main__.py"
    script_path.write_text(source, encoding="utf-8")
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
    }
    command = [sys.executable, "-I", "-S", str(script_path)]
    process = subprocess.Popen(
        command,
        cwd=str(root),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=os.name == "posix",
        preexec_fn=_limited_preexec(policy),
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(
            input=str(stdin or ""),
            timeout=policy.timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        stdout, stderr = process.communicate()
        if not stdout and exc.stdout:
            stdout = exc.stdout if isinstance(exc.stdout, str) else exc.stdout.decode()
        if not stderr and exc.stderr:
            stderr = exc.stderr if isinstance(exc.stderr, str) else exc.stderr.decode()

    return_code = process.returncode
    result = {
        "ok": bool(not timed_out and return_code == 0),
        "status": "timeout" if timed_out else (
            "ok" if return_code == 0 else "runtime_error"
        ),
        "return_code": return_code,
        "stdout": _truncate(stdout or "", policy.max_output_chars),
        "stderr": _truncate(stderr or "", policy.max_output_chars),
        "timed_out": timed_out,
        "elapsed_ms": (time.perf_counter() - started) * 1000,
    }
    if temporary is not None:
        temporary.cleanup()
    return result


def python_executor(arguments: dict[str, Any]) -> dict[str, Any]:
    code = arguments.get("code")
    stdin = arguments.get("stdin", "")
    if not isinstance(code, str):
        raise CodeSafetyError("code must be a string")
    if not isinstance(stdin, str):
        raise CodeSafetyError("stdin must be a string")
    result = execute_python(code, stdin=stdin)
    result["result_text"] = result["stdout"].strip()
    return result


def python_executor_tool_spec() -> ToolSpec:
    return ToolSpec(
        name="python_executor",
        description=(
            "Execute a short Python program in a restricted offline subprocess. "
            "Use only safe standard-library imports and print the result."
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source code to execute.",
                },
                "stdin": {
                    "type": "string",
                    "description": "Optional standard input.",
                },
            },
            "required": ["code"],
            "additionalProperties": False,
        },
        handler=python_executor,
    )
