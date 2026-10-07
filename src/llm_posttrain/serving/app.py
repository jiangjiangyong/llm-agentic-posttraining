from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from llm_posttrain.agent.code_environment import CodeAgentEnvironment
from llm_posttrain.agent.protocol_normalizer import ProtocolNormalizingBackend
from llm_posttrain.agent.runtime import AgentRuntime
from llm_posttrain.models.loader import GenerationResult
from llm_posttrain.tools.registry import build_default_registry


class Message(BaseModel):
    role: str
    content: str
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name is not None:
            value["name"] = self.name
        return value


class ChatCompletionRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)


class AgentRunRequest(ChatCompletionRequest):
    task_id: str | None = None
    normalize_protocol: bool = False


class CodeAgentRunRequest(BaseModel):
    task: dict[str, Any]
    task_id: str | None = None
    normalize_protocol: bool = False


def _message_dicts(messages: list[Message]) -> list[dict[str, Any]]:
    return [message.to_dict() for message in messages]


def create_app(
    runner: Any,
    *,
    backend_name: str,
    model_path: str,
    adapter_path: str | None = None,
    max_steps: int = 3,
    web_dir: str | Path = "web",
) -> FastAPI:
    app = FastAPI(title="LLM Agentic Runtime", version="0.1.0")
    web_root = Path(web_dir)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ready": True,
            "backend": backend_name,
            "model_path": model_path,
            "adapter_path": adapter_path,
            "max_steps": max_steps,
            "capabilities": [
                "chat_completion",
                "calculator_agent",
                "code_agent",
                "protocol_normalization",
            ],
        }

    @app.get("/", response_class=FileResponse)
    def index() -> FileResponse:
        index_path = web_root / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="web demo is unavailable")
        return FileResponse(index_path)

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
        messages = _message_dicts(request.messages)
        try:
            result: GenerationResult = runner.generate_with_stats(messages)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
            },
            "latency_ms": result.latency_ms,
            "backend": backend_name,
        }

    @app.post("/v1/agent/run")
    def agent_run(request: AgentRunRequest) -> dict[str, Any]:
        messages = _message_dicts(request.messages)
        backend = (
            ProtocolNormalizingBackend(runner)
            if request.normalize_protocol
            else runner
        )
        runtime = AgentRuntime(
            backend=backend,
            registry=build_default_registry(),
            max_steps=max_steps,
        )
        try:
            result = runtime.run(
                messages,
                task_id=request.task_id or f"api-{uuid.uuid4().hex}",
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        payload = {
            "task_id": result.trace.task_id,
            "success": result.success,
            "final_answer": result.final_answer,
            "error": result.error,
            "elapsed_ms": result.elapsed_ms,
            "trace": result.trace.to_dict(),
            "backend": backend_name,
        }
        if isinstance(backend, ProtocolNormalizingBackend):
            payload["protocol_normalization"] = [
                item.to_dict() for item in backend.records
            ]
        return payload

    @app.post("/v1/code-agent/run")
    def code_agent_run(request: CodeAgentRunRequest) -> dict[str, Any]:
        task = dict(request.task)
        messages = task.get("prompt_messages", task.get("messages"))
        if not isinstance(messages, list) or not messages:
            raise HTTPException(
                status_code=422,
                detail="task must contain a non-empty prompt_messages or messages list",
            )
        backend = (
            ProtocolNormalizingBackend(runner)
            if request.normalize_protocol
            else runner
        )
        environment = CodeAgentEnvironment(max_steps=max_steps)
        try:
            result = environment.run(
                task,
                backend.generate if isinstance(backend, ProtocolNormalizingBackend) else runner.generate,
                task_id=request.task_id or str(task.get("id", uuid.uuid4().hex)),
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        payload = result.to_dict()
        payload["final_answer"] = result.final_output
        payload["trace"] = {
            "steps": result.steps,
            "tool_calls": result.tool_calls,
            "tool_results": result.tool_results,
            "workspace_files": result.workspace_files,
        }
        payload["backend"] = backend_name
        if isinstance(backend, ProtocolNormalizingBackend):
            payload["protocol_normalization"] = [
                item.to_dict() for item in backend.records
            ]
        return payload

    return app
