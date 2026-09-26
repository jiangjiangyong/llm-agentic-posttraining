from pathlib import Path

from fastapi.testclient import TestClient

from llm_posttrain.models.loader import GenerationResult
from llm_posttrain.serving.app import create_app


class FakeRunner:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.index = 0

    def generate_with_stats(self, messages: list[dict[str, object]]) -> GenerationResult:
        del messages
        return GenerationResult(
            text="chat response",
            prompt_tokens=3,
            completion_tokens=2,
            latency_ms=1.5,
        )

    def generate(self, messages: list[dict[str, object]]) -> str:
        del messages
        response = self.responses[self.index]
        self.index += 1
        return response


def make_app() -> TestClient:
    app = create_app(
        FakeRunner(
            [
                '<tool_call>{"name":"calculator","arguments":{"expression":"2 + 2"}}</tool_call>',
                "4",
            ]
        ),
        backend_name="test",
        model_path="test-model",
        web_dir=Path(__file__).resolve().parents[1] / "web",
    )
    return TestClient(app)


def test_health_and_chat_completion() -> None:
    client = make_app()
    health = client.get("/health")
    chat = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert health.status_code == 200
    assert health.json()["backend"] == "test"
    assert chat.status_code == 200
    assert chat.json()["choices"][0]["message"]["content"] == "chat response"
    assert chat.json()["usage"]["total_tokens"] == 5


def test_agent_endpoint_runs_tool_runtime() -> None:
    client = make_app()
    response = client.post(
        "/v1/agent/run",
        json={"task_id": "api_test", "messages": [{"role": "user", "content": "calculate 2 + 2"}]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task_id"] == "api_test"
    assert body["success"] is True
    assert body["final_answer"] == "4"
