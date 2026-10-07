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
    assert "code_agent" in health.json()["capabilities"]
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


def test_agent_endpoint_can_normalize_legacy_tool_call() -> None:
    app = create_app(
        FakeRunner(
            [
                '{"type":"tool_call","name":"calculator","arguments":{"expression":"2 + 2"}}',
                "4",
            ]
        ),
        backend_name="test",
        model_path="test-model",
        web_dir=Path(__file__).resolve().parents[1] / "web",
    )
    client = TestClient(app)
    response = client.post(
        "/v1/agent/run",
        json={
            "task_id": "api_normalized_test",
            "normalize_protocol": True,
            "messages": [{"role": "user", "content": "calculate 2 + 2"}],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["final_answer"] == "4"
    assert [item["status"] for item in body["protocol_normalization"]] == [
        "legacy_to_canonical",
        "not_tool_call",
    ]


def test_code_agent_endpoint_runs_multi_tool_environment_and_resolver() -> None:
    app = create_app(
        FakeRunner(
            [
                (
                    '<tool_call>{"name":"mock_search","arguments":'
                    '{"query":"python executor timeout policy","top_k":1}}'
                    "</tool_call>"
                ),
                (
                    '<tool_call>{"name":"write_file","arguments":'
                    '{"path":"research/note.md","content_ref":"result_snippet",'
                    '"marker_ref":"context_packet"}}</tool_call>'
                ),
                (
                    '<tool_call>{"name":"read_file","arguments":'
                    '{"path":"research/note.md"}}</tool_call>'
                ),
                "READY",
            ]
        ),
        backend_name="test",
        model_path="test-model",
        max_steps=5,
        web_dir=Path(__file__).resolve().parents[1] / "web",
    )
    client = TestClient(app)
    task = {
        "id": "code_api_task",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Search, write, read, and finish. "
                    "Context packet API42: cedar, river."
                ),
            }
        ],
        "expected": {
            "required_tools": ["mock_search", "write_file", "read_file"],
            "expected_tool_calls": 3,
            "expected_arguments": {
                "mock_search": {"query": "python executor timeout policy"},
                "write_file": {"path": "research/note.md"},
                "read_file": {"path": "research/note.md"},
            },
            "execution_tools": ["mock_search", "write_file", "read_file"],
            "file_checks": [
                {
                    "path": "research/note.md",
                    "contains": ["three second timeout", "Case marker: API42"],
                }
            ],
            "final_answer_contains": ["READY"],
        },
    }

    response = client.post(
        "/v1/code-agent/run",
        json={"task_id": "code_api_test", "task": task},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task_id"] == "code_api_test"
    assert body["semantic_success"] is True
    assert body["final_answer"] == "READY"
    assert body["trace"]["workspace_files"] == ["research/note.md"]
    assert body["steps"][1]["content_ref_resolutions"][0]["applied"] is True


def test_code_agent_endpoint_rejects_missing_task_messages() -> None:
    client = make_app()

    response = client.post("/v1/code-agent/run", json={"task": {}})

    assert response.status_code == 422
