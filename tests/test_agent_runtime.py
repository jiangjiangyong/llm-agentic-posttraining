from llm_posttrain.agent.protocol import ToolCallParser
from llm_posttrain.agent.runtime import AgentRuntime
from llm_posttrain.tools.registry import build_default_registry


class FakeBackend:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.seen_messages: list[list[dict[str, object]]] = []
        self.index = 0

    def generate(self, messages: list[dict[str, object]]) -> str:
        self.seen_messages.append(messages)
        response = self.responses[self.index]
        self.index += 1
        return response


def test_parser_accepts_canonical_tool_call() -> None:
    parsed = ToolCallParser().parse(
        '<tool_call>{"name":"calculator","arguments":{"expression":"2 + 2"}}</tool_call>'
    )
    assert parsed.kind == "tool_call"
    assert parsed.tool_call is not None
    assert parsed.tool_call.name == "calculator"
    assert parsed.tool_call.arguments == {"expression": "2 + 2"}


def test_parser_accepts_json_tool_call_with_trailing_generation_noise() -> None:
    parsed = ToolCallParser().parse(
        '{"type":"tool_call","name":"calculator","arguments":'
        '{"expression":"125 * 36"}} trailing text'
    )
    assert parsed.kind == "tool_call"
    assert parsed.tool_call is not None
    assert parsed.tool_call.name == "calculator"
    assert parsed.tool_call.arguments == {"expression": "125 * 36"}


def test_parser_repairs_missing_expression_quote() -> None:
    parsed = ToolCallParser().parse(
        '{"type":"tool_call","name":"calculator","arguments":'
        '{"expression":"125 * 36}} trailing text'
    )
    assert parsed.kind == "tool_call"
    assert parsed.tool_call is not None
    assert parsed.tool_call.name == "calculator"
    assert parsed.tool_call.arguments == {"expression": "125 * 36"}


def test_parser_rejects_unterminated_block() -> None:
    parsed = ToolCallParser().parse(
        '<tool_call>{"name":"calculator","arguments":{}}'
    )
    assert parsed.kind == "invalid"


def test_runtime_executes_tool_then_returns_final() -> None:
    backend = FakeBackend(
        [
            '<tool_call>{"name":"calculator","arguments":{"expression":"2 + 2"}}</tool_call>',
            "4",
        ]
    )
    result = AgentRuntime(
        backend=backend,
        registry=build_default_registry(),
        max_steps=3,
    ).run(
        [{"role": "user", "content": "calculate 2 + 2"}],
        task_id="runtime_test",
    )
    assert result.success is True
    assert result.final_answer == "4"
    assert [event.kind for event in result.trace.events] == [
        "model_output",
        "tool_call",
        "tool_result",
        "model_output",
        "final_answer",
    ]
    assert backend.seen_messages[1][-1]["role"] == "tool"
    assert "tool_observation" in str(backend.seen_messages[1][-1]["content"])


def test_runtime_stops_on_invalid_output() -> None:
    backend = FakeBackend(["<tool_call>{bad json</tool_call>"])
    result = AgentRuntime(
        backend=backend,
        registry=build_default_registry(),
    ).run([{"role": "user", "content": "calculate"}])
    assert result.success is False
    assert result.trace.events[-1].kind == "parse_error"
