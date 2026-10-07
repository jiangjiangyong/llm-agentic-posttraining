from llm_posttrain.agent.protocol import ToolCallParser
from llm_posttrain.agent.protocol_normalizer import (
    CanonicalProtocolNormalizer,
    ProtocolNormalizingBackend,
)
from llm_posttrain.agent.environment import CalculatorEnvironment
from llm_posttrain.tools.registry import build_default_registry
from scripts.run_protocol_normalization_isolation import episode_semantic_success


def test_canonical_output_is_preserved() -> None:
    normalizer = CanonicalProtocolNormalizer()
    raw = '<tool_call>{"name":"calculator","arguments":{"expression":"2 + 2"}}</tool_call>'

    result = normalizer.normalize(raw)

    assert result.status == "already_canonical"
    assert result.normalized_text == raw
    assert result.applied is False


def test_legacy_typed_json_is_only_wrapped_after_parse() -> None:
    normalizer = CanonicalProtocolNormalizer(ToolCallParser())
    raw = '{"type":"tool_call","name":"calculator","arguments":{"expression":"2 + 2"}}'

    result = normalizer.normalize(raw)

    assert result.status == "legacy_to_canonical"
    assert result.applied is True
    assert result.normalized_text == (
        '<tool_call>{"name":"calculator","arguments":{"expression":"2 + 2"}}'
        "</tool_call>"
    )
    assert '"type"' not in result.normalized_text


def test_normalizer_does_not_invent_a_call_from_prose() -> None:
    result = CanonicalProtocolNormalizer().normalize("The answer is 4.")

    assert result.status == "not_tool_call"
    assert result.normalized_text is None
    assert result.canonical_available is False


def test_malformed_tool_json_is_not_silently_repaired_by_normalizer() -> None:
    result = CanonicalProtocolNormalizer().normalize(
        '{"type":"tool_call","name":"calculator","arguments":}'
    )

    assert result.status == "invalid_tool_call"
    assert result.normalized_text is None


def test_semantic_success_can_be_read_from_trace_artifacts() -> None:
    assert episode_semantic_success({"trace": {"success": True}}) is True
    assert episode_semantic_success({"trace": {"success": False}}) is False


class FakeBackend:
    def __init__(self) -> None:
        self.outputs = iter(
            [
                '{"type":"tool_call","name":"calculator","arguments":{"expression":"2 + 2"}}',
                "4",
            ]
        )

    def generate(self, messages: list[dict[str, object]]) -> str:
        del messages
        return next(self.outputs)


def test_runtime_backend_normalizes_only_parser_recognized_tool_calls() -> None:
    backend = ProtocolNormalizingBackend(FakeBackend())
    task = {
        "id": "normalizer-runtime",
        "messages": [{"role": "user", "content": "Compute 2 + 2."}],
        "expected": {
            "tool_name": "calculator",
            "arguments": {"expression": "2 + 2"},
            "final_answer": "4",
        },
    }
    result = CalculatorEnvironment(build_default_registry()).run(
        task, backend.generate, task_id=task["id"]
    )
    assert result.semantic_success is True
    assert result.strict_success is True
    assert [item.status for item in backend.records] == [
        "legacy_to_canonical",
        "not_tool_call",
    ]
