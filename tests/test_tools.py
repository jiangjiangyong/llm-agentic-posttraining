from llm_posttrain.tools.registry import build_default_registry


def test_calculator_success() -> None:
    result = build_default_registry().execute(
        "calculator",
        {"expression": "(2 + 3) * 4"},
    )
    assert result.ok is True
    assert result.output["result"] == 20
    assert result.output["result_text"] == "20"


def test_calculator_rejects_calls_and_names() -> None:
    registry = build_default_registry()
    call_result = registry.execute("calculator", {"expression": "__import__('os')"})
    name_result = registry.execute("missing", {})
    assert call_result.ok is False
    assert "unsupported" in (call_result.error or "")
    assert name_result.ok is False
    assert "unknown tool" in (name_result.error or "")


def test_registry_rejects_extra_arguments() -> None:
    result = build_default_registry().execute(
        "calculator",
        {"expression": "2 + 2", "extra": 1},
    )
    assert result.ok is False
    assert "unexpected arguments" in (result.error or "")
