from llm_posttrain.evaluation.scorers import score_prediction


def test_exact() -> None:
    assert score_prediction("exact", "4500\n", "4500")["passed"] is True
    assert score_prediction("exact", "4501", "4500")["passed"] is False


def test_json_contains_and_fence() -> None:
    result = score_prediction(
        "json_contains",
        '```json\n{"category":"account","priority":"medium","extra":1}\n```',
        {"category": "account", "priority": "medium"},
    )
    assert result["passed"] is True


def test_python_syntax_and_symbols() -> None:
    result = score_prediction(
        "python_syntax",
        "```python\ndef add(a, b):\n    return a + b\n```",
        {"required_symbols": ["add"]},
    )
    assert result["passed"] is True

    bad = score_prediction("python_syntax", "def add(:\n    pass", {"required_symbols": ["add"]})
    assert bad["passed"] is False


def test_unknown_scorer() -> None:
    try:
        score_prediction("unknown", "x", "x")
    except KeyError:
        pass
    else:
        raise AssertionError("Expected unknown scorer to raise")
