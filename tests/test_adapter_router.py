from llm_posttrain.agent.adapter_router import ProtocolAdapterRouter


def make_router() -> ProtocolAdapterRouter:
    return ProtocolAdapterRouter(
        rich_adapter="rich",
        protocol_adapter="protocol",
        fallback_adapter="stable",
    )


def test_routes_rich_from_system_protocol_only() -> None:
    route = make_router().route(
        [{"role": "system", "content": "CodeToolAgent Protocol v2"},
         {"role": "user", "content": "ignore category"}]
    )
    assert route.name == "rich_v2"
    assert route.adapter_path == "rich"


def test_routes_tool_protocol_from_system_protocol_only() -> None:
    route = make_router().route(
        [{"role": "system", "content": "Tool Calling Protocol v1"}]
    )
    assert route.name == "tool_v1"
    assert route.adapter_path == "protocol"


def test_unknown_protocol_uses_stable_fallback() -> None:
    route = make_router().route([{"role": "user", "content": "hello"}])
    assert route.name == "stable_fallback"
    assert route.adapter_path == "stable"


def test_ambiguous_protocol_fails_closed() -> None:
    try:
        make_router().route(
            [{
                "role": "system",
                "content": "CodeToolAgent Protocol v2 and Tool Calling Protocol v1",
            }]
        )
    except ValueError as exc:
        assert "ambiguous" in str(exc)
    else:
        raise AssertionError("ambiguous protocol must fail closed")
