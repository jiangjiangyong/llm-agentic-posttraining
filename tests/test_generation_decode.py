from llm_posttrain.models.loader import decode_generation


class FakeTokenizer:
    def batch_decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens is False
        return ['<tool_call>{"name":"calculator"}</tool_call><|im_end|>']


def test_generation_decode_preserves_tool_tags() -> None:
    text = decode_generation(FakeTokenizer(), [[1, 2, 3]])
    assert text == '<tool_call>{"name":"calculator"}</tool_call>'
