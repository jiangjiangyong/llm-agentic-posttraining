from __future__ import annotations

import torch

from llm_posttrain.agent.protocol_constraint import (
    CanonicalCalculatorProtocolLogitsProcessor,
)


class FakeTokenizer:
    def __init__(self) -> None:
        self.eos_token_id = 99
        self.all_special_ids = [99]
        self._encode_map = {
            '<tool_call>{"name":"': [10, 11],
            'calculator': [2],
            ',"arguments":{"expression":"': [12, 13],
            '}}': [14],
            "</tool_call>": [15],
            '"': [1],
        }
        self._decode_map = {
            1: '"',
            2: "calculator",
            3: "125 * 36",
            4: "x",
            10: '<tool_call>{',
            11: '"name":"',
            12: ',"arguments":{',
            13: '"expression":"',
            14: '}}',
            15: '</tool_call>',
            99: '<|endoftext|>',
        }

    def __len__(self) -> int:
        return 100

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(self._encode_map[text])

    def decode(self, ids: list[int], **_: object) -> str:
        return "".join(self._decode_map.get(int(value), "") for value in ids)


def allowed(processor: CanonicalCalculatorProtocolLogitsProcessor, generated: list[int]) -> set[int]:
    input_ids = torch.tensor([[50] + generated], dtype=torch.long)
    scores = torch.zeros((1, 100), dtype=torch.float32)
    result = processor(input_ids, scores)
    return {index for index, value in enumerate(result[0]) if not torch.isneginf(value)}


def test_processor_forces_prefix_and_structure() -> None:
    processor = CanonicalCalculatorProtocolLogitsProcessor(FakeTokenizer(), [1])
    assert allowed(processor, []) == {10}
    assert allowed(processor, [10]) == {11}
    assert allowed(processor, [10, 11]) == {2, 3, 4}
    assert allowed(processor, [10, 11, 2]) == {2, 3, 4, 1}
    assert allowed(processor, [10, 11, 2, 1]) == {12}
    assert allowed(processor, [10, 11, 2, 1, 12]) == {13}
    assert allowed(processor, [10, 11, 2, 1, 12, 13]) == {3}
    assert allowed(processor, [10, 11, 2, 1, 12, 13, 3]) == {3, 1}
    assert allowed(processor, [10, 11, 2, 1, 12, 13, 3, 1]) == {14}
    assert allowed(processor, [10, 11, 2, 1, 12, 13, 3, 1, 14]) == {15}
    assert allowed(processor, [10, 11, 2, 1, 12, 13, 3, 1, 14, 15]) == {99}


def test_processor_rejects_invalid_structural_prefix() -> None:
    processor = CanonicalCalculatorProtocolLogitsProcessor(FakeTokenizer(), [1])
    assert allowed(processor, [10, 4]) == {99}


def test_processor_requires_nonempty_name_and_expression() -> None:
    processor = CanonicalCalculatorProtocolLogitsProcessor(FakeTokenizer(), [1])
    assert 1 not in allowed(processor, [10, 11])
    assert 1 not in allowed(processor, [10, 11, 2, 1, 12, 13])


def test_processor_can_fix_registered_tool_name_and_filter_non_arithmetic_text() -> None:
    processor = CanonicalCalculatorProtocolLogitsProcessor(
        FakeTokenizer(), [1], tool_name="calculator"
    )
    assert allowed(processor, [10, 11]) == {2}
    assert allowed(processor, [10, 11, 2]) == {1}
    assert allowed(processor, [10, 11, 2, 1, 12, 13]) == {3}
