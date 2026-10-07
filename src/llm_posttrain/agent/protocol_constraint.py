from __future__ import annotations

from collections.abc import Sequence
import re
from typing import Any

import torch
from transformers import LogitsProcessor


class CanonicalCalculatorProtocolLogitsProcessor(LogitsProcessor):
    """Constrain the calculator tool-call shell and calculator value grammar.

    The registered tool name can be fixed by the caller, while the arithmetic
    expression remains model-generated. Value tokens are restricted to an
    arithmetic-character grammar so unrelated prose or tool-observation text
    cannot be copied into the JSON string. This is inference-time protocol
    control, not a semantic oracle and not evidence that the model learned the
    protocol by itself.

    The grammar is intentionally scoped to the calculator benchmark used by
    this experiment. A later generalization should derive the argument schema
    from the registered tool contract rather than silently reusing this class.
    """

    _PREFIX = '<tool_call>{"name":"'
    _MIDDLE = ',"arguments":{"expression":"'
    _SUFFIX = '}}'
    _CLOSE_TAG = "</tool_call>"
    _ARITHMETIC_VALUE_PATTERN = re.compile(r"^[0-9+*/%().\- ]+$")

    def __init__(
        self,
        tokenizer: Any,
        prompt_lengths: Sequence[int],
        *,
        tool_name: str | None = None,
        max_name_tokens: int = 16,
        max_expression_tokens: int = 64,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_lengths = tuple(int(value) for value in prompt_lengths)
        if not self.prompt_lengths:
            raise ValueError("prompt_lengths must not be empty")
        if max_name_tokens < 1 or max_expression_tokens < 1:
            raise ValueError("value token budgets must be positive")
        self.max_name_tokens = int(max_name_tokens)
        self.max_expression_tokens = int(max_expression_tokens)
        self.tool_name = tool_name.strip() if tool_name is not None else None
        if tool_name is not None and not self.tool_name:
            raise ValueError("tool_name must be non-empty when provided")

        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            raise ValueError("tokenizer must define eos_token_id")
        self.eos_token_id = int(eos_token_id)
        self.quote_token_id = self._single_token_id('"')
        self.prefix_ids = tuple(self._encode(self._PREFIX))
        self.middle_ids = tuple(self._encode(self._MIDDLE))
        self.suffix_ids = tuple(self._encode(self._SUFFIX))
        self.close_tag_ids = tuple(self._encode(self._CLOSE_TAG))
        self.name_ids = (
            tuple(self._encode(self.tool_name)) if self.tool_name is not None else ()
        )
        structural_ids = {
            *self.prefix_ids,
            *self.middle_ids,
            *self.suffix_ids,
            *self.close_tag_ids,
            *self.name_ids,
        }
        self.name_value_token_ids = tuple(
            token_id
            for token_id in self._build_value_token_ids(arithmetic_only=False)
            if token_id not in structural_ids
        )
        self.expression_value_token_ids = tuple(
            token_id
            for token_id in self._build_value_token_ids(arithmetic_only=True)
            if token_id not in structural_ids
        )
        if not self.name_value_token_ids or not self.expression_value_token_ids:
            raise ValueError("tokenizer has no usable JSON string value tokens")

    def _encode(self, text: str) -> list[int]:
        values = self.tokenizer.encode(text, add_special_tokens=False)
        if values and isinstance(values[0], list):
            values = values[0]
        return [int(value) for value in values]

    def _single_token_id(self, text: str) -> int:
        ids = self._encode(text)
        if len(ids) != 1:
            raise ValueError(f"{text!r} must be represented by one tokenizer token")
        return ids[0]

    def _decode_token(self, token_id: int) -> str:
        try:
            return str(
                self.tokenizer.decode(
                    [int(token_id)],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )
        except TypeError:
            return str(self.tokenizer.decode([int(token_id)], skip_special_tokens=False))

    def _build_value_token_ids(self, *, arithmetic_only: bool) -> list[int]:
        special_ids = {
            int(value)
            for value in getattr(self.tokenizer, "all_special_ids", [])
        }
        value_ids: list[int] = []
        for token_id in range(len(self.tokenizer)):
            if token_id in special_ids or token_id == self.quote_token_id:
                continue
            piece = self._decode_token(token_id)
            if not piece or '"' in piece:
                continue
            if "<tool_call>" in piece.lower() or "</tool_call>" in piece.lower():
                continue
            if any(ord(character) < 32 for character in piece):
                continue
            if arithmetic_only and not self._ARITHMETIC_VALUE_PATTERN.fullmatch(piece):
                continue
            value_ids.append(token_id)
        return value_ids

    @staticmethod
    def _prefix_matches(generated: list[int], expected: tuple[int, ...]) -> bool:
        width = min(len(generated), len(expected))
        return generated[:width] == list(expected[:width])

    def _fixed_next(
        self,
        generated: list[int],
        expected: tuple[int, ...],
    ) -> list[int] | None:
        if len(generated) > len(expected) or not self._prefix_matches(generated, expected):
            return [self.eos_token_id]
        if len(generated) < len(expected):
            return [expected[len(generated)]]
        return None

    def _prefix_next(
        self,
        generated: list[int],
        expected: tuple[int, ...],
    ) -> list[int] | None:
        if len(generated) < len(expected):
            if not self._prefix_matches(generated, expected):
                return [self.eos_token_id]
            return [expected[len(generated)]]
        if generated[: len(expected)] != list(expected):
            return [self.eos_token_id]
        return None

    def _allowed_tokens(self, generated: list[int]) -> list[int]:
        prefix_result = self._prefix_next(generated, self.prefix_ids)
        if prefix_result is not None:
            return prefix_result

        name_start = len(self.prefix_ids)
        if self.name_ids:
            name_result = self._prefix_next(
                generated[name_start:], self.name_ids
            )
            if name_result is not None:
                return name_result
            name_close_start = name_start + len(self.name_ids)
            middle_expected = (self.quote_token_id, *self.middle_ids)
            middle_result = self._prefix_next(
                generated[name_close_start:], middle_expected
            )
            if middle_result is not None:
                return middle_result
            expression_start = name_close_start + len(middle_expected)
            try:
                expression_close = generated.index(
                    self.quote_token_id, expression_start
                )
            except ValueError:
                expression_length = len(generated) - expression_start
                if expression_length == 0:
                    return list(self.expression_value_token_ids)
                if expression_length >= self.max_expression_tokens:
                    return [self.quote_token_id]
                return [*self.expression_value_token_ids, self.quote_token_id]

            suffix_start = expression_close + 1
            suffix = tuple(generated[suffix_start:])
            expected_suffix = self.suffix_ids + self.close_tag_ids
            suffix_result = self._prefix_next(list(suffix), expected_suffix)
            if suffix_result is not None:
                return suffix_result
            return [self.eos_token_id]

        try:
            name_close = generated.index(self.quote_token_id, name_start)
        except ValueError:
            if len(generated) == name_start:
                return list(self.name_value_token_ids)
            if len(generated) - name_start >= self.max_name_tokens:
                return [self.quote_token_id]
            return [*self.name_value_token_ids, self.quote_token_id]

        structural_start = name_close + 1
        structural = tuple(generated[structural_start:])
        middle_result = self._prefix_next(list(structural), self.middle_ids)
        if middle_result is not None:
            return middle_result

        expression_start = structural_start + len(self.middle_ids)
        try:
            expression_close = generated.index(self.quote_token_id, expression_start)
        except ValueError:
            if len(generated) == expression_start:
                return list(self.expression_value_token_ids)
            if len(generated) - expression_start >= self.max_expression_tokens:
                return [self.quote_token_id]
            return [*self.expression_value_token_ids, self.quote_token_id]

        suffix_start = expression_close + 1
        suffix = tuple(generated[suffix_start:])
        expected_suffix = self.suffix_ids + self.close_tag_ids
        suffix_result = self._prefix_next(list(suffix), expected_suffix)
        if suffix_result is not None:
            return suffix_result
        return [self.eos_token_id]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[0] > len(self.prompt_lengths):
            raise ValueError("prompt_lengths does not cover every generation row")
        constrained = torch.full_like(scores, float("-inf"))
        for row in range(input_ids.shape[0]):
            prompt_length = self.prompt_lengths[row]
            if prompt_length < 0 or prompt_length > input_ids.shape[1]:
                raise ValueError("prompt length is outside input_ids")
            generated = input_ids[row, prompt_length:].tolist()
            allowed = self._allowed_tokens([int(value) for value in generated])
            constrained[row, allowed] = scores[row, allowed]
        return constrained
