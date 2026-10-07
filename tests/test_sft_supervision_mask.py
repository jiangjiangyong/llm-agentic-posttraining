from __future__ import annotations

from scripts.train_sft_qlora import tokenize_record


class FakeTokenizer:
    def _encode(self, text: str) -> list[int]:
        return [1000 + ord(char) for char in text]

    def __call__(self, text: str, *, add_special_tokens: bool):
        del add_special_tokens
        return {"input_ids": self._encode(text)}

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool = False,
    ):
        del enable_thinking
        rendered = ""
        for message in messages:
            rendered += (
                "<|im_start|>"
                + str(message["role"])
                + "\n"
                + str(message.get("content", ""))
                + "<|im_end|>\n"
            )
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        return self._encode(rendered) if tokenize else rendered


def test_supervise_false_masks_observed_assistant_prefix():
    tokenizer = FakeTokenizer()
    record = {
        "id": "masked-prefix-smoke",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {
                "role": "assistant",
                "content": "BAD_PREFIX_UNIQUE",
                "supervise": False,
            },
            {"role": "tool", "content": "observation"},
            {
                "role": "assistant",
                "content": "GOOD_TAIL_UNIQUE",
                "supervise": True,
            },
        ],
    }

    item = tokenize_record(tokenizer, record, max_seq_length=512)
    bad_ids = tokenizer("BAD_PREFIX_UNIQUE", add_special_tokens=False)[
        "input_ids"
    ]
    good_ids = tokenizer("GOOD_TAIL_UNIQUE", add_special_tokens=False)[
        "input_ids"
    ]
    bad_start = item["input_ids"].index(bad_ids[0])
    good_start = item["input_ids"].index(good_ids[0])

    assert all(
        label == -100
        for label in item["labels"][bad_start : bad_start + len(bad_ids)]
    )
    assert item["labels"][good_start : good_start + len(good_ids)] == good_ids
