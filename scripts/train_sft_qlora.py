from __future__ import annotations

import argparse
import inspect
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from llm_posttrain.config import load_yaml


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at line {line_number}")
            records.append(value)
    return records


def chat_token_ids(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    value = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
    if isinstance(value, dict):
        value = value["input_ids"]
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token_id) for token_id in value]


def text_token_ids(tokenizer, text: str) -> list[int]:
    value = tokenizer(text, add_special_tokens=False)["input_ids"]
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token_id) for token_id in value]


def find_subsequence(
    sequence: list[int],
    needle: list[int],
    start: int,
) -> int:
    if not needle:
        raise ValueError("cannot search for an empty token sequence")
    width = len(needle)
    for index in range(start, len(sequence) - width + 1):
        if sequence[index : index + width] == needle:
            return index
    return -1


def tokenize_record(
    tokenizer,
    record: dict[str, Any],
    max_seq_length: int,
) -> dict[str, Any]:
    messages = record["messages"]
    full_ids = chat_token_ids(
        tokenizer,
        messages,
        add_generation_prompt=False,
    )
    labels = [-100] * len(full_ids)
    assistant_count = 0
    cursor = 0
    assistant_header_ids = text_token_ids(
        tokenizer,
        "<|im_start|>assistant\n",
    )

    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        assistant_count += 1
        header_start = find_subsequence(full_ids, assistant_header_ids, cursor)
        if header_start < 0:
            raise ValueError(
                f"{record.get('id')}: assistant header not found at index {index}"
            )
        content = str(message.get("content", ""))
        content_ids = text_token_ids(tokenizer, content)
        content_start = find_subsequence(
            full_ids,
            content_ids,
            header_start + len(assistant_header_ids),
        )
        if content_start < 0:
            raise ValueError(
                f"{record.get('id')}: assistant content not found at index {index}"
            )
        content_end = content_start + len(content_ids)
        labels[content_start:content_end] = full_ids[content_start:content_end]
        cursor = content_end

    if assistant_count == 0 or not any(label != -100 for label in labels):
        raise ValueError(f"{record.get('id')}: no supervised assistant tokens")
    if len(full_ids) > max_seq_length:
        full_ids = full_ids[:max_seq_length]
        labels = labels[:max_seq_length]
    if not any(label != -100 for label in labels):
        raise ValueError(
            f"{record.get('id')}: max_seq_length truncated every assistant target"
        )
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "record_id": record.get("id"),
        "category": record.get("category"),
    }


class MessageDataset:
    def __init__(
        self,
        tokenizer,
        records: list[dict[str, Any]],
        max_seq_length: int,
    ) -> None:
        self.items = [
            tokenize_record(tokenizer, record, max_seq_length)
            for record in records
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        return {
            key: value
            for key, value in item.items()
            if key in {"input_ids", "attention_mask", "labels"}
        }

    def token_summary(self) -> dict[str, int]:
        return {
            "records": len(self.items),
            "input_tokens": sum(len(item["input_ids"]) for item in self.items),
            "supervised_tokens": sum(
                sum(label != -100 for label in item["labels"])
                for item in self.items
            ),
        }


@dataclass
class PaddingCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_id] * padding)
            attention_mask.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def build_model(
    model_path: str,
    use_4bit: bool,
    gradient_checkpointing: bool,
    init_adapter: str | None = None,
):
    if use_4bit and not torch.cuda.is_available():
        raise RuntimeError("4-bit QLoRA requires CUDA on this server")
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
    model.config.use_cache = False
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    if init_adapter:
        model = PeftModel.from_pretrained(
            model,
            init_adapter,
            is_trainable=True,
            local_files_only=True,
        )
    else:
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def make_training_args(
    output_dir: str,
    *,
    epochs: float,
    learning_rate: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    use_4bit: bool,
    seed: int,
) -> TrainingArguments:
    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "logging_steps": 1,
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "fp16": torch.cuda.is_available(),
        "gradient_checkpointing": True,
        "optim": "paged_adamw_8bit" if use_4bit else "adamw_torch",
        "report_to": [],
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "seed": seed,
        "data_seed": seed,
        "label_names": ["labels"],
    }
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    return TrainingArguments(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/base_model.yaml")
    parser.add_argument("--train-data", default="data/sft/train.jsonl")
    parser.add_argument("--valid-data", default="data/sft/valid.jsonl")
    parser.add_argument("--output-dir", default="models/adapters/sft_qwen3_1.7b")
    parser.add_argument("--init-adapter", default=None)
    parser.add_argument(
        "--summary-output",
        default="artifacts/sft_training_summary.json",
    )
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model_config = load_yaml(args.model_config)
    model_path = model_config["model"]["local_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_records = read_jsonl(args.train_data)
    valid_records = read_jsonl(args.valid_data)
    if args.max_train_samples is not None:
        train_records = train_records[: args.max_train_samples]
    if args.max_valid_samples is not None:
        valid_records = valid_records[: args.max_valid_samples]
    if not train_records or not valid_records:
        raise ValueError("train and validation data must both be non-empty")

    print(f"Train records: {len(train_records)}")
    print(f"Validation records: {len(valid_records)}")
    train_dataset = MessageDataset(tokenizer, train_records, args.max_seq_length)
    valid_dataset = MessageDataset(tokenizer, valid_records, args.max_seq_length)
    print(f"Train token summary: {train_dataset.token_summary()}")
    print(f"Valid token summary: {valid_dataset.token_summary()}")

    use_4bit = not args.no_4bit
    model = build_model(
        model_path,
        use_4bit=use_4bit,
        gradient_checkpointing=True,
        init_adapter=args.init_adapter,
    )
    training_args = make_training_args(
        args.output_dir,
        epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_4bit=use_4bit,
        seed=args.seed,
    )
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": valid_dataset,
        "data_collator": PaddingCollator(tokenizer),
    }
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    started = time.perf_counter()
    train_output = trainer.train()
    elapsed_seconds = time.perf_counter() - started
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    eval_metrics = trainer.evaluate()

    if hasattr(model, "get_nb_trainable_parameters"):
        trainable, total = model.get_nb_trainable_parameters()
    else:
        trainable = 0
        total = 0
        for parameter in model.parameters():
            total += parameter.numel()
            if parameter.requires_grad:
                trainable += parameter.numel()
    summary = {
        "model_path": model_path,
        "init_adapter": args.init_adapter,
        "adapter_output_dir": args.output_dir,
        "train_data": args.train_data,
        "valid_data": args.valid_data,
        "train_records": len(train_records),
        "valid_records": len(valid_records),
        "max_seq_length": args.max_seq_length,
        "assistant_only_labels": True,
        "quantization": "4bit_nf4_double_quant" if use_4bit else "none",
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_ratio": trainable / total if total else 0.0,
        "train_seconds": elapsed_seconds,
        "train_metrics": train_output.metrics,
        "eval_metrics": eval_metrics,
        "log_history": trainer.state.log_history,
    }
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
