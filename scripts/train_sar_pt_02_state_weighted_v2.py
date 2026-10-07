#!/usr/bin/env python3
"""One targeted SAR-PT-02 redesign: weighted state/action multi-task SFT.

The first one-pass joint SFT improved action replay but lost state transfer on
the source-disjoint internal holdout.  This variant keeps the same data,
initial adapter, one epoch, and optimizer family, while weighting state
extraction examples more heavily to test the task-interference hypothesis.
"""
from __future__ import annotations

import argparse
import inspect
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import Trainer

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_posttrain.config import load_yaml
from scripts.data_access_policy import assert_no_final_test_input
from scripts.train_sft_qlora import (
    PaddingCollator,
    build_model,
    read_jsonl,
    tokenize_record,
)
from scripts.train_sar_pt_02_sft_v1 import (
    NumericalAuditCallback,
    make_training_args,
    patch_trainer_accelerator,
    sha256_file,
)


class WeightedMessageDataset(Dataset):
    def __init__(
        self,
        tokenizer: Any,
        records: list[dict[str, Any]],
        max_seq_length: int,
        state_weight: float,
        action_weight: float,
    ) -> None:
        self.items: list[dict[str, Any]] = []
        for record in records:
            item = tokenize_record(tokenizer, record, max_seq_length)
            sample_type = str(record.get("sample_type", ""))
            if sample_type == "state_extraction_positive":
                weight = state_weight
            elif sample_type == "native_action_positive":
                weight = action_weight
            else:
                raise ValueError(f"unexpected sample_type: {sample_type}")
            item["loss_weight"] = float(weight)
            item["sample_type"] = sample_type
            self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": item["labels"],
            "loss_weight": item["loss_weight"],
        }

    def summary(self) -> dict[str, Any]:
        return {
            "records": len(self.items),
            "state_records": sum(
                item["sample_type"] == "state_extraction_positive"
                for item in self.items
            ),
            "action_records": sum(
                item["sample_type"] == "native_action_positive"
                for item in self.items
            ),
            "input_tokens": sum(len(item["input_ids"]) for item in self.items),
            "supervised_tokens": sum(
                sum(label != -100 for label in item["labels"])
                for item in self.items
            ),
        }


class WeightedPaddingCollator(PaddingCollator):
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = super().__call__(features)
        batch["loss_weight"] = torch.tensor(
            [float(feature["loss_weight"]) for feature in features],
            dtype=torch.float32,
        )
        return batch


class WeightedTokenTrainer(Trainer):
    """Compute a normalized per-example causal-LM loss before task weighting."""

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> Any:
        del num_items_in_batch
        labels = inputs.pop("labels")
        weights = inputs.pop("loss_weight").to(labels.device)
        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(shift_labels.shape)
        valid = shift_labels.ne(-100)
        per_example = (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        loss = (per_example * weights).sum() / weights.sum().clamp_min(1e-6)
        if return_outputs:
            return loss, outputs
        return loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/base_model.yaml")
    parser.add_argument(
        "--train-data",
        default="data/sar_pt_01b_state_aware_contract/train.jsonl",
    )
    parser.add_argument(
        "--valid-data",
        default="data/sar_pt_01b_state_aware_contract/internal_holdout.jsonl",
    )
    parser.add_argument(
        "--init-adapter",
        default="models/adapters/sar_pt_v2g_research_focus_state_replay_sft_seed20261004",
    )
    parser.add_argument(
        "--output-dir",
        default="models/adapters/sar_pt_02_state_weighted_v2_seed20261004",
    )
    parser.add_argument(
        "--summary-output",
        default="artifacts/sar_pt_02_state_weighted_v2_seed20261004.json",
    )
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--state-loss-weight", type=float, default=3.0)
    parser.add_argument("--action-loss-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    train_path = ROOT / args.train_data
    valid_path = ROOT / args.valid_data
    init_adapter = ROOT / args.init_adapter
    output_dir = ROOT / args.output_dir
    summary_path = ROOT / args.summary_output
    assert_no_final_test_input(args.train_data, "SAR-PT-02 weighted training")
    assert_no_final_test_input(args.valid_data, "SAR-PT-02 weighted internal holdout")
    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError("SAR-PT-01B data files are missing")
    if not init_adapter.exists():
        raise FileNotFoundError(f"init adapter is missing: {init_adapter}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    if args.state_loss_weight <= 0 or args.action_loss_weight <= 0:
        raise ValueError("loss weights must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_config = load_yaml(ROOT / args.model_config)
    model_path = ROOT / model_config["model"]["local_path"]
    train_records = read_jsonl(train_path)
    valid_records = read_jsonl(valid_path)
    if not train_records or not valid_records:
        raise ValueError("train and internal holdout must be non-empty")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = WeightedMessageDataset(
        tokenizer,
        train_records,
        args.max_seq_length,
        args.state_loss_weight,
        args.action_loss_weight,
    )
    valid_dataset = WeightedMessageDataset(
        tokenizer,
        valid_records,
        args.max_seq_length,
        args.state_loss_weight,
        args.action_loss_weight,
    )
    use_4bit = not args.no_4bit
    model = build_model(
        str(model_path),
        use_4bit=use_4bit,
        gradient_checkpointing=True,
        init_adapter=str(init_adapter),
    )
    patch_trainer_accelerator(init_scale=8192.0)
    training_args = make_training_args(
        str(output_dir),
        learning_rate=args.learning_rate,
        seed=args.seed,
        use_4bit=use_4bit,
    )
    callback = NumericalAuditCallback(init_scale=8192.0)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": valid_dataset,
        "data_collator": WeightedPaddingCollator(tokenizer),
        "callbacks": [callback],
    }
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = WeightedTokenTrainer(**trainer_kwargs)
    callback.trainer_ref = trainer
    started = time.perf_counter()
    train_output = trainer.train()
    train_seconds = time.perf_counter() - started
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    eval_metrics = trainer.evaluate()
    numerical = callback.summary()
    numerical["event_trace"] = callback.events
    if not numerical["instrumentation_hooks_called"]:
        raise RuntimeError("numerical instrumentation hooks did not run")

    trainable = 0
    total = 0
    for parameter in model.parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()
    summary = {
        "schema": "sar_pt_02_state_weighted_v2",
        "model_path": str(model_path.relative_to(ROOT)),
        "init_adapter": str(init_adapter.relative_to(ROOT)),
        "adapter_output_dir": str(output_dir.relative_to(ROOT)),
        "train_data": str(train_path.relative_to(ROOT)),
        "valid_data": str(valid_path.relative_to(ROOT)),
        "train_data_sha256": sha256_file(train_path),
        "valid_data_sha256": sha256_file(valid_path),
        "train_dataset_summary": train_dataset.summary(),
        "valid_dataset_summary": valid_dataset.summary(),
        "max_seq_length": args.max_seq_length,
        "learning_rate": args.learning_rate,
        "num_train_epochs": 1.0,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "state_loss_weight": args.state_loss_weight,
        "action_loss_weight": args.action_loss_weight,
        "gradient_scaler_init_scale": 8192.0,
        "quantization": "4bit_nf4_double_quant" if use_4bit else "none",
        "lora_parent_preserved": True,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_ratio": trainable / total if total else 0.0,
        "train_seconds": train_seconds,
        "train_metrics": train_output.metrics,
        "eval_metrics": eval_metrics,
        "numerical_audit": numerical,
        "redesign_reason": "The first joint SFT improved action replay but lost internal-holdout state transfer; this is one bounded state/action task-interference test.",
        "claim_boundary": [
            "This is a one-pass weighted SAR-PT-02 redesign initialized from M2g.",
            "The internal holdout remains the first promotion checkpoint; Rich native evaluation is not run by this script.",
            "Hard negatives are not included in this SFT loss.",
        ],
        "status": "TRAINED_WEIGHTED_INTERNAL_HOLDOUT_PENDING",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "schema": summary["schema"],
        "status": summary["status"],
        "adapter_output_dir": summary["adapter_output_dir"],
        "train_dataset_summary": summary["train_dataset_summary"],
        "valid_dataset_summary": summary["valid_dataset_summary"],
        "train_loss": summary["train_metrics"].get("train_loss"),
        "eval_loss": summary["eval_metrics"].get("eval_loss"),
        "numerical_audit": {
            key: value for key, value in numerical.items() if key != "event_trace"
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
