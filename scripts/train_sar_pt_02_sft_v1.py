#!/usr/bin/env python3
"""SAR-PT-02 deterministic state-aware multi-task SFT v1.

This is one bounded smoke only.  It initializes from the frozen M2g adapter,
trains one pass over the SAR-PT-01B state/action records, evaluates the
internal holdout, and writes numerical instrumentation.  It does not run Rich
evaluation; the next gate must inspect the holdout first.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from accelerate.utils import GradScalerKwargs
from transformers import Trainer, TrainerCallback, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_posttrain.config import load_yaml
from scripts.data_access_policy import assert_no_final_test_input
from scripts.train_sft_qlora import (
    MessageDataset,
    PaddingCollator,
    build_model,
    read_jsonl,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value.detach()).all().item())


class NumericalAuditCallback(TrainerCallback):
    """Collects in-memory per-optimizer-step finite/delta evidence.

    The callback does not print every event.  It writes the aggregate trace
    once after training so normal logs remain low volume.
    """

    def __init__(self, init_scale: float) -> None:
        self.init_scale = init_scale
        self.trainer_ref: Trainer | None = None
        self.events: list[dict[str, Any]] = []
        self._current: dict[str, Any] | None = None
        self._before_delta: float | None = None
        self._microstep = 0

    @staticmethod
    def trainable_parameters(model: Any) -> list[torch.nn.Parameter]:
        return [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]

    @classmethod
    def parameter_delta_scalar(cls, model: Any) -> float:
        total = 0.0
        for parameter in cls.trainable_parameters(model):
            value = parameter.detach().float()
            total += float(value.abs().sum().item())
        return total

    @classmethod
    def parameters_finite(cls, model: Any) -> tuple[bool, int]:
        bad = 0
        for parameter in cls.trainable_parameters(model):
            if not finite_tensor(parameter):
                bad += 1
        return bad == 0, bad

    @classmethod
    def gradients_finite(cls, model: Any) -> tuple[bool, int]:
        bad = 0
        for parameter in cls.trainable_parameters(model):
            if parameter.grad is None:
                continue
            if not finite_tensor(parameter.grad):
                bad += 1
        return bad == 0, bad

    @staticmethod
    def optimizer_state_finite(optimizer: Any) -> tuple[bool, int]:
        bad = 0
        for state in optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor) and not finite_tensor(value):
                    bad += 1
        return bad == 0, bad

    def on_pre_optimizer_step(
        self,
        args: Any,
        state: Any,
        control: Any,
        model: Any = None,
        optimizer: Any = None,
        **kwargs: Any,
    ) -> None:
        del args, control, kwargs
        self._microstep += 1
        if model is None:
            return
        grad_ok, bad_grads = self.gradients_finite(model)
        self._before_delta = self.parameter_delta_scalar(model)
        self._current = {
            "microstep": self._microstep,
            "global_step_before": int(getattr(state, "global_step", 0))
            if state is not None else None,
            "pre_optimizer_gradients_finite": grad_ok,
            "nonfinite_gradients": bad_grads,
            "optimizer_step_hook_seen": False,
        }
        del optimizer

    def on_optimizer_step(
        self,
        args: Any,
        state: Any,
        control: Any,
        model: Any = None,
        optimizer: Any = None,
        **kwargs: Any,
    ) -> None:
        del args, state, control, model, optimizer, kwargs
        if self._current is not None:
            self._current["optimizer_step_hook_seen"] = True

    def on_step_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        model: Any = None,
        optimizer: Any = None,
        **kwargs: Any,
    ) -> None:
        del args, control, kwargs
        if self._current is None or model is None:
            return
        params_ok, bad_params = self.parameters_finite(model)
        optimizer_ok, bad_optimizer = self.optimizer_state_finite(optimizer)
        after_delta = self.parameter_delta_scalar(model)
        before_delta = self._before_delta or 0.0
        scaler_scale = None
        if self.trainer_ref is not None:
            scaler = getattr(self.trainer_ref.accelerator, "scaler", None)
            if scaler is not None and hasattr(scaler, "get_scale"):
                scaler_scale = float(scaler.get_scale())
        event = {
            **self._current,
            "global_step_after": int(getattr(state, "global_step", 0)),
            "post_step_parameters_finite": params_ok,
            "nonfinite_parameters": bad_params,
            "optimizer_state_finite": optimizer_ok,
            "nonfinite_optimizer_state": bad_optimizer,
            "parameter_delta_abs": abs(after_delta - before_delta),
            "optimizer_executed": abs(after_delta - before_delta) > 0.0,
            "grad_scaler_scale": scaler_scale,
            "grad_scaler_init_scale_requested": self.init_scale,
            "found_inf": not (
                self._current["pre_optimizer_gradients_finite"]
                and params_ok
                and optimizer_ok
            ),
        }
        self.events.append(event)
        self._current = None
        self._before_delta = None

    def summary(self) -> dict[str, Any]:
        nonfinite = [event for event in self.events if event["found_inf"]]
        skipped = [
            event for event in nonfinite
            if not event["optimizer_executed"]
        ]
        return {
            "instrumentation_hooks_called": bool(self.events),
            "events": len(self.events),
            "nonfinite_microsteps": len(nonfinite),
            "skipped_nonfinite_microsteps": len(skipped),
            "nonfinite_parameter_events": sum(
                int(event["nonfinite_parameters"]) for event in self.events
            ),
            "max_nonfinite_gradients": max(
                (int(event["nonfinite_gradients"]) for event in self.events),
                default=0,
            ),
            "requested_grad_scaler_init_scale": self.init_scale,
            "observed_grad_scaler_scales": sorted({
                event["grad_scaler_scale"]
                for event in self.events
                if event["grad_scaler_scale"] is not None
            }),
            "all_post_step_parameters_finite": all(
                event["post_step_parameters_finite"] for event in self.events
            ),
            "all_optimizer_states_finite": all(
                event["optimizer_state_finite"] for event in self.events
            ),
        }


def patch_trainer_accelerator(init_scale: float) -> None:
    """Install GradScalerKwargs before Trainer constructs Accelerator."""
    import transformers.trainer as trainer_module

    original = trainer_module.Accelerator

    if getattr(original, "_sar_pt_02_patched", False):
        return

    def wrapped_accelerator(*args: Any, **kwargs: Any) -> Any:
        handlers = list(kwargs.get("kwargs_handlers") or [])
        handlers.append(GradScalerKwargs(init_scale=init_scale))
        kwargs["kwargs_handlers"] = handlers
        return original(*args, **kwargs)

    wrapped_accelerator._sar_pt_02_patched = True
    trainer_module.Accelerator = wrapped_accelerator


def make_training_args(
    output_dir: str,
    *,
    learning_rate: float,
    seed: int,
    use_4bit: bool,
) -> TrainingArguments:
    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": 1.0,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": learning_rate,
        "logging_steps": 10,
        "save_strategy": "no",
        "report_to": [],
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "seed": seed,
        "data_seed": seed,
        "label_names": ["labels"],
        "fp16": torch.cuda.is_available(),
        "gradient_checkpointing": True,
        "optim": "paged_adamw_8bit" if use_4bit else "adamw_torch",
        "max_grad_norm": 1.0,
        "warmup_ratio": 0.0,
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
        default="models/adapters/sar_pt_02_state_aware_sft_v1_seed20261004",
    )
    parser.add_argument(
        "--summary-output",
        default="artifacts/sar_pt_02_state_aware_sft_v1_seed20261004.json",
    )
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    train_path = ROOT / args.train_data
    valid_path = ROOT / args.valid_data
    init_adapter = ROOT / args.init_adapter
    output_dir = ROOT / args.output_dir
    summary_path = ROOT / args.summary_output

    assert_no_final_test_input(args.train_data, "SAR-PT-02 training")
    assert_no_final_test_input(args.valid_data, "SAR-PT-02 internal holdout")
    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError("SAR-PT-01B data files are missing")
    if not init_adapter.exists():
        raise FileNotFoundError(f"init adapter is missing: {init_adapter}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")

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
    if sum(row.get("sample_type") == "state_extraction_positive" for row in train_records) != sum(
        row.get("sample_type") == "native_action_positive" for row in train_records
    ):
        raise ValueError("state/action sampling is not equal")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = MessageDataset(tokenizer, train_records, args.max_seq_length)
    valid_dataset = MessageDataset(tokenizer, valid_records, args.max_seq_length)
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
        "data_collator": PaddingCollator(tokenizer),
        "callbacks": [callback],
    }
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
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
        "schema": "sar_pt_02_state_aware_sft_v1",
        "model_path": str(model_path.relative_to(ROOT)),
        "init_adapter": str(init_adapter.relative_to(ROOT)),
        "adapter_output_dir": str(output_dir.relative_to(ROOT)),
        "parent_adapter_sha256_not_directory": "directory_hash_not_used",
        "train_data": str(train_path.relative_to(ROOT)),
        "valid_data": str(valid_path.relative_to(ROOT)),
        "train_data_sha256": sha256_file(train_path),
        "valid_data_sha256": sha256_file(valid_path),
        "train_records": len(train_records),
        "valid_records": len(valid_records),
        "state_records": sum(
            row.get("sample_type") == "state_extraction_positive"
            for row in train_records
        ),
        "action_records": sum(
            row.get("sample_type") == "native_action_positive"
            for row in train_records
        ),
        "max_seq_length": args.max_seq_length,
        "learning_rate": args.learning_rate,
        "num_train_epochs": 1.0,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
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
        "claim_boundary": [
            "This is a one-pass SAR-PT-02 smoke initialized from M2g.",
            "The internal holdout is the first promotion checkpoint; Rich native evaluation is intentionally not run by this script.",
            "Numerical instrumentation is recorded per optimizer step without per-step log printing.",
            "No guard-on result is included in this artifact.",
        ],
        "status": "TRAINED_INTERNAL_HOLDOUT_PENDING",
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
        "train_records": summary["train_records"],
        "valid_records": summary["valid_records"],
        "train_loss": summary["train_metrics"].get("train_loss"),
        "eval_loss": summary["eval_metrics"].get("eval_loss"),
        "numerical_audit": {
            key: value
            for key, value in numerical.items()
            if key != "event_trace"
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
