"""Run a guarded stage-wise DPO pilot for the research protocol.

This is intentionally separate from the historical DPO entrypoint. It keeps
policy/reference loading compatible with the project, but aborts before saving
an adapter if loss, gradients, gradient norm, or trainable parameters become
non-finite.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from llm_posttrain.config import load_yaml
from llm_posttrain.models.loader import model_input_device
from llm_posttrain.training.dpo import DPOCollator, DPODataset, dpo_loss, sequence_logprob
from scripts.train_dpo import load_sft_adapter, move_batch, read_jsonl

try:
    from scripts.data_access_policy import assert_no_final_test_input
except ModuleNotFoundError:
    from data_access_policy import assert_no_final_test_input

ROOT = Path(__file__).resolve().parents[1]
NL = chr(10)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value.detach()).all().item())


def finite_parameters(parameters: list[torch.nn.Parameter]) -> bool:
    return all(finite_tensor(parameter) for parameter in parameters)


def average_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    keys = sorted(metrics[0])
    return {key: sum(item[key] for item in metrics) / len(metrics) for key in keys}


def guarded_epoch(
    policy_model: Any,
    reference_model: Any,
    loader: DataLoader,
    *,
    beta: float,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_accumulation_steps: int,
    length_normalize: bool,
    max_grad_norm: float,
) -> tuple[dict[str, float], int, dict[str, int]]:
    training = optimizer is not None
    if training:
        policy_model.train()
        optimizer.zero_grad(set_to_none=True)
    else:
        policy_model.eval()
        reference_model.eval()
    metrics_list: list[dict[str, float]] = []
    updates = 0
    checks = {"loss": 0, "gradients": 0, "parameters": 0}
    total_steps = len(loader)
    for step, raw_batch in enumerate(loader):
        batch = move_batch(raw_batch, device)
        if training:
            loss = None
            policy_chosen = sequence_logprob(
                policy_model, batch["chosen"], length_normalize=length_normalize
            )
            policy_rejected = sequence_logprob(
                policy_model, batch["rejected"], length_normalize=length_normalize
            )
            with torch.no_grad():
                reference_chosen = sequence_logprob(
                    reference_model, batch["chosen"], length_normalize=length_normalize
                )
                reference_rejected = sequence_logprob(
                    reference_model, batch["rejected"], length_normalize=length_normalize
                )
            loss, metrics = dpo_loss(
                policy_chosen,
                policy_rejected,
                reference_chosen,
                reference_rejected,
                beta=beta,
            )
            checks["loss"] += 1
            if not finite_tensor(loss):
                raise FloatingPointError(f"non-finite DPO loss at step {step}")
            (loss / gradient_accumulation_steps).backward()
            trainable = [p for p in policy_model.parameters() if p.requires_grad]
            if not finite_parameters([p.grad for p in trainable if p.grad is not None]):
                raise FloatingPointError(f"non-finite gradient at step {step}")
            checks["gradients"] += 1
            should_update = (
                (step + 1) % gradient_accumulation_steps == 0
                or step + 1 == total_steps
            )
            if should_update:
                norm = torch.nn.utils.clip_grad_norm_(
                    trainable, max_norm=max_grad_norm
                )
                if not finite_tensor(norm):
                    raise FloatingPointError(f"non-finite gradient norm at step {step}")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if not finite_parameters(trainable):
                    raise FloatingPointError(f"non-finite parameter at step {step}")
                checks["parameters"] += 1
                updates += 1
        else:
            with torch.no_grad():
                policy_chosen = sequence_logprob(
                    policy_model, batch["chosen"], length_normalize=length_normalize
                )
                policy_rejected = sequence_logprob(
                    policy_model, batch["rejected"], length_normalize=length_normalize
                )
                reference_chosen = sequence_logprob(
                    reference_model, batch["chosen"], length_normalize=length_normalize
                )
                reference_rejected = sequence_logprob(
                    reference_model, batch["rejected"], length_normalize=length_normalize
                )
                loss, metrics = dpo_loss(
                    policy_chosen,
                    policy_rejected,
                    reference_chosen,
                    reference_rejected,
                    beta=beta,
                )
            checks["loss"] += 1
            if not finite_tensor(loss):
                raise FloatingPointError(f"non-finite validation loss at step {step}")
        metrics_list.append(metrics)
    return average_metrics(metrics_list), updates, checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="configs/base_model.yaml")
    parser.add_argument("--train-data", default="data/sar_pt_research_protocol_preference_v1/train.jsonl")
    parser.add_argument("--valid-data", default="data/sar_pt_research_protocol_preference_v1/holdout.jsonl")
    parser.add_argument("--policy-adapter", default="models/adapters/sar_pt_v2g_research_focus_state_replay_sft_seed20261004")
    parser.add_argument("--reference-adapter", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--no-length-normalize", action="store_true")
    args = parser.parse_args()
    if args.batch_size != 1:
        raise ValueError("guarded implementation currently requires --batch-size 1")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient accumulation must be positive")
    assert_no_final_test_input(args.train_data, "research preference DPO training")
    assert_no_final_test_input(args.valid_data, "research preference DPO validation")
    set_seed(args.seed)
    config = load_yaml(args.model_config)
    model_path = str(config["model"]["local_path"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
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
    train_dataset = DPODataset(tokenizer, train_records, args.max_seq_length)
    valid_dataset = DPODataset(tokenizer, valid_records, args.max_seq_length)
    collator = DPOCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=collator,
        num_workers=0,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    use_4bit = not args.no_4bit
    reference_adapter = args.reference_adapter or args.policy_adapter
    print("Loading guarded trainable DPO policy", flush=True)
    policy_model = load_sft_adapter(
        model_path, args.policy_adapter, use_4bit=use_4bit, trainable=True
    )
    print("Loading frozen DPO reference", flush=True)
    reference_model = load_sft_adapter(
        model_path, reference_adapter, use_4bit=use_4bit, trainable=False
    )
    device = model_input_device(policy_model)
    trainable = [p for p in policy_model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("policy has no trainable adapter parameters")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    history: list[dict[str, Any]] = []
    total_checks = {"loss": 0, "gradients": 0, "parameters": 0}
    started = time.perf_counter()
    for epoch in range(1, args.num_train_epochs + 1):
        train_metrics, updates, train_checks = guarded_epoch(
            policy_model,
            reference_model,
            train_loader,
            beta=args.beta,
            device=device,
            optimizer=optimizer,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            length_normalize=not args.no_length_normalize,
            max_grad_norm=args.max_grad_norm,
        )
        valid_metrics, _, valid_checks = guarded_epoch(
            policy_model,
            reference_model,
            valid_loader,
            beta=args.beta,
            device=device,
            optimizer=None,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            length_normalize=not args.no_length_normalize,
            max_grad_norm=args.max_grad_norm,
        )
        for key in total_checks:
            total_checks[key] += train_checks.get(key, 0) + valid_checks.get(key, 0)
        record = {
            "epoch": epoch,
            "updates": updates,
            "train": train_metrics,
            "validation": valid_metrics,
            "train_checks": train_checks,
            "validation_checks": valid_checks,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    elapsed = time.perf_counter() - started
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    summary = {
        "schema": "sar_pt_research_protocol_preference_v1_guarded_dpo",
        "model_path": model_path,
        "policy_init_adapter": args.policy_adapter,
        "reference_adapter": reference_adapter,
        "output_dir": args.output_dir,
        "train_data": args.train_data,
        "valid_data": args.valid_data,
        "train_records": len(train_records),
        "valid_records": len(valid_records),
        "max_seq_length": args.max_seq_length,
        "quantization": "4bit_nf4_double_quant" if use_4bit else "none",
        "dpo": {
            "beta": args.beta,
            "learning_rate": args.learning_rate,
            "epochs": args.num_train_epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_grad_norm": args.max_grad_norm,
            "length_normalize": not args.no_length_normalize,
        },
        "trainable_parameters": sum(p.numel() for p in trainable),
        "finite_guard": {
            "loss_gradient_parameter_checks": total_checks,
            "all_checks_passed": True,
        },
        "train_seconds": elapsed,
        "history": history,
        "final_test_accessed": False,
    }
    summary_path = ROOT / args.summary_output
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + NL, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
