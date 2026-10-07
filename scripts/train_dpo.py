from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from llm_posttrain.config import load_yaml
from llm_posttrain.models.loader import model_input_device
from llm_posttrain.training.dpo import DPOCollator, DPODataset, dpo_loss, sequence_logprob
try:
    from scripts.data_access_policy import assert_no_final_test_input
except ModuleNotFoundError:
    from data_access_policy import assert_no_final_test_input


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at line {line_number} in {path}")
            records.append(value)
    return records


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_base_model(model_path: str, *, use_4bit: bool, trainable: bool):
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }
    if use_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    if torch.cuda.is_available():
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    if use_4bit and trainable:
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=True,
        )
    return base_model


def load_sft_adapter(
    model_path: str,
    adapter_path: str,
    *,
    use_4bit: bool,
    trainable: bool,
):
    base_model = load_base_model(model_path, use_4bit=use_4bit, trainable=trainable)
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=trainable,
        local_files_only=True,
    )
    model.config.use_cache = False
    if trainable:
        model.train()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.print_trainable_parameters()
    else:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return model


def move_batch(batch: dict[str, dict[str, torch.Tensor]], device: torch.device):
    return {
        side: {key: value.to(device) for key, value in values.items()}
        for side, values in batch.items()
    }


def compute_batch(
    policy_model: Any,
    reference_model: Any,
    batch: dict[str, dict[str, torch.Tensor]],
    *,
    beta: float,
    length_normalize: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    policy_chosen = sequence_logprob(policy_model, batch["chosen"], length_normalize=length_normalize)
    policy_rejected = sequence_logprob(policy_model, batch["rejected"], length_normalize=length_normalize)
    with torch.no_grad():
        reference_chosen = sequence_logprob(reference_model, batch["chosen"], length_normalize=length_normalize)
        reference_rejected = sequence_logprob(reference_model, batch["rejected"], length_normalize=length_normalize)
    return dpo_loss(
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
        beta=beta,
    )


def average_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    keys = metrics[0].keys()
    return {key: sum(item[key] for item in metrics) / len(metrics) for key in keys}


def run_epoch(
    policy_model: Any,
    reference_model: Any,
    loader: DataLoader,
    *,
    beta: float,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_accumulation_steps: int,
    length_normalize: bool,
) -> tuple[dict[str, float], int]:
    training = optimizer is not None
    if training:
        policy_model.train()
        optimizer.zero_grad(set_to_none=True)
    else:
        policy_model.eval()
        reference_model.eval()
    epoch_metrics: list[dict[str, float]] = []
    updates = 0
    total_steps = len(loader)
    for step, raw_batch in enumerate(loader):
        batch = move_batch(raw_batch, device)
        if training:
            loss, metrics = compute_batch(policy_model, reference_model, batch, beta=beta, length_normalize=length_normalize)
            (loss / gradient_accumulation_steps).backward()
            should_update = ((step + 1) % gradient_accumulation_steps == 0) or (step + 1 == total_steps)
            if should_update:
                trainable_params = [parameter for parameter in policy_model.parameters() if parameter.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
        else:
            with torch.no_grad():
                loss, metrics = compute_batch(policy_model, reference_model, batch, beta=beta, length_normalize=length_normalize)
        epoch_metrics.append(metrics)
    return average_metrics(epoch_metrics), updates


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a LoRA SFT adapter with a manual DPO loop.")
    parser.add_argument("--model-config", default="configs/base_model.yaml")
    parser.add_argument("--train-data", default="data/preference/train.jsonl")
    parser.add_argument("--valid-data", default="data/preference/valid.jsonl")
    parser.add_argument("--policy-adapter", default="models/adapters/sft_qwen3_1.7b")
    parser.add_argument("--reference-adapter", default=None)
    parser.add_argument("--output-dir", default="models/adapters/dpo_qwen3_1.7b")
    parser.add_argument("--summary-output", default="artifacts/dpo_training_summary.json")
    parser.add_argument("--max-seq-length", type=int, default=768)
    parser.add_argument("--num-train-epochs", type=int, default= 1)
    parser.add_argument("--learning-rate", type=float, default= 1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--no-length-normalize", action="store_true")
    args = parser.parse_args()
    assert_no_final_test_input(args.train_data, "DPO training")
    assert_no_final_test_input(args.valid_data, "DPO validation")
    if args.batch_size != 1:
        raise ValueError("This memory-safe implementation currently requires --batch-size 1")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient accumulation must be positive")

    set_seed(args.seed)
    model_config = load_yaml(args.model_config)
    model_path = str(model_config["model"]["local_path"])
    reference_adapter = args.reference_adapter or args.policy_adapter
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

    train_dataset = DPODataset(tokenizer, train_records, args.max_seq_length)
    valid_dataset = DPODataset(tokenizer, valid_records, args.max_seq_length)
    print(f"Train records: {len(train_dataset)}")
    print(f"Validation records: {len(valid_dataset)}")
    print(f"Train token summary: {train_dataset.token_summary()}")
    print(f"Valid token summary: {valid_dataset.token_summary()}")
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
    print("Loading trainable policy from the SFT adapter")
    policy_model = load_sft_adapter(
        model_path,
        args.policy_adapter,
        use_4bit=use_4bit,
        trainable=True,
    )
    print("Loading frozen reference from the SFT adapter")
    reference_model = load_sft_adapter(
        model_path,
        reference_adapter,
        use_4bit=use_4bit,
        trainable=False,
    )
    device = model_input_device(policy_model)
    trainable_params = [parameter for parameter in policy_model.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise RuntimeError("Policy has no trainable adapter parameters")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, args.num_train_epochs + 1):
        train_metrics, updates = run_epoch(
            policy_model,
            reference_model,
            train_loader,
            beta=args.beta,
            length_normalize=not args.no_length_normalize,
            device=device,
            optimizer=optimizer,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        valid_metrics, _ = run_epoch(
            policy_model,
            reference_model,
            valid_loader,
            beta=args.beta,
            length_normalize=not args.no_length_normalize,
            device=device,
            optimizer=None,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        record = {
            "epoch": epoch,
            "updates": updates,
            "train": train_metrics,
            "validation": valid_metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

    elapsed_seconds = time.perf_counter() - started
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    summary = {
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
            "loss": "-logsigmoid(beta * ((pi_chosen-pi_rejected) - (ref_chosen-ref_rejected)))",
            "logprob_reduction": "sum" if args.no_length_normalize else "mean_over_response_tokens",
        },
        "trainable_parameters": sum(parameter.numel() for parameter in trainable_params),
        "train_seconds": elapsed_seconds,
        "history": history,
    }
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
