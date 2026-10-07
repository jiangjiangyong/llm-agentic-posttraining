from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from llm_posttrain.agent.environment import CalculatorEnvironment
from llm_posttrain.config import load_yaml
from llm_posttrain.rl.grpo import group_relative_advantages, sample_completion, trajectory_logprob
from llm_posttrain.tools.registry import build_default_registry
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


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_base_model(model_path: str, *, use_4bit: bool, trainable: bool):
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }
    if use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.float16
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if use_4bit and trainable:
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=True,
        )
    return base_model


def load_trainable_adapter(model_path: str, adapter_path: str, *, use_4bit: bool):
    base_model = load_base_model(model_path, use_4bit=use_4bit, trainable=True)
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=True,
        local_files_only=True,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


def generation_fn(model: Any, tokenizer: Any, args: argparse.Namespace):
    def generate(messages: list[dict[str, Any]]) -> str:
        return sample_completion(
            model,
            tokenizer,
            messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
        )

    return generate


def collect_group(
    model: Any,
    tokenizer: Any,
    environment: CalculatorEnvironment,
    task: dict[str, Any],
    *,
    num_rollouts: int,
    args: argparse.Namespace,
    group_index: int,
) -> list[Any]:
    model.eval()
    model.config.use_cache = True
    generate = generation_fn(model, tokenizer, args)
    return [
        environment.run(
            task,
            generate,
            task_id=f"{task['id']}:group_{group_index}:rollout_{rollout_index}",
        )
        for rollout_index in range(num_rollouts)
    ]


def summarize_rewards(episodes: list[Any]) -> dict[str, Any]:
    if not episodes:
        return {"episodes": 0, "mean_reward": 0.0, "success_rate": 0.0}
    reward_keys = (
        "format_reward",
        "canonical_format_reward",
        "tool_selection_reward",
        "argument_reward",
        "execution_reward",
        "final_answer_reward",
        "total_reward",
    )
    return {
        "episodes": len(episodes),
        "mean_reward": sum(item.reward.total_reward for item in episodes) / len(episodes),
        "success_rate": sum(int(item.success) for item in episodes) / len(episodes),
        "mean_components": {
            key: sum(float(getattr(item.reward, key)) for item in episodes) / len(episodes)
            for key in reward_keys
        },
    }


def write_episode_records(path: str | Path, records: list[dict[str, Any]]) -> None:
    write_jsonl(path, records)


def train_epoch(
    model: Any,
    tokenizer: Any,
    environment: CalculatorEnvironment,
    tasks: list[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    *,
    args: argparse.Namespace,
    epoch: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    optimizer.zero_grad(set_to_none=True)
    episode_records: list[dict[str, Any]] = []
    all_episodes: list[Any] = []
    signal_groups = 0
    updates = 0
    pending_groups = 0
    for group_index, task in enumerate(tasks):
        episodes = collect_group(
            model,
            tokenizer,
            environment,
            task,
            num_rollouts=args.num_rollouts,
            args=args,
            group_index=group_index,
        )
        rewards = [episode.reward.total_reward for episode in episodes]
        advantages = group_relative_advantages(rewards)
        all_episodes.extend(episodes)
        for rollout_index, (episode, advantage) in enumerate(zip(episodes, advantages)):
            row = episode.to_dict()
            row["epoch"] = epoch
            row["group_index"] = group_index
            row["rollout_index"] = rollout_index
            row["advantage"] = advantage
            episode_records.append(row)
        if not any(abs(value) > 1e-8 for value in advantages):
            continue
        signal_groups += 1
        model.train()
        model.config.use_cache = False
        group_loss: torch.Tensor | None = None
        for episode, advantage in zip(episodes, advantages):
            if abs(advantage) <= 1e-8:
                continue
            logprob, _ = trajectory_logprob(
                model,
                tokenizer,
                episode,
                max_seq_length=args.max_seq_length,
                normalize_by_tokens=True,
            )
            contribution = -float(advantage) * logprob
            group_loss = contribution if group_loss is None else group_loss + contribution
        if group_loss is None:
            continue
        (group_loss / args.gradient_accumulation_groups).backward()
        pending_groups += 1
        if pending_groups % args.gradient_accumulation_groups == 0:
            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
    if pending_groups % args.gradient_accumulation_groups != 0:
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        updates += 1
    summary = summarize_rewards(all_episodes)
    summary.update(
        {
            "epoch": epoch,
            "groups": len(tasks),
            "signal_groups": signal_groups,
            "updates": updates,
        }
    )
    return summary, episode_records


def evaluate(
    model: Any,
    tokenizer: Any,
    environment: CalculatorEnvironment,
    tasks: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    model.config.use_cache = True

    def greedy(messages: list[dict[str, Any]]) -> str:
        return sample_completion(
            model,
            tokenizer,
            messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=False,
        )

    episodes = [environment.run(task, greedy, task_id=f"{task['id']}:validation") for task in tasks]
    rows = [episode.to_dict() for episode in episodes]
    summary = summarize_rewards(episodes)
    summary["mode"] = "greedy_validation"
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal Agentic GRPO loop on CalculatorEnvironment.")
    parser.add_argument("--model-config", default="configs/base_model.yaml")
    parser.add_argument("--train-data", default="data/agentic_rl/train.jsonl")
    parser.add_argument("--valid-data", default="data/agentic_rl/valid.jsonl")
    parser.add_argument("--init-adapter", default="models/adapters/sft_qwen3_1.7b")
    parser.add_argument("--output-dir", default="models/adapters/grpo_qwen3_1.7b")
    parser.add_argument("--rollout-dir", default="outputs/grpo")
    parser.add_argument("--summary-output", default="artifacts/grpo_training_summary.json")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--gradient-accumulation-groups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--max-train-tasks", type=int, default=None)
    parser.add_argument("--max-valid-tasks", type=int, default=None)
    parser.add_argument("--rollout-only", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()
    assert_no_final_test_input(args.train_data, "GRPO training")
    assert_no_final_test_input(args.valid_data, "GRPO validation")
    if args.num_rollouts < 2:
        raise ValueError("GRPO requires at least two rollouts per prompt")
    if args.gradient_accumulation_groups < 1:
        raise ValueError("gradient accumulation groups must be positive")

    set_seed(args.seed)
    model_config = load_yaml(args.model_config)
    model_path = str(model_config["model"]["local_path"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_tasks = read_jsonl(args.train_data)
    valid_tasks = read_jsonl(args.valid_data)
    if args.max_train_tasks is not None:
        train_tasks = train_tasks[: args.max_train_tasks]
    if args.max_valid_tasks is not None:
        valid_tasks = valid_tasks[: args.max_valid_tasks]
    if not train_tasks or not valid_tasks:
        raise ValueError("train and validation tasks must be non-empty")

    use_4bit = not args.no_4bit
    print("Loading SFT Adapter as GRPO policy")
    model = load_trainable_adapter(model_path, args.init_adapter, use_4bit=use_4bit)
    environment = CalculatorEnvironment(build_default_registry())
    output_dir = Path(args.output_dir)
    rollout_dir = Path(args.rollout_dir)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    if args.rollout_only:
        model.eval()
        model.config.use_cache = True
        records: list[dict[str, Any]] = []
        rollout_episodes: list[Any] = []
        for group_index, task in enumerate(train_tasks):
            episodes = collect_group(
                model,
                tokenizer,
                environment,
                task,
                num_rollouts=args.num_rollouts,
                args=args,
                group_index=group_index,
            )
            rollout_episodes.extend(episodes)
            advantages = group_relative_advantages([item.reward.total_reward for item in episodes])
            for rollout_index, (episode, advantage) in enumerate(zip(episodes, advantages)):
                row = episode.to_dict()
                row.update({"group_index": group_index, "rollout_index": rollout_index, "advantage": advantage})
                records.append(row)
        rollout_path = rollout_dir / "rollouts_only.jsonl"
        write_episode_records(rollout_path, records)
        summary = {
            "mode": "rollout_only",
            "model_path": model_path,
            "init_adapter": args.init_adapter,
            "train_tasks": len(train_tasks),
            "num_rollouts": args.num_rollouts,
            "rollout_summary": summarize_rewards(rollout_episodes),
            "rollout_path": str(rollout_path),
            "reward_values": [row["reward"]["total_reward"] for row in records],
        }
    else:
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
        epoch_records: list[dict[str, Any]] = []
        for epoch in range(1, args.num_epochs + 1):
            train_summary, records = train_epoch(
                model,
                tokenizer,
                environment,
                train_tasks,
                optimizer,
                args=args,
                epoch=epoch,
            )
            validation_summary, validation_rows = evaluate(
                model,
                tokenizer,
                environment,
                valid_tasks,
                args=args,
            )
            train_path = rollout_dir / f"rollouts_epoch_{epoch}.jsonl"
            valid_path = rollout_dir / f"validation_epoch_{epoch}.jsonl"
            write_episode_records(train_path, records)
            write_episode_records(valid_path, validation_rows)
            record = {
                "epoch": epoch,
                "train": train_summary,
                "validation": validation_summary,
                "rollout_path": str(train_path),
                "validation_path": str(valid_path),
            }
            history.append(record)
            epoch_records.extend(records)
            print(json.dumps(record, ensure_ascii=False))
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        summary = {
            "mode": "train",
            "model_path": model_path,
            "init_adapter": args.init_adapter,
            "output_dir": args.output_dir,
            "train_data": args.train_data,
            "valid_data": args.valid_data,
            "train_tasks": len(train_tasks),
            "valid_tasks": len(valid_tasks),
            "num_rollouts": args.num_rollouts,
            "algorithm": "group_relative_policy_gradient_without_critic",
            "reward_weights": {
                "format": 0.10,
                "canonical_format": 0.10,
                "tool_selection": 0.15,
                "arguments": 0.25,
                "execution": 0.20,
                "final_answer": 0.20,
            },
            "max_seq_length": args.max_seq_length,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "learning_rate": args.learning_rate,
            "quantization": "4bit_nf4_double_quant" if use_4bit else "none",
            "trainable_parameters": sum(parameter.numel() for parameter in trainable),
            "train_seconds": time.perf_counter() - started,
            "history": history,
        }
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
