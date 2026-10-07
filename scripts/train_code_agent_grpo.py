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

from llm_posttrain.agent.code_environment import CodeAgentEnvironment
from llm_posttrain.config import load_yaml
from llm_posttrain.models.loader import model_input_device
from llm_posttrain.rl.grpo import (
    group_relative_advantages,
    sample_completion,
    trajectory_logprob,
)
try:
    from scripts.data_access_policy import assert_no_final_test_input
except ModuleNotFoundError:
    from data_access_policy import assert_no_final_test_input


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_base_model(
    model_path: str,
    *,
    use_4bit: bool,
    trainable: bool,
) -> Any:
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
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if use_4bit and trainable:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
    return model


def load_trainable_adapter(
    model_path: str,
    adapter_path: str,
    *,
    use_4bit: bool,
) -> Any:
    model = load_base_model(model_path, use_4bit=use_4bit, trainable=True)
    model = PeftModel.from_pretrained(
        model,
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


def collect_group(
    model: Any,
    tokenizer: Any,
    environment: CodeAgentEnvironment,
    task: dict[str, Any],
    *,
    args: argparse.Namespace,
    group_index: int,
) -> list[Any]:
    model.eval()
    model.config.use_cache = True

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

    return [
        environment.run(
            task,
            generate,
            task_id=(
                f"{task['id']}:group_{group_index}:rollout_{rollout_index}"
            ),
        )
        for rollout_index in range(args.num_rollouts)
    ]


def summarize_rewards(episodes: list[Any]) -> dict[str, Any]:
    keys = (
        "format_reward",
        "canonical_format_reward",
        "tool_selection_reward",
        "argument_reward",
        "execution_reward",
        "unit_test_reward",
        "final_answer_reward",
        "efficiency_reward",
        "total_reward",
    )
    if not episodes:
        return {
            "episodes": 0,
            "mean_reward": 0.0,
            "semantic_success_rate": 0.0,
            "strict_success_rate": 0.0,
            "mean_components": {key: 0.0 for key in keys},
        }
    return {
        "episodes": len(episodes),
        "mean_reward": sum(
            item.reward.total_reward for item in episodes
        ) / len(episodes),
        "semantic_success_rate": sum(
            int(item.semantic_success) for item in episodes
        ) / len(episodes),
        "strict_success_rate": sum(
            int(item.strict_success) for item in episodes
        ) / len(episodes),
        "mean_components": {
            key: sum(float(getattr(item.reward, key)) for item in episodes)
            / len(episodes)
            for key in keys
        },
    }


def train_epoch(
    model: Any,
    tokenizer: Any,
    environment: CodeAgentEnvironment,
    tasks: list[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    *,
    args: argparse.Namespace,
    epoch: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    optimizer.zero_grad(set_to_none=True)
    records: list[dict[str, Any]] = []
    episodes_all: list[Any] = []
    pending_groups = 0
    signal_groups = 0
    updates = 0

    for group_index, task in enumerate(tasks):
        episodes = collect_group(
            model,
            tokenizer,
            environment,
            task,
            args=args,
            group_index=group_index,
        )
        episodes_all.extend(episodes)
        advantages = group_relative_advantages(
            [episode.reward.total_reward for episode in episodes]
        )
        for rollout_index, (episode, advantage) in enumerate(
            zip(episodes, advantages)
        ):
            row = episode.to_dict()
            row.update(
                {
                    "epoch": epoch,
                    "group_index": group_index,
                    "rollout_index": rollout_index,
                    "advantage": advantage,
                }
            )
            records.append(row)

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
            group_loss = (
                contribution
                if group_loss is None
                else group_loss + contribution
            )
        if group_loss is None:
            continue
        (group_loss / args.gradient_accumulation_groups).backward()
        pending_groups += 1
        if pending_groups % args.gradient_accumulation_groups == 0:
            trainable = [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ]
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1

    if pending_groups and pending_groups % args.gradient_accumulation_groups:
        trainable = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        updates += 1

    summary = summarize_rewards(episodes_all)
    summary.update(
        {
            "epoch": epoch,
            "groups": len(tasks),
            "signal_groups": signal_groups,
            "updates": updates,
        }
    )
    return summary, records


def evaluate(
    model: Any,
    tokenizer: Any,
    environment: CodeAgentEnvironment,
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

    episodes = [
        environment.run(task, greedy, task_id=f"{task['id']}:validation")
        for task in tasks
    ]
    return summarize_rewards(episodes), [
        episode.to_dict() for episode in episodes
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run code-agent group-relative policy-gradient training."
    )
    parser.add_argument("--model-config", default="configs/base_model.yaml")
    parser.add_argument("--train-data", default="data/code_agent/train.jsonl")
    parser.add_argument("--valid-data", default="data/code_agent/valid.jsonl")
    parser.add_argument(
        "--init-adapter",
        default="models/adapters/code_agent_sft_v2_qwen3_1.7b",
    )
    parser.add_argument(
        "--output-dir",
        default="models/adapters/code_agent_grpo_qwen3_1.7b",
    )
    parser.add_argument(
        "--rollout-dir",
        default="outputs/code_agent_grpo",
    )
    parser.add_argument(
        "--summary-output",
        default="artifacts/code_agent_grpo_training_summary.json",
    )
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--gradient-accumulation-groups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--max-train-tasks", type=int, default=None)
    parser.add_argument("--max-valid-tasks", type=int, default=None)
    parser.add_argument("--progress-state", action="store_true")
    parser.add_argument("--rollout-only", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()
    assert_no_final_test_input(args.train_data, "Code Agent GRPO training")
    assert_no_final_test_input(args.valid_data, "Code Agent GRPO validation")

    if args.num_rollouts < 2:
        raise ValueError("num-rollouts must be at least 2")
    if args.gradient_accumulation_groups < 1:
        raise ValueError("gradient accumulation must be positive")

    set_seed(args.seed)
    model_config = load_yaml(args.model_config)
    model_path = str(model_config["model"]["local_path"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_tasks = read_jsonl(args.train_data)
    valid_tasks = read_jsonl(args.valid_data)
    if args.max_train_tasks is not None:
        train_tasks = train_tasks[:args.max_train_tasks]
    if args.max_valid_tasks is not None:
        valid_tasks = valid_tasks[:args.max_valid_tasks]
    if not train_tasks or not valid_tasks:
        raise ValueError("train and validation tasks must be non-empty")

    use_4bit = not args.no_4bit
    model = load_trainable_adapter(
        model_path,
        args.init_adapter,
        use_4bit=use_4bit,
    )
    environment = CodeAgentEnvironment(
        max_steps=args.max_steps,
        progress_state=args.progress_state,
    )
    output_dir = Path(args.output_dir)
    rollout_dir = Path(args.rollout_dir)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    if args.rollout_only:
        model.eval()
        model.config.use_cache = True
        episodes: list[Any] = []
        records: list[dict[str, Any]] = []
        for group_index, task in enumerate(train_tasks):
            group = collect_group(
                model,
                tokenizer,
                environment,
                task,
                args=args,
                group_index=group_index,
            )
            episodes.extend(group)
            advantages = group_relative_advantages(
                [episode.reward.total_reward for episode in group]
            )
            for rollout_index, (episode, advantage) in enumerate(
                zip(group, advantages)
            ):
                row = episode.to_dict()
                row.update(
                    {
                        "group_index": group_index,
                        "rollout_index": rollout_index,
                        "advantage": advantage,
                    }
                )
                records.append(row)
        rollout_path = rollout_dir / "rollouts_only.jsonl"
        write_jsonl(rollout_path, records)
        summary = {
            "mode": "rollout_only",
            "model_path": model_path,
            "init_adapter": args.init_adapter,
            "toolset": "code_agent_v2",
            "train_tasks": len(train_tasks),
            "num_rollouts": args.num_rollouts,
            "rollout_summary": summarize_rewards(episodes),
            "rollout_path": str(rollout_path),
        }
    else:
        trainable = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
        for epoch in range(1, args.num_epochs + 1):
            train_summary, train_rows = train_epoch(
                model,
                tokenizer,
                environment,
                train_tasks,
                optimizer,
                args=args,
                epoch=epoch,
            )
            valid_summary, valid_rows = evaluate(
                model,
                tokenizer,
                environment,
                valid_tasks,
                args=args,
            )
            train_path = rollout_dir / f"rollouts_epoch_{epoch}.jsonl"
            valid_path = rollout_dir / f"validation_epoch_{epoch}.jsonl"
            write_jsonl(train_path, train_rows)
            write_jsonl(valid_path, valid_rows)
            record = {
                "epoch": epoch,
                "train": train_summary,
                "validation": valid_summary,
                "train_rollout_path": str(train_path),
                "validation_path": str(valid_path),
            }
            history.append(record)
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
            "algorithm": "code_agent_group_relative_policy_gradient_without_critic",
            "toolset": "code_agent_v2",
            "progress_state": args.progress_state,
            "reward_weights": {
                "format": 0.10,
                "canonical_format": 0.05,
                "tool_selection": 0.15,
                "arguments": 0.10,
                "execution": 0.15,
                "unit_test": 0.20,
                "final_answer": 0.20,
                "efficiency": 0.05,
            },
            "max_seq_length": args.max_seq_length,
            "max_new_tokens": args.max_new_tokens,
            "max_steps": args.max_steps,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "learning_rate": args.learning_rate,
            "quantization": "4bit_nf4_double_quant" if use_4bit else "none",
            "trainable_parameters": sum(
                parameter.numel() for parameter in trainable
            ),
            "train_seconds": time.perf_counter() - started,
            "history": history,
        }

    output_summary = Path(args.summary_output)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
