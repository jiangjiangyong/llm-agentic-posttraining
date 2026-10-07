from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import date
from pathlib import Path
from typing import Any


BACKENDS = ("base", "sft", "dpo", "grpo")


def read_json(root: Path, relative_path: str) -> dict[str, Any]:
    return json.loads((root / relative_path).read_text(encoding="utf-8"))


def percent(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.2f}%"


def decimal(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def number(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def git_head(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def build_report(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    benchmark = read_json(root, "artifacts/agent_benchmark_summary.json")
    sft = read_json(root, "artifacts/sft_training_summary.json")
    dpo = read_json(root, "artifacts/dpo_training_summary_v2.json")
    grpo = read_json(root, "artifacts/grpo_training_summary.json")
    flywheel = read_json(root, "artifacts/failure_flywheel_manifest.json")
    flywheel_sft = read_json(root, "artifacts/failure_flywheel_sft_training_summary.json")
    flywheel_dpo = read_json(root, "artifacts/failure_flywheel_dpo_training_summary.json")
    serving = read_json(root, "artifacts/serving_compatibility.json")
    sft_manifest = read_json(root, "artifacts/sft_manifest.json")
    preference_manifest = read_json(root, "artifacts/preference_manifest.json")
    rl_manifest = read_json(root, "artifacts/agentic_rl_manifest.json")

    benchmark_rows: dict[str, dict[str, Any]] = {}
    for backend in BACKENDS:
        data = benchmark["backends"][backend]
        benchmark_rows[backend] = {
            "general": data["general"],
            "tool_calling": data["tool_calling"],
            "agent_environment": data["agent_environment"],
        }

    curves: list[dict[str, Any]] = []
    for item in sft.get("log_history", []):
        for metric in ("loss", "eval_loss"):
            if metric in item:
                curves.append(
                    {
                        "stage": "sft",
                        "split": "train" if metric == "loss" else "validation",
                        "step": item.get("step"),
                        "epoch": item.get("epoch"),
                        "metric": metric,
                        "value": item[metric],
                    }
                )
    for item in dpo.get("history", []):
        epoch = item.get("epoch")
        for split_name in ("train", "validation"):
            split = item.get(split_name, {})
            for metric in ("loss", "preference_accuracy", "margin"):
                if metric in split:
                    curves.append(
                        {
                            "stage": "dpo_v2",
                            "split": split_name,
                            "step": item.get("updates"),
                            "epoch": epoch,
                            "metric": metric,
                            "value": split[metric],
                        }
                    )
    for item in grpo.get("history", []):
        for split_name in ("train", "validation"):
            split = item.get(split_name, {})
            for metric in ("mean_reward", "success_rate", "total_reward"):
                if metric in split:
                    curves.append(
                        {
                            "stage": "grpo",
                            "split": split_name,
                            "step": split.get("updates"),
                            "epoch": item.get("epoch"),
                            "metric": metric,
                            "value": split[metric],
                        }
                    )

    report = {
        "generated_on": date.today().isoformat(),
        "git_head": git_head(root),
        "benchmark": benchmark_rows,
        "training": {
            "sft": {
                "train_records": sft["train_records"],
                "valid_records": sft["valid_records"],
                "quantization": sft["quantization"],
                "trainable_parameters": sft["trainable_parameters"],
                "total_parameters": sft["total_parameters"],
                "trainable_ratio": sft["trainable_ratio"],
                "train_seconds": sft["train_seconds"],
                "train_loss": sft["train_metrics"]["train_loss"],
                "eval_loss": sft["eval_metrics"]["eval_loss"],
                "lora": sft["lora"],
            },
            "dpo_v2": {
                "train_records": dpo["train_records"],
                "valid_records": dpo["valid_records"],
                "train_seconds": dpo["train_seconds"],
                "config": dpo["dpo"],
                "validation": dpo["history"][-1]["validation"],
            },
            "grpo": {
                "train_tasks": grpo["train_tasks"],
                "valid_tasks": grpo["valid_tasks"],
                "rollouts_per_task": grpo["num_rollouts"],
                "algorithm": grpo["algorithm"],
                "learning_rate": grpo["learning_rate"],
                "max_new_tokens": grpo["max_new_tokens"],
                "temperature": grpo["temperature"],
                "top_p": grpo["top_p"],
                "train_seconds": grpo["train_seconds"],
                "train": grpo["history"][-1]["train"],
                "validation": grpo["history"][-1]["validation"],
            },
        },
        "data_integrity": {
            "sft_manifest": {
                "total": sft_manifest["num_examples"],
                "train": sft_manifest["train_count"],
                "validation": sft_manifest["validation_count"],
                "evaluation_prompt_overlap": sft_manifest["evaluation_prompt_overlap_count"],
            },
            "preference_manifest": {
                "source": preference_manifest["source_count"],
                "pairs": preference_manifest["pair_count"],
                "train": preference_manifest["train_count"],
                "validation": preference_manifest["validation_count"],
                "evaluation_prompt_overlap": preference_manifest["evaluation_prompt_overlap_count"],
            },
            "agentic_rl_manifest": {
                "source": rl_manifest["source_count"],
                "train": rl_manifest["train_count"],
                "validation": rl_manifest["validation_count"],
                "evaluation_prompt_overlap": rl_manifest["evaluation_prompt_overlap_count"],
            },
        },
        "failure_flywheel": {
            "source_rollouts": flywheel["source_rollout_count"],
            "failed_rollouts": flywheel["failed_rollout_count"],
            "failed_task_groups": flywheel["failed_task_count"],
            "sft_records": flywheel["sft_count"],
            "sft_train": flywheel["sft_train_count"],
            "sft_validation": flywheel["sft_validation_count"],
            "dpo_pairs": flywheel["dpo_count"],
            "dpo_train": flywheel["dpo_train_count"],
            "dpo_validation": flywheel["dpo_validation_count"],
            "failure_type_counts": flywheel["failure_type_counts"],
            "primary_failure_counts": flywheel["primary_failure_counts"],
            "flywheel_sft_train_loss": flywheel_sft["train_metrics"]["train_loss"],
            "flywheel_sft_eval_loss": flywheel_sft["eval_metrics"]["eval_loss"],
            "flywheel_dpo_validation_preference_accuracy": flywheel_dpo["history"][-1]["validation"]["preference_accuracy"],
        },
        "serving": serving,
    }
    return report, curves


def markdown_report(report: dict[str, Any]) -> str:
    benchmark = report["benchmark"]
    training = report["training"]
    integrity = report["data_integrity"]
    flywheel = report["failure_flywheel"]
    serving = report["serving"]

    lines = [
        "# 阶段 14：实验总表与求职材料",
        "",
        f"> 生成日期：{report['generated_on']}；Git HEAD：`{report['git_head']}`。本文件由 `scripts/build_experiment_report.py` 根据实验 JSON 自动生成。",
        "",
        "## 项目结论",
        "",
        "本项目完成了 `Data -> Training -> Reward -> Evaluation -> Experiment` 闭环：以 Qwen3-1.7B-Base 为基座，完成 QLoRA-SFT、DPO、最小 group-relative policy gradient、Agent 环境奖励、统一 benchmark、失败数据飞轮和 FastAPI/Web Demo。",
        "",
        "当前求职展示的稳定 checkpoint 是原始 SFT Adapter。DPO-v2、GRPO 和飞轮增量实验保留为可解释的算法对照、限制和负结果，不把小规模单 seed 结果描述成生产级或统计显著结论。",
        "",
        "## 训练总表",
        "",
        "| 阶段 | 数据/规模 | 关键配置 | 训练结果 | 产物 |",
        "| --- | --- | --- | --- | --- |",
        f"| SFT | {training['sft']['train_records']} train / {training['sft']['valid_records']} valid | 4-bit NF4；LoRA r={training['sft']['lora']['r']}，alpha={training['sft']['lora']['alpha']}；目标模块 q/k/v/o + gate/up/down | train loss {decimal(training['sft']['train_loss'])}；eval loss {decimal(training['sft']['eval_loss'])}；可训练参数 {training['sft']['trainable_parameters']:,} / {training['sft']['total_parameters']:,} ({percent(training['sft']['trainable_ratio'])}) | `models/adapters/sft_qwen3_1.7b` |",
        f"| DPO-v2 | {training['dpo_v2']['train_records']} train / {training['dpo_v2']['valid_records']} valid | beta={training['dpo_v2']['config']['beta']}；lr={training['dpo_v2']['config']['learning_rate']}；response-token mean log-prob | valid preference accuracy {percent(training['dpo_v2']['validation']['preference_accuracy'])}；margin {decimal(training['dpo_v2']['validation']['margin'])} | `models/adapters/dpo_qwen3_1.7b_v2` |",
        f"| GRPO | {training['grpo']['train_tasks']} train tasks；{training['grpo']['valid_tasks']} valid tasks；{training['grpo']['rollouts_per_task']} rollouts/task | {training['grpo']['algorithm']}；lr={training['grpo']['learning_rate']}；max new tokens {training['grpo']['max_new_tokens']} | train reward {decimal(training['grpo']['train']['mean_reward'])}；valid reward {decimal(training['grpo']['validation']['mean_reward'])}；updates {training['grpo']['train']['updates']} | `models/adapters/grpo_qwen3_1.7b` |",
        "",
        "## 统一 Benchmark",
        "",
        "评测集固定为 12 条通用题、6 条工具调用题、6 条 Agent 环境验证题。Agent 语义成功与严格 canonical wrapper 成功分开统计。",
        "",
        "| Backend | 通用集 | 工具调用成功 | Agent mean reward | Agent semantic success | 严格 Agent success |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for backend, label in (("base", "Base"), ("sft", "SFT"), ("dpo", "DPO-v2"), ("grpo", "GRPO")):
        row = benchmark[backend]
        lines.append(
            f"| {label} | {row['general']['passed']}/{row['general']['total']} ({percent(row['general']['accuracy'])}) | {percent(row['tool_calling']['task_success_rate'])} | {decimal(row['agent_environment']['mean_reward'])} | "
            f"{percent(row['agent_environment'].get('semantic_success_rate', 0.0))} | {percent(row['agent_environment']['strict_success_rate'])} |"
        )
    lines.extend(
        [
            "",
            "### Reward 组件",
            "",
            "| Backend | Format | Canonical format | Tool selection | Arguments | Execution | Final answer |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    component_names = (
        "format_reward",
        "canonical_format_reward",
        "tool_selection_reward",
        "argument_reward",
        "execution_reward",
        "final_answer_reward",
    )
    for backend, label in (("base", "Base"), ("sft", "SFT"), ("dpo", "DPO-v2"), ("grpo", "GRPO")):
        components = benchmark[backend]["agent_environment"]["mean_components"]
        values = " | ".join(decimal(components[name]) for name in component_names)
        lines.append(f"| {label} | {values} |")

    lines.extend(
        [
            "",
            "## 数据完整性",
            "",
            f"- SFT：总样本 {integrity['sft_manifest']['total']}，划分 {integrity['sft_manifest']['train']}/{integrity['sft_manifest']['validation']}；与评测 prompt 重叠 {integrity['sft_manifest']['evaluation_prompt_overlap']}。",
            f"- Preference：source {integrity['preference_manifest']['source']}，pair {integrity['preference_manifest']['pairs']}，划分 {integrity['preference_manifest']['train']}/{integrity['preference_manifest']['validation']}；与评测 prompt 重叠 {integrity['preference_manifest']['evaluation_prompt_overlap']}。",
            f"- Agentic RL：source {integrity['agentic_rl_manifest']['source']}，划分 {integrity['agentic_rl_manifest']['train']}/{integrity['agentic_rl_manifest']['validation']}；与评测 prompt 重叠 {integrity['agentic_rl_manifest']['evaluation_prompt_overlap']}。",
            "- 所有模型比较使用同一冻结评测文件；训练集、验证集和评测集按 manifest 保存 SHA-256 或 overlap 结果。",
            "",
            "## Failure Data Flywheel",
            "",
            f"GRPO rollout 共 {flywheel['source_rollouts']} 条，其中严格环境失败 {flywheel['failed_rollouts']} 条，按 task group 去重后得到 {flywheel['failed_task_groups']} 个失败任务组；生成 {flywheel['sft_records']} 条 SFT 修复样本和 {flywheel['dpo_pairs']} 条 DPO pair，分别按 {flywheel['sft_train']}/{flywheel['sft_validation']}、{flywheel['dpo_train']}/{flywheel['dpo_validation']} 划分。",
            "",
            "主要失败计数：",
            "",
            "| Failure type | Count |",
            "| --- | ---: |",
        ]
    )
    for key, value in flywheel["failure_type_counts"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "parser 修复后的飞轮重测仍显示增量训练存在回归：原始 SFT 的工具任务成功率为 100.00%、Agent mean reward 为 0.7917，飞轮 SFT 分别为 66.67% 和 0.7917；原始 DPO-v2 为 16.67% 和 0.5417，飞轮 DPO 为 0.00% 和 0.4917。因此稳定 serving checkpoint 仍选原始 SFT Adapter。",
            "",
            "## Serving",
            "",
            f"- GPU：{serving.get('gpu_name', '-')}；Compute Capability：{serving.get('compute_capability', '-')}；CUDA：{serving.get('cuda_version', '-')}；Torch：{serving.get('torch', '-')}。",
            f"- vLLM installed：{serving.get('vllm_installed', False)}；实际 backend：`{serving.get('recommended_backend', '-')}`。",
            "- API：`GET /health`、`POST /v1/chat/completions`、`POST /v1/agent/run`；Web Demo：`web/index.html`。",
            "- 真实 smoke：普通聊天成功；Agent calculator 请求经过 parser 容错后成功返回 `4500`，trace 完整记录 model output、tool call、tool result、final answer。",
            "",
            "## 简历项目描述",
            "",
            "**面向代码与工具调用能力的大模型后训练及 Agentic RL 优化平台**",
            "",
            "- 基于 Qwen3-1.7B-Base 搭建可复现实验闭环，完成 4-bit NF4 QLoRA-SFT、response-token mean log-prob DPO，以及面向工具调用环境的最小 group-relative policy-gradient 训练；SFT 仅更新约 1.003% 参数。",
            "- 构建 code/JSON/math/tool-calling 多类型 SFT 与 preference 数据管线，加入 schema 校验、去重、task-group 划分、SHA-256 manifest 和 evaluation prompt overlap 检查，避免训练评测泄漏。",
            "- 实现 Tool Schema、ToolCallParser、AgentRuntime、CalculatorEnvironment 与分项 reward，将格式、工具选择、参数、执行、最终答案拆成可诊断指标；统一评测覆盖 12 条通用题、6 条工具题、6 条 Agent 验证题。",
            "- 实现 GRPO rollout、group-relative advantage、LoRA 更新和 failure data flywheel；通过失败 trajectory 自动生成 oracle SFT 修复样本与 DPO pairs，并保留增量训练负结果用于回归分析。",
            "- 统一 benchmark 中原始 SFT 达到通用集 91.67%、parser 修复后的工具任务成功率 100.00%、Agent mean reward 0.7917；实现 FastAPI + Transformers Adapter serving 与浏览器 Demo，提供完整 Agent trace。",
            "",
            "## 面试时必须主动说明",
            "",
            "1. 当前 GRPO 是明确标注的 `group_relative_policy_gradient_without_critic` 最小实现，不等同于带 PPO clipping、critic、分布式 rollout 的生产级 GRPO。",
            "2. DPO-v2 虽然通用集达到 100%，但工具调用退化为 0%，说明小规模 preference 数据存在格式和长度偏置；因此没有把 DPO 结果包装成最终模型。",
            "3. Agent strict success 仍为 0%，因为自定义环境要求 canonical `<tool_call>...</tool_call>`，而 frozen tool evaluator 允许更宽松的可解析格式；协议合规和端到端成功必须分列。",
            "4. 当前 RTX 2080 Ti 为 Compute Capability 7.5，环境未安装 vLLM，实际采用 Transformers fallback；简历不写“已完成 vLLM 生产部署”。",
            "",
            "## 可复现命令",
            "",
            "```bash",
            "source .venv/bin/activate",
            "PYTHONPATH=src pytest -q",
            "PYTHONPATH=src python scripts/run_agent_benchmark.py --backends base,sft,dpo,grpo --max-new-tokens 256 --output-dir outputs/agent_benchmark --manifest-output artifacts/agent_benchmark_manifest.json --summary-output artifacts/agent_benchmark_summary.json",
            "PYTHONPATH=src python scripts/build_experiment_report.py",
            "PYTHONPATH=src python scripts/serve_api.py --backend adapter --adapter-path models/adapters/sft_qwen3_1.7b --host 0.0.0.0 --port 8000",
            "```",
            "",
            "## 剩余限制与下一轮工作",
            "",
            "- 当前数据集和环境以 calculator 为主，规模适合求职项目验证链路，不足以证明广泛 Agent 泛化。",
            "- 下一轮优先扩展 canonical format 数据、多个工具和代码执行 sandbox，并增加多 seed、更多 validation task 和吞吐/显存 benchmark。",
            "- parser 容错必须单独记录 repair rate，不能替代模型本身的协议学习；后续应通过数据和训练减少 malformed output。",
            "",
        ]
    )
    return "\n".join(lines)


def resume_and_interview() -> str:
    return """# 求职材料：大模型算法工程师项目

## 项目名称

面向代码与工具调用能力的大模型后训练及 Agentic RL 优化平台

## 简历版描述

基于 Qwen3-1.7B-Base 搭建 `Data -> Training -> Reward -> Evaluation -> Experiment` 闭环，完成 4-bit NF4 QLoRA-SFT、response-token mean log-prob DPO 和面向工具调用环境的最小 group-relative policy-gradient 训练；构建 code/JSON/math/tool-calling 数据管线、Tool Schema、AgentRuntime、分项 reward、统一 benchmark 和 failure data flywheel。统一评测覆盖 12 条通用题、6 条工具调用题和 6 条 Agent 验证题，原始 SFT Adapter 在通用集达到 91.67%、parser 修复后的工具任务成功率 100.00%、Agent mean reward 0.7917；另提供 FastAPI + Transformers Adapter API、Web Demo 和完整 Agent trace。

## 推荐拆成简历 bullet

1. 基于 Qwen3-1.7B-Base 完成 4-bit NF4 QLoRA-SFT、response-token mean log-prob DPO 和最小 group-relative policy-gradient 训练，SFT 仅更新 17,432,576 / 1,738,007,552 参数。
2. 构建 code/JSON/math/tool-calling 数据管线，完成 schema 校验、去重、task-group 划分、SHA-256 manifest 和 evaluation prompt overlap 检查，保证训练与评测隔离。
3. 实现 Tool Schema、ToolCallParser、AgentRuntime、CalculatorEnvironment 和分项 reward，将格式、工具选择、参数、执行、最终答案拆分为可诊断指标。
4. 实现 rollout、group-relative advantage、LoRA 更新和 failure data flywheel，从失败 trajectory 生成 oracle SFT 修复样本与 DPO pairs，并通过统一 benchmark 验证增量训练的正负结果。
5. 使用 FastAPI + Transformers Adapter 实现 `/v1/chat/completions`、`/v1/agent/run` 和浏览器 Demo，返回 token、延迟、工具执行结果和 Agent trace。

## 90 秒项目介绍

这个项目的核心不是做一个普通 Agent Demo，而是研究模型如何通过后训练学会稳定的工具调用协议。第一步固定 Base Model 和独立评测集，建立通用能力、工具调用和 Agent 环境三类基线。第二步用带有 canonical tool call、tool observation 和 final answer 的消息构建 SFT 数据，先让模型学会任务格式。然后从 SFT checkpoint 构造 preference pair 做 DPO，再把 Agent 放进确定性的 calculator environment 中采样 trajectory，用格式、工具选择、参数、执行和最终答案组成 reward，做 group-relative policy update。最后把失败 rollout 按 task group 去重，重新生成修复 SFT/DPO 数据，并用同一套 benchmark 验证有没有回归。实验结果显示，当前小规模数据上原始 SFT 是最稳定的 serving checkpoint；DPO 提升了通用集但明显退化工具任务，GRPO 在 parser 修复后的统一 benchmark 中与 SFT 持平但严格 canonical success 仍为零，这些对照结果帮助定位下一轮数据问题。

## 高频面试问题与回答要点

### 1. Base Model 和 Agent Runtime 的关系是什么？

Base Model 只负责根据 messages 生成下一段文本；AgentRuntime 负责注入工具 schema、解析 tool call、调用 allow-listed tool、把 observation 追加回上下文并再次请求模型。模型是策略，Runtime 是环境交互循环，两者通过消息协议连接。

### 2. Tool Calling 是模型能力还是 Agent 功能？

两部分都有。Runtime 可以规定 schema、parser 和 executor，但模型必须学会选择工具、生成正确参数和在 observation 后结束。这个项目用 SFT/DPO/RL 训练模型行为，用 Runtime 提供可执行环境。

### 3. SFT 训练了什么？

在本项目里，SFT 不是只训练答案，而是训练完整 messages 到 assistant response 的行为：什么时候输出工具调用、工具名和参数如何对齐、收到 observation 后如何输出 final answer。训练标签只对 assistant response 计算 loss，避免模型被 user/system token 主导。

### 4. 为什么 DPO 之后通用准确率升到 100%，工具成功率却变成 0？

Preference pair 数量小，且 rejected response 的长度和格式分布可能造成偏置。DPO 优化的是 chosen/rejected 的相对 log-prob，不直接优化环境执行成功；模型可能更偏好普通文本或非 canonical JSON。统一 benchmark 把这当作真实退化结果，而不是只报告通用集提升。

### 5. 这个项目里的 GRPO 和标准 GRPO 有什么边界？

这里实现的是 `group_relative_policy_gradient_without_critic`：同一 prompt 采样 4 条 rollout，执行环境得到 reward，在组内中心化得到 advantage，再做 LoRA 更新。当前没有 critic、PPO ratio clipping、分布式 rollout 和 production trainer，因此面试中称为最小可验证 GRPO-style 实验。

### 6. Reward 如何计算？

Reward 分成 format、canonical format、tool selection、arguments、execution、final answer 六项，每项有明确判定函数和权重。这样可以知道失败发生在输出格式、工具选择、参数、执行还是最终答案，而不是只看一个黑盒总分。

### 7. 为什么需要 Agent Environment？

单轮文本评测无法判断工具是否真的被调用、参数是否能执行、观察结果是否被利用。Environment 接收模型动作，执行安全的 calculator tool，返回 observation 和 reward，形成可用于 rollout 和 policy update 的 trajectory。

### 8. 为什么 Evaluation 必须独立于 Training？

如果训练样本和评测 prompt 重叠，准确率可能只是记忆。项目为 SFT、preference、Agentic RL 分别保存 manifest，检查 evaluation prompt overlap，并让 Base/SFT/DPO/GRPO 使用同一冻结评测集。

### 9. Failure Data Flywheel 做了什么？

它读取 GRPO 训练 rollout，按 base task id 与 oracle 任务合并，只保留失败训练任务，按 task group 去重，再生成 canonical SFT trajectory 和 chosen/rejected DPO pair。失败输出不会直接执行 Python，工具执行仍由 allow-listed environment 完成。

### 10. 为什么 serving 采用 Transformers 而不是 vLLM？

当前服务器是 RTX 2080 Ti，Compute Capability 7.5，项目环境未安装 vLLM；我用兼容性脚本记录了 GPU、CUDA、Torch 和 vLLM 状态，实际部署 Transformers + 4-bit PEFT Adapter。这样结果可复现，也避免虚构吞吐或 vLLM 生产能力。

### 11. parser 容错是不是掩盖模型问题？

不能替代训练。容错只针对已观察到的 calculator schema malformed JSON，并且原始 model output 仍写入 trace；后续还应统计 repair rate，并通过 canonical format 数据和训练降低 repair 需求。评测中应同时报告原始协议成功率和容错后端到端成功率。

### 12. 你最终选择哪个 checkpoint？

原始 SFT Adapter。它在当前统一 benchmark 上同时保持通用能力、工具任务成功率和最高 Agent mean reward。DPO-v2、GRPO、飞轮增量保留为算法实验、对照和负结果，不能因为某一个子集更高就替换稳定 baseline。

## 阶段 14 协议对齐复盘

阶段 14 新增了 canonical tool-call 数据，并记录了从原始 SFT 增量训练、从 Base 重训和 Runtime system prompt 对齐的对照实验。严格协议成功率仍为 0%，因此没有把 canonical Adapter 替换为最终 checkpoint。

评测现在分开报告：
- Agent semantic success：工具选择、参数、真实执行和最终答案全部正确。
- Agent strict protocol success：在 semantic success 之外，还要求原始输出满足 canonical `<tool_call>` 且 payload 不含 `type`。

当前默认 benchmark 中，原始 SFT 与 GRPO 的 semantic success 为 66.67%，strict protocol success 为 0%；DPO-v2 的通用集为 100%，但 Agent semantic success 为 0%。这组负结果说明 DPO 的通用能力提升没有转化为工具环境能力，原始 SFT 仍是稳定 serving checkpoint。


## 证据文件

- `artifacts/agent_benchmark_summary.json`

- `artifacts/sft_training_summary.json`
- `artifacts/dpo_training_summary_v2.json`
- `artifacts/grpo_training_summary.json`
- `artifacts/failure_flywheel_manifest.json`
- `artifacts/serving_compatibility.json`
- `artifacts/serving_agent_smoke_recovered_v2.json`
- `artifacts/training_curves.csv`
"""


def write_curves(path: Path, curves: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("stage", "split", "step", "epoch", "metric", "value")
    unique_curves = []
    seen: set[tuple[Any, ...]] = set()
    for row in curves:
        key = tuple(row.get(field) for field in fields)
        if key in seen:
            continue
        seen.add(key)
        unique_curves.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(unique_curves)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reproducible stage 14 experiment reports.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-json", default="artifacts/experiment_summary.json")
    parser.add_argument("--output-md", default="artifacts/experiment_summary.md")
    parser.add_argument("--output-resume", default="docs/resume_and_interview.md")
    parser.add_argument("--curves-csv", default="artifacts/training_curves.csv")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    report, curves = build_report(root)
    for relative_path, content in (
        (args.output_json, json.dumps(report, ensure_ascii=False, indent=2) + "\n"),
        (args.output_md, markdown_report(report)),
        (args.output_resume, resume_and_interview()),
    ):
        output = root / relative_path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
    write_curves(root / args.curves_csv, curves)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_resume}")
    print(f"Wrote {args.curves_csv} ({len(curves)} rows)")


if __name__ == "__main__":
    main()
