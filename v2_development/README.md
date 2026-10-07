# llm-agentic-posttraining V2 Development Layer

这是 `llm-agentic-posttraining` 的独立开发层，面向 V2 路线中的大模型适配、Agent Runtime、Typed State、Checkpoint、Failure-aware Recovery 和可复现实验。

## 隔离边界

- 本目录是新增的开发层；父目录中的算法代码、数据、模型、权重和已有实验文件不作为本目录的写入目标。
- 当前阶段不修改父项目的包结构，也不依赖父项目的运行时环境。
- 后续与 V1 的连接通过 `ModelAdapter`、数据 manifest 和只读 evidence adapter 完成，避免把 Runtime Recovery 结果混写成模型原生能力。

## 当前可交付内容（无 GPU 可运行）

- `ModelAdapter` 契约和可运行的 `MockModelAdapter`。
- CPU 可运行的 `AgentState`、状态快照和 `StateTransition` / diff。
- JSON Checkpoint Store，支持原子保存和恢复。
- F1–F12 Failure Taxonomy 的最小分类器。
- 按失败类型选择策略的 Recovery Policy，包含重试上限和循环打断。
- Tool Registry、参数校验和安全的纯 Python calculator 工具。
- CPU smoke runner、单元测试和最小 metrics 汇总。
- GPU/HF/PEFT/API 适配器的明确延后接口；没有 GPU 时不会尝试加载权重。

## 快速运行

本目录只使用 Python 标准库：

```bash
python -m unittest discover -s . -p "test_*.py" -v
python scripts/smoke_cpu.py
```

如果不安装包，也可以直接运行；测试和 smoke script 会把 `src/` 加入路径。

## 目录

```text
v2_development/
├── configs/                 # CPU 当前配置、GPU 后续配置
├── docs/                    # 路线映射、边界和 Evidence Gate
├── scripts/                 # CPU smoke 与后续 benchmark 入口
├── src/llm_agent_v2/
│   ├── checkpoint.py        # JSON checkpoint
│   ├── failure.py           # Failure Taxonomy
│   ├── models.py            # ModelAdapter 与 Mock/Deferred adapters
│   ├── recovery.py          # Failure-aware Recovery Policy
│   ├── runtime.py           # Agent loop 与 trace
│   ├── state.py             # Typed State、Transition、diff
│   └── tools.py             # Tool schema、registry、CPU tools
└── tests/
```

## GPU 恢复后的接入顺序

1. 先完成 V1 evidence freeze，并把模型、adapter、数据和环境 manifest 固化。
2. 在 `models.py` 的统一契约下接入 Hugging Face / PEFT adapter；GPU 侧只增加实现，不改 Runtime。
3. 用同一套 task、tool、prompt、step budget 跑 B0–B5 baseline。
4. 再接入真实 checkpoint、long-horizon benchmark、fault injection 和 serving。

## 当前明确不做

本阶段不新增 RL 算法、多智能体、GraphRAG、Redis/Kubernetes 集群、第二个外部 benchmark 或大规模 UI。它们只有在内部 State/Checkpoint/Recovery 证据稳定后才进入评估。
