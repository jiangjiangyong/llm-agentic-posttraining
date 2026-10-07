# V2 开发层状态（CPU 阶段）

更新时间：2026-10-07

## 已完成

- 新建独立目录 `v2_development`，不改父项目算法代码、数据、模型和权重。
- 完成标准库实现：Model Adapter、Typed Agent State、State Transition/Diff、JSON Checkpoint、Failure Taxonomy、Recovery Policy、Tool Registry、Agent Runtime。
- `MockModelAdapter` 可以在 CPU 上驱动多步工具调用；GPU/HF/PEFT adapter 只保留契约和明确的 deferred 状态。
- 内置 `calculator`、`workspace_set`、`workspace_get` 三个确定性工具。
- 9 个单元测试通过，CPU smoke 通过。

## 证据边界

当前结果只证明：

```text
CPU runtime contract + deterministic tools + state/checkpoint/recovery wiring
```

当前结果不能证明：

```text
真实模型 Tool-use 能力
模型训练收益
V1 Adapter 的 native capability
真实长程任务成功率
```

## GPU 恢复后的第一批工作

1. 完成 V1 model/data/result/environment manifest，并确认冻结 commit。
2. 用 `PEFTModelAdapter` 接入冻结的 base model 与 adapter；先跑单个 tool-call 回归。
3. 固定 prompt、generation config、tool、task、seed、max steps 和 token budget。
4. 建立 B0–B5 baseline，再进入 2/4/8/16-step long-horizon benchmark。
5. 将每条结果分为 Raw、Normalized、Resolved、Environment Final State，单独记录 Recovery 贡献。

## 不应在当前阶段做的事

不要因 smoke 通过而宣称模型能力提升；不要把 recovery 后结果写成 model native capability；不要在 GPU 未恢复前加载父目录权重或运行训练任务。
