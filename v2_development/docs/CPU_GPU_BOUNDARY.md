# CPU / GPU 边界

## 当前无 GPU 时允许做的事情

- 编译和运行状态、transition、diff、checkpoint、failure、recovery、registry。
- 用 `MockModelAdapter` 跑确定性 Agent loop。
- 构造 long-horizon task schema、fault injection 和 metrics 接口。
- 编写真实模型 adapter 的配置、契约和测试替身。

## GPU 恢复后才执行的事情

- 加载父项目中经过 V1 freeze 的 base model 和 adapter。
- 真实 generation、tool-call 能力评估、SFT/DPO/RL 复现实验。
- CUDA/显存/量化/吞吐/延迟测量。
- B0–B5 在真实模型上的对照和 long-horizon 结果。

## 强制边界

没有 GPU 时，`HuggingFaceModelAdapter` 和 `PEFTModelAdapter` 只返回模型元信息或抛出明确的 `GpuRequiredError`，不得静默回退成“模型成功运行”。Mock 结果只能标记为 runtime/engineering smoke evidence，不能写进 model native capability 结论。
