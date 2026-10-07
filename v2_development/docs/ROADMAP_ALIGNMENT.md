# V2 路线映射

本文件把上传的研发路线映射到当前开发层，防止“功能已写”被误认为“实验结论已证明”。

| 路线阶段 | 当前状态 | CPU 阶段证据 | GPU/后续工作 |
| --- | --- | --- | --- |
| Phase 0 V1 Freeze | 入口已建立 | `README` 与 Evidence Gate 模板 | 读取 V1 只读证据，生成正式 manifests |
| Phase 1 结构重构 | 独立层已建立 | `src/llm_agent_v2` 不依赖父项目 | 通过 adapter/wrapper 逐步接入稳定模块 |
| Phase 2 Model Adapter | CPU contract + Mock 已实现 | `models.py` 单测、smoke | HF/PEFT/API 真实实现 |
| Phase 3 Typed State | 已实现 | 序列化、immutable 字段保护测试 | 接入真实环境状态 |
| Phase 4 Transition/Diff | 已实现 | 工具调用前后状态 diff 测试 | 补充复杂 workspace diff |
| Phase 5 Checkpoint | JSON 首版已实现 | 原子保存/加载测试 | 真实环境元数据、SQLite 评估 |
| Phase 6 Failure Taxonomy | F1–F12 枚举和最小分类器 | 分类单测 | 完整 trajectory 标注与统计 |
| Phase 7 Recovery | 策略选择和限次重试已实现 | recovery 单测、loop break | fault injection + conditional recovery |
| Phase 8–10 Benchmark/Evaluation | smoke 骨架 | metrics 输出 | 2/4/8/16 step、B0–B5、多 seed、CI |
| Phase 12 Memory | 尚未实现 | 不提前引入依赖 | State/Checkpoint/Recovery 稳定后实现 |
| Phase 13–15 Serving/Plugin | 尚未实现 | 保留接口边界 | session、trace、auth、plugin |

## Evidence Gate

进入下一阶段前必须补齐：问题、baseline、指标、收益、额外代价、failure 变化和复现命令。当前 CPU smoke 只证明组件可运行，不证明模型能力提升。
