# LLM Agentic Post-Training

## 项目定位

本项目是一个面向长程工具调用任务的“大模型能力适配 + Agent Runtime + 可靠性评测 + 服务化”研发平台。

项目不把目标限定为训练一个能够调用工具的模型，而是完整研究下面这条工程链路：

~~~text
数据与后训练
    ↓
模型能力适配
    ↓
Model Adapter
    ↓
Agent Runtime
    ↓
Typed State
    ↓
Tool / Environment
    ↓
Checkpoint
    ↓
Failure Detection
    ↓
Failure-aware Recovery
    ↓
Evaluation / Trace
    ↓
Serving / Observability
~~~

项目同时覆盖四类研发能力：

| 能力方向 | 主要内容 |
| --- | --- |
| 大模型开发 | 模型加载、PEFT/QLoRA、SFT、DPO、轻量 Agentic RL、推理和评测 |
| 智能体开发 | Agent Runtime、工具注册、状态转移、环境交互、轨迹和失败恢复 |
| 可靠性工程 | Typed State、Checkpoint、Failure Taxonomy、Recovery Policy、长程任务评测 |
| 应用工程 | Model Adapter、FastAPI、Session、Trace、指标、Web Demo 和服务化接口 |

## V1 基础能力

V1 是算法层和实验层，已经形成一条可复用的后训练与 Agent 评测流水线：

~~~text
Dataset → Training → Model/Adapter → Agent Runtime → Reward → Benchmark → Report
~~~

当前代码包含以下基础模块：

1. **模型与适配器**：基础模型加载、PEFT Adapter 推理、模型配置和设备检查。
2. **后训练**：SFT/QLoRA、偏好数据构造、DPO，以及不依赖 Critic 的最小组相对策略梯度实验。
3. **Agent Runtime**：统一的对话/工具调用协议、动作解析、轨迹记录、终止条件和环境执行。
4. **工具与环境**：计算器、文件操作、代码执行、单元测试、离线检索和 JSON 校验等工具。
5. **奖励与评测**：任务成功、工具调用、协议一致性、语义成功、严格成功和奖励计算。
6. **实验闭环**：基准评测、失败样本整理、偏好数据构造、消融脚本和证据记录。
7. **服务化**：Transformers/PEFT 推理路径、FastAPI 接口、健康检查和最小 Web 客户端。

V1 的核心价值是把“模型是否会做”与“Runtime 是否帮它完成”分层记录。报告必须区分：

~~~text
Raw Model Output
    → Normalized Output
    → Resolved Action
    → Environment Result
    → Final Semantic/Strict Metric
~~~

运行时重试或环境侧修复不能直接写成模型原生能力，这一证据边界是后续所有实验的基础。

## V2 研发方向

V2 不推翻 V1，也不以无限增加训练算法为目标。V2 在已有模型、Adapter、Runtime、Tool、Reward、Evaluation 和 Serving 之上，补齐 Agent 的状态与可靠性工程：

~~~text
V1 Evidence Freeze
    ↓
Model Adapter
    ↓
Typed State
    ↓
State Transition / Diff
    ↓
Checkpoint
    ↓
Failure Taxonomy
    ↓
Recovery Policy
    ↓
Long-horizon Benchmark
    ↓
Memory
    ↓
Serving V2 / Observability
~~~

V2 的主要研究问题是：

> 当工具任务的步骤数和环境状态复杂度增加时，显式 Typed State、Checkpoint 和 Failure-aware Recovery 能否减少错误传播，并在可控的 Token、步骤和延迟成本下提高最终任务成功率？

### Model Adapter

Agent Runtime 只依赖统一的模型接口，不直接绑定某一种本地模型：

~~~text
generate()
chat()
stream()
tool_generate()
get_model_info()
~~~

计划支持本地 HuggingFace 模型、PEFT 模型、兼容 API 的模型和 Mock 模型。统一适配层负责模型版本、生成参数、流式输出和工具调用能力，Runtime 只消费适配器接口。

### Typed State

State 不只保存自然语言历史，而是保存可验证的结构化任务状态：

~~~text
task_id
goal
completed_actions
pending_actions
tool_observations
workspace_state
evidence_refs
memory_refs
current_subgoal
last_tool_status
failure_history
retry_count
budget
uncertainty
checkpoint_id
termination_allowed
~~~

字段按生命周期分为 Immutable、Mutable 和 Derived 三类。状态必须能够说明当前目标、已完成动作、待执行动作、观察证据、预算和终止条件。

### State Transition

每次工具调用都形成一个可回放的转移：

~~~text
State_t → Action_t → Tool → Observation_t → State_t+1
~~~

至少保留 before_state、action、observation、after_state 和 state_diff。这样可以定位 Agent 在哪一步开始偏离，而不仅仅是记录最终失败。

### Checkpoint 与 Recovery

Checkpoint 首版使用 JSON 或 SQLite 即可，保存 Agent State、Workspace Metadata、Tool History、Budget 和 Environment State。触发时机包括工具成功、不可逆操作前、阶段完成后和进入 Recovery 前。

首版 Failure Taxonomy 包括：

~~~text
Planning Error
Tool Selection Error
Argument Error
Tool Execution Error
Observation Error
State Drift
Memory Error
Repeated Action / Loop
Premature Final
Budget Exhaustion
Environment Error
Protocol Error
~~~

Recovery 流程统一为：

~~~text
detect_failure()
    → classify_failure()
    → select_recovery()
    → execute_recovery()
    → verify_recovery()
~~~

Recovery 只能使用当前状态、工具错误、当前观察和历史轨迹，不能读取期望动作、标准答案、评测反馈、未来观察或测试标签。

### Long-horizon Evaluation

内部评测按任务长度建立 2-step、4-step、8-step、16-step 分层桶。任务类型包括计算、结构化 JSON、文件操作、代码执行、检索后读写、多文件处理和“代码 → 测试 → 修复 → 再测试”。

建议对照组保持模型、Prompt、工具、任务、Token Budget 和最大步数一致：

~~~text
B0  Direct Answer
B1  ReAct
B2  ReAct + Trace
B3  ReAct + Typed State
B4  Typed State + Checkpoint
B5  Typed State + Checkpoint + Recovery
~~~

环境可以可重复地注入超时、非法参数、空观察、过期观察、部分结果、文件缺失、测试失败、重复调用和环境变化，以建立可核验的可靠性基准。

### Serving 与 Observability

服务化生命周期为：

~~~text
Request → Auth → Session → Model → Agent → Tool → Trace → Response
~~~

每个请求最终应能关联 session_id、task_id、trace_id、model_version、agent_version、prompt_version、tool_calls、steps、failures、recovery、tokens、latency 和 outcome。

## 代码结构

~~~text
.
├── configs/                 # 模型与 Agent Runtime 配置
├── requirements/            # 分阶段依赖
├── scripts/                 # 数据、训练、评测、审计和服务入口
├── src/llm_posttrain/       # V1 算法、Runtime、工具、评测和服务代码
├── tests/                   # V1 单元测试与回归测试
├── web/                     # 本地服务的最小 Web 客户端
├── v2_development/          # V2 大模型与 Agent 开发层
│   ├── configs/             # CPU 演示和未来 GPU 配置
│   ├── src/llm_agent_v2/    # Adapter、State、Checkpoint、Recovery、Runtime
│   ├── tests/               # V2 纯 CPU 测试
│   └── docs/                # V2 边界、证据门和路线说明
├── pyproject.toml
└── README.md
~~~

V1 包中，agent 负责协议、运行时、代码环境和动作处理；tools 负责工具注册及 schema；models 负责基础模型和 Adapter；training、rl 负责后训练；rewards、evaluation 负责评价；serving 提供本地 API。

V2 开发层用于逐步完成模型适配接口、结构化 State、State Transition、Checkpoint、失败分类、Recovery 和 CPU 可验证的协议/状态/工具单测。获得 GPU 后，再接入真实模型推理、训练和服务。

## 当前无 GPU 环境的开发边界

无 GPU 时先完成：

- 接口、数据结构和状态转移；
- 工具 schema、解析和错误分类；
- Checkpoint 序列化与恢复；
- Recovery Policy 的纯逻辑测试；
- MockModelAdapter 和确定性环境；
- CPU smoke test、静态检查和文档；
- 训练/推理入口的参数校验和失败前置检查。

以下工作等待具备 CUDA 的执行节点：

- 基础模型加载与批量推理；
- QLoRA/SFT、DPO 和 Agentic RL 训练；
- Adapter 评测与长程模型对比；
- GPU 上的吞吐、显存、延迟和服务压力测试。

GPU 任务必须使用同一份 manifest、Prompt、Tool 和 Task 配置，并保留原始输出、归一化输出、环境状态、指标和运行时信息。

## 本地开发与验证

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/phase01.txt
python -m pip install -e .
~~~

V1 基础检查：

~~~bash
PYTHONPATH=src python scripts/check_env.py
PYTHONPATH=src pytest -q
~~~

V2 CPU 检查：

~~~bash
cd v2_development
python -m unittest discover -s tests -p 'test_*.py' -v
PYTHONPATH=src python scripts/smoke_cpu.py
~~~

GPU 任务开始前，先运行入口的 --help 和环境检查，确认模型路径、Adapter 路径、数据 manifest、CUDA、PyTorch、Transformers 和 PEFT 版本。

## Evidence Gate

每个新阶段必须先回答：

1. 解决了什么明确问题？
2. Baseline 是什么？
3. 指标如何定义？
4. 是否存在量化收益？
5. 增加了多少步骤、Token、显存或延迟成本？
6. Failure 分布发生了什么变化？
7. 结果能否独立复现？

无法回答时，保留为诊断实验，不进入正式能力结论，也不替换稳定基线。

## 数据、权重与运行产物

GitHub 版本只保存代码、配置、测试和可审阅的说明。以下内容必须放在仓库外，并通过配置或命令行传入：

- 基础模型和 Adapter 权重；
- 原始、训练、验证、Holdout 和评测数据；
- rollout、日志、预测结果和训练曲线；
- 运行环境、缓存和虚拟环境；
- API Key、Token、私有数据和服务器凭据。

发布前必须检查 git ls-files、文件大小、敏感信息扫描和 git diff --check。任何不能脱离当前机器复现的固定路径、凭据或结果，都不应写成仓库内的硬编码依赖。

## 后续研发优先级

1. 冻结并核对 V1 证据边界；
2. 完成 Model Adapter 解耦；
3. 完成 Typed State 与 State Transition；
4. 完成 Checkpoint 和可回放轨迹；
5. 建立 Failure Taxonomy 与 Recovery Policy；
6. 建立 2/4/8/16-step 长程基准和固定对照组；
7. 先在内部基准稳定后再接入外部基准；
8. 增加 Session、Trace、Timeout、Queue 和服务健康检查；
9. 在核心实验稳定后再增加 Memory、权限和领域插件。

暂不以新增大量 RL 算法、多 Agent、复杂 GraphRAG、集群化基础设施或大规模 UI 为优先事项。每次扩展都必须由基线、指标和证据门驱动。

## 交接清单

1. 阅读本文件和 v2_development/docs/ 下的边界说明；
2. 检查 configs/ 与 pyproject.toml，确认依赖和默认路径；
3. 在 CPU 环境执行 V1/V2 smoke test；
4. 只使用外部路径挂载模型和数据，不将它们复制进 Git；
5. 先补充测试，再修改 Runtime、State 或 Recovery；
6. 为每个实验保存 manifest、baseline、原始轨迹、分层指标和失败分类；
7. 将模型原生能力、Runtime 修复和环境侧行为分开汇报；
8. 完成 git diff --check 和敏感信息检查后再提交。

项目的最终叙事是：模型负责提供 Tool-use 能力，Agent Runtime 负责组织行动，State/Checkpoint/Recovery 负责在多步骤环境中维持可靠性，Evaluation 负责给出可核验的证据，Serving 负责把能力变成可调用的服务。
