# llm-agentic-posttraining V2 项目介绍

> 面向长程工具调用的大模型 Agent 开发、运行时可靠性与服务化平台

更新时间：2026-10-07
文档版本：V2 Development Layer / CPU Foundation 0.1
当前运行条件：服务器无可用 GPU；CPU 开发链路已验证

---

## 1. 项目摘要

llm-agentic-posttraining 是一个围绕“大模型能力适配”和“智能体可靠运行”展开的完整研发项目。

项目不再只关注把模型训练成“会调用工具”，而是继续解决工具调用进入真实、多步骤、带环境状态的任务之后的一系列工程问题：

- 模型如何统一接入本地模型、PEFT Adapter 和 API 模型？
- Agent 如何在多轮工具调用中维护结构化状态？
- 每一次行动如何形成可审计的状态转移？
- 工具成功、失败、重复调用和环境变化如何被记录？
- Agent 跑偏后如何恢复，而不是简单地无限 Retry？
- 如何用 long-horizon benchmark 证明 State、Checkpoint 和 Recovery 的真实收益？
- 如何把模型和 Agent Runtime 封装成可观测、可限流、可复现的服务？

因此，项目的核心叙事是：

> 我不是只训练一个会调用工具的模型，而是在实现从“模型具备 Tool-use 能力”到“Agent 能在真实多步骤环境中可靠完成任务”的完整工程链路。

项目由两层组成：

1. **V1 Algorithm Layer**：保留模型、Adapter、训练流程、工具环境、Reward、Trace 和既有实验结果，作为算法能力与历史证据基线。
2. **V2 Development Layer**：在 v2_development 下独立开发大模型适配、Agent Runtime、Typed State、Checkpoint、Failure Recovery、Evaluation 和 Serving 能力。

V2 开发层是新增隔离目录，不修改父目录已有算法代码、数据、模型和权重。

---

## 2. 项目重新定位

### 2.1 V1 的定位

V1 本质上是：

> 面向代码与工具调用任务的 LLM Agent Post-training 实验平台。

V1 重点证明：

- QLoRA / SFT 等模型后训练能力
- DPO 与 Agentic RL 的实验能力
- Tool Calling 与 Agent Environment
- Reward、Benchmark、Trace 和 Failure Analysis
- FastAPI 与 Web Demo 等基础工程能力

V1 更偏向回答：

> 模型是否具备工具使用能力？训练方法是否带来可测量的能力变化？

### 2.2 V2 的定位

V2 的重点调整为：

> 面向长程工具调用任务的大模型 Agent 开发、运行时可靠性与服务化平台。

技术定位为：

~~~text
LLM Capability Adaptation
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

V2 不推翻 V1，也不以继续无限增加训练算法为目标。模型能力线保留并维护，主要研发资源转向 Agent Runtime、可靠性、评测和服务化。

---

## 3. 面向的岗位与能力画像

| 方向 | 项目中对应的重点能力 |
| --- | --- |
| 大模型开发工程师 | Base Model / PEFT Adapter、模型推理抽象、Generation Config、模型评测、Serving Adapter |
| 大模型应用开发工程师 | API 模型接入、Tool Registry、Session、Trace、权限、限流和服务化 |
| 智能体开发工程师 | Runtime、Planner、Typed State、Tool、Checkpoint、Memory、Recovery |
| 智能体应用开发工程师 | Workflow、Plugin、业务环境、Evidence、Failure Analysis、可观测性 |

项目的优势在于把模型侧、Agent 侧、评测侧和工程侧串成一条完整链路，而不是只展示某一个孤立 Demo。

---

## 4. 核心研究问题与工程问题

### 4.1 核心研究问题

> 当 Tool Agent 的任务长度和环境状态复杂度增加时，显式 Typed State、Checkpoint 与 Failure-aware Recovery 能否降低错误传播并提高最终 Task Success？

对应的假设为：

#### H1：Typed State 降低状态相关错误

显式状态应当降低：

- State Drift
- Invalid Tool Call
- Repeated Action
- 因上下文不完整导致的错误规划

#### H2：Checkpoint + Recovery 提高恢复成功率

当工具调用失败、观测过期或状态发生偏移时，Agent 不应从头开始或盲目 Retry，而应根据失败类型选择恢复策略，从而提升：

- Failure Detection Accuracy
- Conditional Recovery Rate
- Task Success Rate

#### H3：可靠性收益必须覆盖额外成本

Recovery 带来的收益必须与额外成本一起报告：

- Additional Steps
- Additional Tokens
- Tool Calls
- Latency / p95 Latency
- Checkpoint Storage

### 4.2 工程问题

V2 同时解决以下工程问题：

1. 模型供应商和模型类型变化时，Runtime 不需要重写。
2. 每次工具调用都能还原调用前后的 State。
3. Agent 的失败可以分类，而不是只记录一个字符串错误。
4. Recovery 有明确策略和边界，不把未来答案或 Evaluator Feedback 偷渡给 Agent。
5. CPU 无 GPU 时仍然可以开发和测试 Runtime，不阻塞整个项目研发。
6. GPU 恢复后可以把真实模型接入同一套 Runtime 与 Evaluation，不改变核心实验接口。

---

## 5. 总体系统架构

~~~text
┌──────────────────────────────────────────────────────────────┐
│                     Model / Data Layer                       │
│  Base Model · QLoRA/SFT/DPO/RL · PEFT Adapter · Manifest     │
└──────────────────────────────┬───────────────────────────────┘
                               │ ModelAdapter
┌──────────────────────────────▼───────────────────────────────┐
│                     Agent Runtime Layer                      │
│  Request → Generate → Tool Call → Observation → Transition   │
└──────────────┬───────────────────────────┬───────────────────┘
               │                           │
┌──────────────▼─────────────┐  ┌──────────▼───────────────────┐
│       Typed Agent State    │  │       Tool / Environment       │
│ goal / budget / evidence   │  │ Registry / schema / execution  │
│ workspace / failures       │  │ observation / state patch     │
└──────────────┬─────────────┘  └──────────┬───────────────────┘
               │                           │
               └──────────────┬────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────┐
│               Transition / Checkpoint / Trace                │
│ State_t → Action_t → Observation_t → State_t+1               │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│          Failure Taxonomy → Recovery Policy → Evaluation     │
│  classify · replan · retry · restore · break loop · abstain   │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│               Serving / Session / Auth / Metrics             │
└──────────────────────────────────────────────────────────────┘
~~~

### 5.1 关键设计原则

- **Adapter / Wrapper 优先于 Rewrite**：V2 不大规模重写 V1 已经稳定的算法和环境模块。
- **Model 与 Runtime 解耦**：Runtime 只依赖 ModelAdapter，不依赖 Qwen、Transformers 或某个 API 厂商。
- **状态优先于自然语言历史**：关键任务状态必须结构化保存，不能只存在于对话上下文中。
- **每次工具调用都可审计**：所有行动形成 Transition 和 State Diff。
- **Recovery 只使用当前可见证据**：只能使用当前 State、Tool Error、Observation 和 History，不能读取 Expected Action、Golden Answer 或未来 Observation。
- **结果分层保存**：Raw、Normalized、Resolved 和 Environment Final State 不混为一谈。
- **CPU/GPU 能力边界明确**：无 GPU 时验证 Runtime 工程链路；有 GPU 后再验证真实模型能力。

---

## 6. 大模型开发工作

大模型开发不是本项目的附属部分，而是 Agent 能力的模型基础。V2 对模型侧的工作重点是“适配、推理、版本化和评测”，而不是无限追加新的 RL 算法。

### 6.1 V1 模型能力基础

父项目保留以下算法能力作为 V1 基线：

- Base Model 管理
- QLoRA / SFT
- DPO
- Agentic RL 最小实验
- Tool-use 数据与 Agent Environment
- Reward、Benchmark 和 Failure Analysis
- Adapter 配置与实验结果

这些能力用于回答模型原生能力问题。V2 Runtime 的 Recovery 结果不能倒写为模型原生能力。

### 6.2 V2 统一 Model Adapter

V2 新增统一的模型适配边界：

~~~python
class ModelAdapter:
    generate()
    chat()
    stream()
    tool_generate()
    get_model_info()
~~~

Runtime 不再关心模型是：

- Qwen3-1.7B 本地模型
- Hugging Face 模型
- PEFT Adapter 模型
- OpenAI-compatible API 模型
- CPU 测试用 Mock 模型

Runtime 只依赖统一的 ModelAdapter。

### 6.3 当前实现状态

当前 V2 开发层已经实现：

- MockModelAdapter：确定性驱动多步 Tool Calling，用于 CPU 测试和 Smoke。
- HuggingFaceModelAdapter：配置和接口层，当前不加载权重。
- PEFTModelAdapter：保存 base model 与 adapter 的接入边界，当前不加载权重。
- OpenAICompatibleAdapter：API 模型接入契约，当前不发起外部网络请求。
- GpuRequiredError：GPU 不可用时显式失败，不静默伪装成真实模型成功。

对应代码：

    src/llm_agent_v2/models.py

### 6.4 当前模型侧的证据边界

当前服务器没有 GPU，nvidia-smi 返回 No devices were found。因此当前已验证的是：

~~~text
Model Adapter contract
Mock generation
Runtime integration
CPU testability
~~~

当前没有宣称：

- Qwen 或其他真实模型的 Tool-use 成功率
- Adapter 带来的模型能力提升
- 量化后的吞吐、显存和延迟
- 真实模型的 long-horizon 任务结果

GPU 恢复后，模型侧工作顺序为：

1. 读取 V1 Model Manifest，确认 Base Model、M2g Adapter、关键 Adapter 和配置。
2. 验证 CUDA、Torch、Transformers、PEFT 和显存条件。
3. 通过 PEFTModelAdapter 接入冻结的 base model 与 adapter。
4. 固定 Prompt、Generation Config、Tool、Task、Seed、Max Steps 和 Token Budget。
5. 先跑单工具调用回归，再跑多步骤 Agent Runtime。
6. 将 Model Native Capability 与 Runtime Recovery 贡献分别统计。

---

## 7. 智能体开发工作

V2 的核心开发工作集中在 Agent Runtime 及其可靠性机制。

### 7.1 Agent Runtime

Runtime 的基本执行循环为：

~~~text
接收 Goal
   ↓
读取 Typed State
   ↓
调用 ModelAdapter
   ↓
解析 ToolCall 或 Final Answer
   ↓
调用 ToolRegistry
   ↓
记录 Observation
   ↓
更新 State
   ↓
生成 Transition / Checkpoint
   ↓
继续、恢复或终止
~~~

当前 CPU Runtime 已支持：

- 多轮工具调用
- 工具参数传递
- 工具结果记录
- State 更新
- Tool Success Checkpoint
- Failure Recovery Checkpoint
- Final Answer
- Step Budget 和 Token Budget
- GPU 缺失时的明确状态返回

对应代码：

    src/llm_agent_v2/runtime.py

### 7.2 Typed Agent State

AgentState 是 V2 的第一个核心模块。它把原来散落在自然语言或临时变量中的任务信息统一成结构化状态。

当前字段包括：

| 类型 | 字段 | 作用 |
| --- | --- | --- |
| Immutable | task_id | 任务身份，不允许在工具调用中改变 |
| Immutable | goal | 原始目标，不允许被工具 Patch 改写 |
| Immutable | environment_id | 当前环境身份 |
| Mutable | completed_actions | 已完成的行动 |
| Mutable | pending_actions | 待完成行动 |
| Mutable | tool_observations | 工具观察结果 |
| Mutable | workspace_state | 环境和工作区结构化状态 |
| Mutable | evidence_refs | 证据引用 |
| Mutable | memory_refs | 记忆引用 |
| Mutable | failure_history | 失败历史 |
| Mutable | retry_count | 重试次数 |
| Mutable | checkpoint_id | 最近 Checkpoint |
| Derived | remaining_budget | 剩余步骤预算 |
| Derived | remaining_tokens | 剩余 Token 预算 |
| Derived | steps_used | 已用步骤 |
| Derived | tokens_used | 已用 Token |
| Control | termination_allowed | 是否允许终止 |

当前状态实现具备：

- clone()：生成安全快照。
- to_dict() / from_dict()：支持 Checkpoint 和 Trace。
- Immutable 字段保护：State Patch 不能修改 task_id、goal、environment_id。
- Workspace State Patch：工具可以显式更新结构化工作区状态。
- Budget tracking：步骤和 Token 预算由 Runtime 统一维护。

对应代码：

    src/llm_agent_v2/state.py

### 7.3 State Transition 与 State Diff

每一次工具调用形成以下可审计链路：

~~~text
State_t
   ↓
Action_t
   ↓
Tool
   ↓
Observation_t
   ↓
State_t+1
~~~

每条 Transition 保存：

- before_state
- action
- observation
- after_state
- state_diff

它可以回答：

- Agent 在哪一步开始偏离？
- 工具是否真的改变了 workspace？
- 失败发生在模型决策、参数、工具执行还是状态更新？
- Recovery 后 State 是否恢复到预期？

对应代码：

    src/llm_agent_v2/state.py

### 7.4 Tool Registry

Tool Registry 为 Agent 提供统一的工具发现、Schema 校验和执行边界。

当前 CPU 版本包括：

- calculator：安全的纯 Python 数值表达式计算。
- workspace_set：向 Typed Workspace 写入 JSON-compatible 值。
- workspace_get：读取当前 Workspace State。

Registry 已经具备：

- Tool Name 注册
- Tool Description
- Input Schema
- Required 参数校验
- 基本类型校验
- Unknown Tool 分类
- Tool Execution 异常边界
- State Patch 返回

对应代码：

    src/llm_agent_v2/tools.py

后续可以在不修改 Runtime 的前提下增加：

- File Tool
- Python Executor
- Search / RAG Tool
- Test Runner
- SQL Tool
- Browser Tool

### 7.5 Checkpoint

Checkpoint 用于保存 Agent 在关键状态变化后的可恢复快照。

首版采用 JSON，不引入 Redis、分布式存储或数据库集群，降低无 GPU 阶段的开发成本。

当前保存内容包括：

- Agent State
- Transition History
- Workspace Metadata
- Checkpoint Reason
- Created Timestamp
- Checkpoint ID

当前触发点包括：

- Tool Success
- Recovery Decision
- 后续可扩展到不可逆操作前和阶段完成后

JSON 文件采用临时文件写入后原子替换，避免半写入文件成为可恢复状态。

对应代码：

    src/llm_agent_v2/checkpoint.py

### 7.6 Failure Taxonomy

V2 把失败标准化为 F1–F12：

| 编号 | Failure | 含义 |
| --- | --- | --- |
| F1 | Planning Error | 规划错误 |
| F2 | Tool Selection Error | 工具选择错误或工具不存在 |
| F3 | Argument Error | 工具参数错误 |
| F4 | Tool Execution Error | 工具执行异常 |
| F5 | Observation Error | 观测缺失、为空或不可用 |
| F6 | State Drift | Agent 认为的状态与环境状态不一致 |
| F7 | Memory Error | 记忆读取或使用错误 |
| F8 | Loop / Repeated Action | 重复行动或循环 |
| F9 | Premature Final | 任务未完成就提前结束 |
| F10 | Budget Exhaustion | 步骤或 Token 预算耗尽 |
| F11 | Environment Error | 环境错误 |
| F12 | Protocol Error | 模型输出或 Tool Protocol 不符合约定 |

对应代码：

    src/llm_agent_v2/failure.py

### 7.7 Failure-aware Recovery

Recovery 不采用“失败后全部 Retry”的策略，而是根据 Failure Type 选择策略：

| Failure | Recovery Strategy |
| --- | --- |
| F2 Tool Selection | Re-plan / Tool Selection Correction |
| F3 Argument | Re-plan；后续增加 Argument Repair |
| F4 Tool Execution | 有上限的 Retry，超过上限后 Abstain |
| F5 Observation | Refresh Observation |
| F6 State Drift | Restore Checkpoint 或 Re-plan |
| F8 Repeated Action | Break Loop |
| F10 Budget | Abstain |
| F11 Environment | 明确返回边界错误，避免伪造成功 |
| F12 Protocol | Normalize Protocol |

当前 Recovery 层已经实现：

- Failure → Recovery Decision 映射
- 有上限 Retry
- Repeated Action Loop Break
- Abstain
- Recovery 前后 State 记录
- Recovery Action、额外步骤、额外 Token、延迟字段

当前需要明确：首版主要实现了 Recovery Policy 和记录边界，部分 restore、replan、refresh observation 仍是后续接入真实环境时执行的策略，不应被描述成已经完成的全自动恢复系统。

对应代码：

    src/llm_agent_v2/recovery.py

---

## 8. 当前 V2 开发层目录

~~~text
v2_development/
├── .gitignore
├── README.md
├── pyproject.toml
├── configs/
│   ├── cpu_demo.json
│   └── gpu_future.json
├── docs/
│   ├── CPU_GPU_BOUNDARY.md
│   ├── DEVELOPMENT_STATUS.md
│   ├── EVIDENCE_GATE.md
│   ├── PROJECT_INTRODUCTION.md
│   └── ROADMAP_ALIGNMENT.md
├── scripts/
│   └── smoke_cpu.py
├── src/
│   └── llm_agent_v2/
│       ├── __init__.py
│       ├── checkpoint.py
│       ├── failure.py
│       ├── models.py
│       ├── recovery.py
│       ├── runtime.py
│       ├── state.py
│       └── tools.py
└── tests/
    ├── __init__.py
    ├── test_checkpoint.py
    ├── test_runtime.py
    ├── test_state.py
    └── test_tools.py
~~~

当前开发层只依赖 Python 标准库，保证无 GPU、无额外模型下载时仍然可以开发 Runtime 和 Agent 工程。

---

## 9. 一次 Agent 运行的完整过程

以 CPU Smoke 为例：

### Step 1：创建任务和初始状态

~~~text
task_id = cpu-smoke
goal = store an answer and calculate 2 + 3
environment_id = cpu-local
~~~

### Step 2：Model Adapter 生成 Tool Call

~~~json
{
  "name": "workspace_set",
  "arguments": {"key": "answer", "value": 2},
  "call_id": "call-1"
}
~~~

### Step 3：Tool Registry 校验并执行

工具返回：

~~~json
{
  "success": true,
  "state_patch": {"workspace_state": {"answer": 2}},
  "output": "{\"key\": \"answer\", \"value\": 2}"
}
~~~

### Step 4：Runtime 形成 State Transition

~~~text
workspace_state: {}
        ↓
workspace_state: {"answer": 2}
~~~

### Step 5：保存 Checkpoint

工具成功后保存 JSON Checkpoint，包含当前 State 和此前 Transition。

### Step 6：继续下一次 Tool Call

~~~json
{
  "name": "calculator",
  "arguments": {"expression": "2 + 3"},
  "call_id": "call-2"
}
~~~

### Step 7：模型生成 Final Answer

Runtime 记录最终状态、工具调用数、Checkpoint 数、Token 使用量和延迟。

CPU Smoke 的实际验证结果为：

~~~text
status       = success
tool_calls   = 2
checkpoints  = 2
failures     = 0
recoveries   = 0
~~~

这证明的是 Runtime 工程链路，不是某个真实大模型的能力结果。

---

## 10. 当前已经完成的交付与证据

| 交付物 | 内容 | 当前证据 |
| --- | --- | --- |
| Model Adapter | Mock / HF / PEFT / API 接口边界 | 代码可导入，Deferred Adapter 明确抛出 GPU/配置错误 |
| Typed State | Immutable、Mutable、Derived 字段 | 状态单测通过 |
| State Transition | Before / Action / Observation / After / Diff | 状态单测和 Runtime 单测通过 |
| Tool Registry | Schema、参数校验、执行边界 | Tool 单测通过 |
| Checkpoint | JSON、原子写入、加载 | Round-trip 单测通过 |
| Failure Taxonomy | F1–F12 | Runtime 失败场景单测通过 |
| Recovery Policy | Retry、Re-plan、Break Loop、Abstain 等策略 | 重复行动测试通过 |
| CPU Runtime | 多步 Tool Calling 和 Final Answer | Smoke 通过 |
| GPU 边界 | 无 GPU 不加载模型 | nvidia-smi 返回 No devices were found |
| 文档与交接 | README、路线映射、状态和 Evidence Gate | 当前文档目录 |

服务器当前使用 Python 3.11.10。V2 开发层最后一次验证结果：9 个单元测试全部通过，CPU Smoke 退出码为 0。

---

## 11. 无 GPU 阶段可以持续开发的内容

无 GPU 并不阻塞整个项目。当前可以继续完成：

### 11.1 Runtime 工程

- 完善 Planner / Executor 分层
- 多工具调用协议
- Tool Timeout 与错误归一化
- Max Steps、Token Budget、Queue Budget
- Session 和 Trace ID
- State Diff 的嵌套字段比较

### 11.2 Agent Reliability

- Fault Injection
- Timeout、Invalid Parameter、Empty Observation、Stale Observation
- Partial Result、File Missing、Test Failure、Duplicate Call
- Failure Detection Accuracy
- Conditional Recovery Rate
- Loop Rate 和 Abstention Rate

### 11.3 CPU Evaluation

- 2-step、4-step、8-step、16-step 任务桶
- Calculation、Structured JSON、File Manipulation 的确定性任务
- Search → Read → Transform → Write 的模拟环境
- Code → Test → Fix → Test 的 Mock Environment
- B0–B5 baseline 的统一配置格式
- Metrics 聚合和结果表输出

### 11.4 Serving Contract

- /health
- /v1/chat/completions
- /v1/agent/run
- Session、Task、Trace、Model Version、Agent Version
- Timeout、Max Steps、Token Budget
- Tool Permission 和 Workspace Permission

这些工作可以先通过 Mock Model、Deterministic Tool 和 CPU Environment 验证，不需要加载大模型权重。

---

## 12. GPU 恢复后的研发计划

### Phase 0：V1 Evidence Freeze

冻结以下内容：

- Base Model
- M2g Adapter 和关键 Adapter
- Adapter Config
- Frozen Dataset、Final Test、Holdout、Manifest、Hash
- M2g Native、Guard-on、DPO NO_GO、GRPO Pilot 等实验状态
- Git Commit、Python、Torch、Transformers、PEFT、CUDA 和硬件信息
- Raw、Normalized、Resolved、Semantic、Runtime、Strict、Reward、Failure Evidence

输出：

~~~text
V1_ALGORITHM_FREEZE.md
V1_MODEL_MANIFEST.json
V1_DATA_MANIFEST.json
V1_RESULT_MANIFEST.json
V1_ENVIRONMENT.md
~~~

### Phase 1：真实 Model Adapter

使用冻结的 V1 Model Manifest 接入：

~~~text
HuggingFaceModelAdapter
PEFTModelAdapter
~~~

先验证单个 Tool Call，再验证多步骤 Runtime。必须保留 Model Native 与 Runtime Recovery 的分层证据。

### Phase 2：真实长程评测

统一固定：

- Model
- Prompt
- Tool
- Task
- Seed
- Max Steps
- Token Budget
- Generation Config

运行：

~~~text
2-step
4-step
8-step
16-step
~~~

### Phase 3：Baseline 与消融

建议 baseline：

| Baseline | State | Trace | Checkpoint | Recovery |
| --- | --- | --- | --- | --- |
| B0 Direct Answer | × | × | × | × |
| B1 ReAct | × | × | × | × |
| B2 ReAct + Trace | × | ✓ | × | × |
| B3 Typed State | ✓ | ✓ | × | × |
| B4 State + Checkpoint | ✓ | ✓ | ✓ | × |
| B5 State + Checkpoint + Recovery | ✓ | ✓ | ✓ | ✓ |

### Phase 4：外部 Benchmark

内部 Benchmark 稳定后再接外部 Benchmark，优先顺序：

1. BFCL：Function Calling、Tool Selection、Argument、Multi-turn。
2. τ-Bench：Stateful Tool Agent、Policy、Database State、pass^k。
3. 根据求职方向选择 SWE-bench 子集、BrowserGym 或 BEIR。

内部结果没有稳定前，不同时接入多个外部 Benchmark。

---

## 13. Evaluation 设计

项目不能只报告 Reward，应同时报告以下指标。

### Task Metrics

- Task Success Rate
- Semantic Success
- pass^k

### Tool Metrics

- Tool Selection Accuracy
- Argument Accuracy
- Tool Execution Success
- Invalid Call Rate

### State Metrics

- State Consistency Rate
- State Drift Rate
- Workspace State Correctness

### Recovery Metrics

- Failure Detection Accuracy
- Recovery Attempt Rate
- Conditional Recovery Rate
- Recovery Cost
- Recovery Action Distribution

### Efficiency Metrics

- Average Steps
- Token Cost
- Tool Calls
- Mean Latency
- p95 Latency
- Checkpoint Storage

### Reliability Metrics

- Loop Rate
- Premature Final Rate
- Abstention Rate
- Repeated Action Rate
- Budget Exhaustion Rate

最终结果应当能够形成如下结论，而不是只堆模块名称：

> 随着任务长度增加，普通 ReAct 的成功率下降；显式 State、Checkpoint 和 Failure-aware Recovery 能够降低错误传播，并在可接受的额外 Step、Token 和 Latency 成本内提高最终 Task Success。

---

## 14. 数据、模型和证据边界

### 14.1 父项目边界

. 是算法层、模型层和历史实验层。V2 开发层只在其下新增：

~~~text
v2_development
~~~

现有父项目文件、数据、模型、Adapter 和权重不作为 V2 首版的写入目标。

### 14.2 结果分层

每条实验结果应保持：

~~~text
Raw
  ↓
Normalized
  ↓
Resolved
  ↓
Environment Final State
~~~

尤其要区分：

- Model Native Capability
- Agent Runtime Capability
- Recovery Contribution
- Environment / Tool Contribution

### 14.3 Recovery 禁止作弊

Recovery 不允许读取：

- Expected Action
- Golden Answer
- Evaluator Feedback
- Future Observation
- Final Test Label

Recovery 只能使用：

- Current State
- Tool Error
- Current Observation
- History
- 已保存的合法 Checkpoint

---

## 15. 运行与验证

进入 V2 开发层：

~~~bash
cd v2_development
~~~

运行单元测试：

~~~bash
python -m unittest discover -s . -p "test_*.py" -v
~~~

运行 CPU Smoke：

~~~bash
python scripts/smoke_cpu.py
~~~

查看 CPU/GPU 边界：

    docs/CPU_GPU_BOUNDARY.md

查看路线映射：

    docs/ROADMAP_ALIGNMENT.md

查看证据门禁：

    docs/EVIDENCE_GATE.md

### 当前验证结论

~~~text
Python              3.11.10
Unit Tests          9 passed
CPU Smoke           passed
GPU                 No devices were found
Real Model Loading  deferred
Parent Data Change  none
~~~

---

## 16. 交接给后续开发者的建议顺序

### 第一步：阅读本目录文档

先阅读：

1. README.md
2. docs/PROJECT_INTRODUCTION.md
3. docs/DEVELOPMENT_STATUS.md
4. docs/ROADMAP_ALIGNMENT.md
5. docs/EVIDENCE_GATE.md

### 第二步：跑通 CPU 基础链路

~~~bash
python -m unittest discover -s . -p "test_*.py" -v
python scripts/smoke_cpu.py
~~~

### 第三步：理解 Runtime 代码路径

建议阅读顺序：

~~~text
models.py
  → tools.py
  → state.py
  → checkpoint.py
  → failure.py
  → recovery.py
  → runtime.py
~~~

### 第四步：先补 CPU Fault Injection

不要先加载真实模型。先增加确定性 Fault Injection，并验证：

- F3 参数错误是否被识别
- F4 工具异常是否限次 Retry
- F5 空观测是否触发 Refresh
- F6 State Drift 是否可以 Restore
- F8 重复行动是否 Break Loop
- F10 预算耗尽是否 Abstain

### 第五步：建立 B0–B5 CPU Baseline

保证每个 baseline 使用相同的 Task、Tool、Budget 和 Trace 格式，只改变 State、Checkpoint 和 Recovery 开关。

### 第六步：GPU 恢复后接入真实模型

只有在 V1 Evidence Freeze 和环境验证完成后，才将冻结的 Base Model 与 Adapter 接入 PEFTModelAdapter。

---

## 17. 对外项目介绍模板

### 17.1 30 秒版本

我做的是一个面向长程工具调用的大模型 Agent 开发与可靠性平台。前一层负责模型后训练、Adapter 和 Tool-use 能力，后一层负责把模型接入 Agent Runtime，通过 Typed State、Checkpoint 和 Failure-aware Recovery 处理多步骤任务中的状态漂移、工具错误和重复行动，并用 long-horizon benchmark 评估成功率、恢复率、Token 和延迟成本。

### 17.2 技术面版本

项目重点不是继续堆叠训练算法，而是研究模型具备 Tool-use 能力后，如何在真实多步骤环境中稳定运行。我将模型侧抽象成统一的 ModelAdapter，将 Agent 状态结构化为 Typed AgentState，并让每次工具调用生成 State Transition、Observation 和 State Diff。针对 F1–F12 失败类型设计不同 Recovery Policy，例如参数错误 Re-plan、工具异常限次 Retry、重复行动 Break Loop、预算耗尽 Abstain。最终通过 2/4/8/16-step 的 long-horizon benchmark 分离评估 Model Native Capability、Runtime Recovery 和 Environment Contribution。

### 17.3 当前阶段可准确使用的简历表述

可以表述为：

> 设计并实现面向长程工具调用 Agent 的 CPU-first Runtime Foundation，完成 Model Adapter 抽象、Typed State、State Transition/Diff、JSON Checkpoint、F1–F12 Failure Taxonomy 和 Failure-aware Recovery Policy；通过 Mock Model 驱动多步 Tool Calling，完成 9 项单测和 CPU Smoke 验证，并明确真实模型能力与 Runtime Recovery 的证据边界。

当前不应表述为：

- 已完成真实大模型的长程任务评测
- 已证明 Recovery 提升了真实模型 Task Success
- 已完成 Production Sandbox
- 已完成 GPU 训练、吞吐和显存优化

这些表述必须等 GPU、真实模型和可复现实验完成后再使用。

---

## 18. 项目最终能力画像

项目完成 V2 后，预计形成四层能力：

### Layer 1 — Model

~~~text
QLoRA · SFT · Preference · RL · Adapter · Model Evaluation
~~~

### Layer 2 — Agent

~~~text
Tool Calling · Runtime · State · Memory · Checkpoint · Recovery · Environment
~~~

### Layer 3 — Evaluation

~~~text
Benchmark · Failure Taxonomy · Task Success · Recovery · Cost · Reliability
~~~

### Layer 4 — Engineering

~~~text
FastAPI · Model Adapter · Session · Trace · Auth · Observability · Deployment
~~~

最终项目既可以作为大模型开发项目，也可以作为 Agent 开发项目：

- 模型侧回答“模型能不能做？”
- Runtime 回答“模型怎么做？”
- State / Recovery 回答“做错了以后怎么办？”
- Evaluation 回答“怎么证明更可靠？”
- Serving 回答“怎么把能力真正提供给用户？”
