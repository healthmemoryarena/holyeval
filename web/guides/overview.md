<!-- 项目概览 - HolyEval -->

## 项目简介

HolyEval 是一个**虚拟用户评测框架**，专为 AI 医疗助手（Theta Health）设计。通过自动化的多轮对话测试和多维度评估，系统性评估 AI 助手的性能表现。

## 架构概览

![HolyEval 架构图](/static/images/architecture.png)

HolyEval 采用模块化设计，如上图所示包含六大核心组件。其中组件 2-6 分别对应项目的三个核心目录：

- **evaluator** — 对应图中 ②③：虚拟用户评测框架 + 对接服务系统
- **generator** — 对应图中 ④⑥：Benchmark Generator + Benchmark Wrapper
- **benchmark** — 对应图中 ⑤：Benchmark Scheduler（批量调度中心）

### ② 虚拟用户评测框架（evaluator/）

提供核心评测能力，基于三层 Agent 架构：

| Agent 类型 | 业务职责 | 能力说明 |
|-----------|---------|---------|
| **TestAgent** | 模拟虚拟用户行为 | 支持 LLM 驱动的自然对话和脚本驱动的精确复现 |
| **TargetAgent** | 对接被测系统 | 统一封装真实系统 API 和模拟 LLM 调用 |
| **EvalAgent** | 评估对话质量 | 提供语义、指标、规则、标准答案等多维度评估方式 |

**核心评测流程**：
1. TestAgent 发起对话（基于用户目标和上下文）
2. TargetAgent 调用被测系统获取响应
3. 重复对话直至达成目标或超过轮次上限
4. EvalAgent 基于完整对话历史进行多维度评估
5. 生成 TestResult（分数、通过率、详细反馈、成本统计）

### ③ 对接服务系统（evaluator/plugin/target_agent/）

支持两种执行模式对接不同类型的被测系统：

- **真实系统对接**：通过 Theta Health HTTP API 与生产环境交互，验证真实表现
- **模拟系统对接**：直接调用 LLM API 模拟系统行为，快速验证评测逻辑

### ④ Benchmark Generator（generator/）

**业务能力**：将外部评测数据集转换为 HolyEval 标准格式

- **数据转换**：支持 HealthBench、MedCalc-Bench、MemoryArena 等主流 AI 评测数据集
- **业务抽象**：将原始评测场景（prompt + rubrics）映射为 BenchItem 标准格式，支持多轮对话上下文（`history` 字段）
- **灵活配置**：支持自定义评估标准、对话历史（history）、虚拟用户配置

### ⑤ Benchmark Scheduler（benchmark/）

**业务能力**：批量评测的调度执行和进度管理

- **批量调度**：支持数千条用例的并发执行（可配置并发数）
- **实时监控**：跟踪每个用例的执行状态（待执行/对话中/评估中/已完成/已取消）
- **灵活控制**：支持随时取消任务、导出进度快照
- **数据组织**：按评测类型（healthbench/medcalc/extraction）分目录管理测试数据

### ⑥ Benchmark Wrapper（generator/ 共享）

**业务能力**：评测结果的标准化封装和持久化

- **结果聚合**：将单条用例结果聚合为完整评测报告（总分、通过率、统计信息）
- **元数据记录**：跟踪每条用例使用的 Agent 类型，便于后续分析
- **报告管理**：自动保存报告到 `benchmark/report/` 目录，支持历史报告查询

## 目录结构与职责分工

```
holyeval/
├── evaluator/              # ②③ 虚拟用户评测框架 + 对接服务
│   ├── core/              # 核心评测引擎（编排器、数据模型）
│   ├── plugin/            # 三层 Agent 插件实现
│   └── utils/             # 共享工具（LLM 调用、数据读取）
│
├── generator/             # ④⑥ 数据生成 + 结果包装
│   ├── healthbench/       # HealthBench 数据转换器
│   ├── medcalc/           # MedCalc-Bench 数据转换器
│   ├── agentclinic/       # AgentClinic 数据转换器
│   ├── medhall/           # MedHall 幻觉用例生成 + 转换
│   └── memoryarena/       # MemoryArena 数据转换器
│
├── benchmark/             # ⑤ 批量调度中心
│   ├── data/              # 评测数据集（按类型分目录）
│   ├── report/            # 评测报告（自动生成）
│   └── basic_runner.py    # CLI 执行器
│
└── web/                   # Web 管理界面
    ├── app/               # API 服务（任务调度、报告查询）
    ├── templates/         # 前端页面
    └── guides/            # 开发指南
```

## 核心评估能力

HolyEval 提供六种评估方式，覆盖不同评测场景：

| 评估器 | 适用场景 | 评估方式 |
|--------|---------|---------|
| **semantic** | 通用语义质量 | 基于 LLM 的主观质量评价（安全性、共情能力、专业性） |
| **healthbench** | HealthBench 标准 | 基于 rubric 的多维度评分（并发评估所有评分项） |
| **medcalc** | 医疗计算 | LLM 答案提取 + 类型化数值匹配（decimal/integer/date/weeks_days） |
| **hallucination** | 医疗幻觉检测 | LLM-as-Judge 检测事实/上下文/引用三类幻觉，评分 0~1 |
| **indicator** | 健康数据准确性 | 验证系统返回的健康指标（运动、睡眠）是否正确 |
| **keyword** | 关键词规则 | 快速验证响应是否包含必要关键词 |
| **preset_answer** | 标准答案匹配 | 精确匹配或关键词匹配预设答案 |
| **redteam_compliance** | 红队合规测试 | LLM-as-Judge 评估医疗 AI 回复合规性（诱导性问题防御） |
| **memoryarena** | Agent 记忆评测 | LLM-as-Judge 逐子任务判定 + Progress Score（5 领域） |

## 执行模式

支持三种执行方式，适应不同使用场景：

- **CLI 执行**：命令行批量跑分，适合 CI/CD 集成和自动化测试
- **Web UI 执行**：可视化界面，实时查看进度和结果，适合手动验证和分析
- **API 执行**：HTTP 接口集成，适合嵌入其他系统或服务

## 下一步

| 操作 | 说明 |
|------|------|
| [执行评测](/tasks) | 选择数据集，配置参数，实时查看进度 |
| [查看报告](/tasks) | 浏览历史评测结果 |
| [Benchmark 数据](/benchmarks) | 按目录浏览数据集 |
| [Agent 注册表](/agents/test) | 查看已注册插件 |

**Skills 快捷命令**：

- `/add-benchmark` — 集成外部 benchmark（研究 → 转换 → 验证全流程）
- `/add-eval-agent` — 脚手架：新增评测逻辑插件
- `/add-target-agent` — 脚手架：新增被测系统插件
- `/run-e2e-test` — 运行端到端测试
- `/run-benchmark` — 运行 benchmark 跑分
