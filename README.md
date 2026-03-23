# HolyEval

Theta Health 虚拟用户评测框架 — 通过合成虚拟用户与被测系统多轮对话，自动评估 AI 医疗助手的表现。

## 快速开始

### 1. 安装依赖

```bash
# 环境要求：Python >= 3.11 + uv
uv sync
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 填入 API Key（至少配一个 LLM 提供商）：

| 变量 | 说明 | 获取方式 |
|------|------|----------|
| `OPENAI_API_KEY` | OpenAI API 密钥 | https://platform.openai.com/api-keys |
| `GOOGLE_API_KEY` | Google AI API 密钥 | https://aistudio.google.com/apikey |
| `THETA_API_BASE_URL` | Theta Health API（可选） | 内部系统 |

### 3. 验证

```bash
python -c "import evaluator; import benchmark; import generator; print('OK')"
```

### 4. 启动 Web UI

```bash
python -m web
# 访问 http://localhost:8000
```

### 5. 运行 Benchmark

```bash
# 语法: python -m benchmark.basic_runner <benchmark> <dataset> --target-type <type> --target-model <model> [options]

# HealthBench — 医疗 AI 综合评测
python -m benchmark.basic_runner healthbench sample --target-type llm_api --target-model gpt-4.1
python -m benchmark.basic_runner healthbench hard --target-type llm_api --target-model gpt-4.1 -p 5

# MedCalc-Bench — 医疗计算评测
python -m benchmark.basic_runner medcalc sample --target-type llm_api --target-model gpt-4.1

# AgentClinic — 多专科临床诊断评测
python -m benchmark.basic_runner agentclinic medqa --target-model gpt-4.1
python -m benchmark.basic_runner agentclinic nejm --target-model gpt-4.1

# MedHall — 医疗幻觉检测
python -m benchmark.basic_runner medhall theta --target-model gpt-4.1

# MemoryArena — Agent 多子任务记忆评测
python -m benchmark.basic_runner memoryarena sample --target-model gpt-4.1

# 更多选项
python -m benchmark.basic_runner healthbench sample --target-type llm_api --target-model gpt-4.1 --limit 10 -p 3 -v
```

| 选项 | 说明 |
|------|------|
| `--target-type` | 被测系统类型（`llm_api` / `theta_api`），**必填** |
| `--target-model` | 模型名称（如 `gpt-4.1`） |
| `--limit N` | 只跑前 N 条用例 |
| `--ids x,y,z` | 指定用例 ID |
| `-p N` | 并发数（默认 1） |
| `-v` | 详细日志 |

## 架构总览

```
┌─ 1. 合成虚拟用户 ──────────────┐  ┌─ 2. 虚拟用户评测框架 ─────────────────┐  ┌─ 3. 封装被测系统 ────────┐
│                                │  │                                       │  │                         │
│  ┌────────────────────────┐    │  │   ┌───────────┐    交互语义    ┌──────────────┐   ┌────────────┐  │
│  │ 用户画像 Static Profile│    │  │   │ TestAgent │ ──操作/反馈──▶│ TargetAgent  │   │GUI-执行引擎│  │
│  │ · 年龄/性别/职业       │    │  │   │ 测试执行   │◀────────────  │ 被测目标代理  │   │点击/输入/截图│ │
│  │ · 病史/慢病/过敏史     │    │  │   │ 代理      │               │              │   └────────────┘  │
│  └────────────────────────┘    │  │   └─────┬─────┘               └──────┬───────┘   ┌────────────┐  │
│  ┌────────────────────────┐    │  │         │ 上报交互过程               │ 上报处理日志│LLM-执行引擎│  │
│  │ 用户数据 Dynamic Data  │───▶│  │         ▼                            ▼            │API调用/多轮 │  │
│  │ · 当前症状/体检指标    │ 合成│  │   ┌──────────────┐                               │对话         │  │
│  │ · 用药记录/睡眠运动    │ 引擎│  │   │  EvalAgent   │                               └────────────┘  │
│  └────────────────────────┘  ──┼──│   │  评估代理     │                                              │
│  ┌────────────────────────┐  ▶ │  │   │  裁判+分析师  │                               支持:           │
│  │ 用户行为 Dynamic Behavior│  │  │   │  评估质量     │                               ChatGPT/Theta  │
│  │ · 饮食/作息/运动       │    │  │   │  量化打分     │                               Gemini/...     │
│  │ · 压力事件             │    │  │   │  生成报告     │                                              │
│  └────────────────────────┘    │  │   └──────────────┘                                              │
└────────────────────────────────┘  └───────────────────────────────────────┘  └─────────────────────────┘
```

### 核心执行流程

```
do_single_test(TestCase) -> TestResult

1. 初始化   — 创建 TestAgent / TargetAgent / EvalAgent
2. 对话循环 — TestAgent ↔ TargetAgent 交替，直到 is_finished 或 max_turns
3. 评估     — EvalAgent.run(memory_list, session_info) -> EvalResult
4. 汇总     — 封装 TestResult 返回
```

所有调用路径（CLI / Web UI / API）统一通过 `evaluator/core/orchestrator.py:do_single_test()` 执行。

## 项目结构

```
holyeval/
├── evaluator/                              # 评测执行引擎
│   ├── core/                               #   框架核心
│   │   ├── schema.py                       #     数据结构定义（Pydantic v2）
│   │   ├── bench_schema.py                 #     Benchmark 数据模型
│   │   ├── orchestrator.py                 #     编排器 do_single_test / BatchSession
│   │   └── interfaces/                     #     三类 Agent 抽象接口
│   ├── plugin/                             #   插件实现（__init_subclass__ 自动注册）
│   │   ├── eval_agent/                     #     评估代理
│   │   ├── target_agent/                   #     被测目标代理
│   │   └── test_agent/                     #     虚拟用户代理
│   ├── utils/                              #   公共工具
│   │   ├── llm.py                          #     底层大模型调用（do_execute）
│   │   ├── benchmark_reader.py             #     读取 benchmark/data/ 目录
│   │   ├── report_reader.py                #     读写 benchmark/report/ 报告
│   │   └── agent_inspector.py              #     反射 plugin registry 提取元数据
│   ├── cli.py                              #   CLI（list 命令）
│   └── tests/                              #   测试
│
├── benchmark/                              # Benchmark 数据与报告
│   ├── basic_runner.py                     #   CLI 跑分执行器
│   ├── data/                               #   评测数据集（按类型分目录）
│   │   ├── healthbench/                    #     HealthBench 医疗 AI 评测
│   │   ├── medcalc/                        #     MedCalc-Bench 医疗计算评测
│   │   ├── agentclinic/                    #     AgentClinic 多专科临床诊断
│   │   ├── medhall/                        #     MedHall 医疗幻觉检测
│   │   └── extraction/                     #     信息提取评测
│   └── report/                             #   评测报告输出
│
├── generator/                              # 数据转换工具
│   ├── healthbench/                        #   HealthBench → HolyEval
│   ├── medcalc/                            #   MedCalc-Bench → HolyEval
│   ├── agentclinic/                        #   AgentClinic → HolyEval
│   └── medhall/                            #   MedHall 幻觉用例生成 + 转换
│
├── web/                                    # Web UI（内部开发工具，FastAPI + Alpine.js）
│   ├── app/                                #   API 服务
│   ├── templates/                          #   前端页面
│   └── guides/                             #   开发指南（Markdown）
│
├── hma-web/                                # HMA 公开评测平台（独立业务应用）
│   └── ...                                 #   DB / Auth / 榜单 / 消息通知（详见 hma-web/README.md）
│
└── pyproject.toml                          # uv workspace 根配置
```

## 插件系统

三类 Agent 通过 `__init_subclass__` 机制自动注册，开发者只需继承基类并指定 `name`：

```python
class CustomEvalAgent(AbstractEvalAgent, name="my_eval"):
    async def run(self, memory_list, session_info=None) -> EvalResult:
        ...
# 查找: AbstractEvalAgent.get("my_eval")
```

### 已注册插件

| Agent 类型 | 名称 | 说明 |
|-----------|------|------|
| **TestAgent** | `auto` | LLM 驱动的虚拟用户，根据 goal/context 模拟自然对话 |
| | `manual` | 脚本驱动，按序发送 strict_inputs，零 LLM 调用 |
| **TargetAgent** | `llm_api` | 通用 LLM API（OpenAI / Gemini / Anthropic） |
| | `theta_api` | Theta Health 产品 HTTP API |
| **EvalAgent** | `semantic` | LLM 语义评估（安全性、共情、专业性） |
| | `healthbench` | HealthBench rubric 多维评分 |
| | `medcalc` | MedCalc-Bench 医疗计算评估（LLM 提取 + 类型化匹配） |
| | `hallucination` | 医疗幻觉检测（事实/上下文/引用三类，LLM-as-Judge） |
| | `indicator` | 健康指标数据比对 |
| | `keyword` | 关键词规则匹配 |
| | `preset_answer` | 标准答案匹配（精确/关键词模式） |
| | `redteam_compliance` | 红队合规评估（LLM-as-Judge 医疗合规性评分） |
| | `memoryarena` | MemoryArena 多子任务评估（LLM-as-Judge + Progress Score） |

## Benchmark 数据集

| 评测套件 | 数据集 | 数量 | 评估器 | 说明 |
|----------|--------|------|--------|------|
| **healthbench** | sample | 100 | healthbench | 按主题等比抽样，快速验证 |
| | hard | 1,000 | healthbench | 高难度子集 |
| | consensus | 3,671 | healthbench | 医生共识子集 |
| | full | 5,000 | healthbench | 完整数据集 |
| **medcalc** | sample | 5 | medcalc | 覆盖多种计算器类型 |
| | full | ~1,047 | medcalc | 完整测试集 |
| **agentclinic** | medqa | 107 | preset_answer | USMLE 多专科 OSCE 临床诊断 |
| | nejm | 15 | preset_answer | NEJM 病例 MCQ 诊断 |
| **medhall** | sample | 30 | hallucination | 三类幻觉（事实/上下文/引用）各 10 条 |
| **extraction** | simple | 9 | preset_answer | 健康数据提取 |
| **aq_redteam** | full | 50 | redteam_compliance | 红队合规评测（5 类诱导性问题） |
| **memoryarena** | sample | 10 | memoryarena | 按领域等比抽样 |
| | full | 701 | memoryarena | 5 领域全量（shopping/travel/search/math/physics） |

## Web UI

启动 `python -m web` 后访问 http://localhost:8000，提供以下功能：

| 页面 | 路由 | 功能 |
|------|------|------|
| 执行评测 | `/tasks` | 选择数据集，配置参数，发起任务，SSE 实时进度 |
| 任务详情 | `/tasks/{id}` | 进度卡片 + 可展开用例列表（对话/反馈/评分） |
| 历史报告 | `/tasks` | 按 benchmark/文件名筛选，按时间倒序 |
| 报告详情 | `/reports/{benchmark}/{filename}` | 完整报告查看 |
| 数据集浏览 | `/benchmarks` | 按目录浏览 benchmark 数据集 |
| Agent 注册表 | `/agents/*` | 查看已注册插件元信息 |
| 开发指南 | `/guides/*` | Markdown 渲染的开发文档 |

## 核心数据结构

所有数据模型基于 Pydantic v2，定义在 `evaluator/core/schema.py`：

| 结构体 | 说明 |
|--------|------|
| `UserInfo` | 虚拟用户配置：`auto`（LLM 驱动）或 `manual`（脚本驱动） |
| `TargetInfo` | 被测目标配置（Discriminated Union）：`llm_api` / `theta_api` |
| `EvalInfo` | 评估配置（Discriminated Union）：`semantic` / `healthbench` / `medcalc` / `hallucination` / `indicator` / `keyword` / `preset_answer` |
| `TestCase` | 测试用例聚合根（user + target + eval + history + 元数据） |
| `EvalResult` | 评估结果（pass/fail、score、feedback、trace） |
| `TestResult` | 单测试执行结果 |
| `BenchItem` | Benchmark 数据项（user + eval + 可选 history，运行时转为 TestCase） |
| `BenchReport` | Benchmark 报告（聚合统计 + 单条结果） |

## 开发指南

### Claude Code Skills

| 命令 | 说明 |
|------|------|
| `/quick-start` | 项目初始化引导 |
| `/add-benchmark` | 集成外部 benchmark（研究 → 转换 → 验证全流程） |
| `/add-eval-agent` | 新增评测逻辑插件 |
| `/add-target-agent` | 新增被测系统插件 |
| `/run-benchmark` | 运行 benchmark 跑分 |
| `/run-e2e-test` | 运行端到端测试 |

### 测试

```bash
pytest evaluator/tests/              # 全部测试
pytest evaluator/tests/test_e2e.py   # 端到端测试（需要 API Key）
ruff check .                         # 代码检查
ruff format .                        # 格式化
```

## 依赖关系

```
evaluator  ←──  benchmark（调用 evaluator 执行评测、引用数据结构）
           ←──  generator（引用 evaluator 数据结构）
           ←──  web（调用 evaluator 服务层、管理任务生命周期）
           ←──  hma-web（单向依赖，调用 evaluator 执行评测）
```

**框架层**（evaluator / benchmark / generator / web）通过根目录 uv workspace 统一管理，遵循 GitOps 原则，无外部有状态依赖。

**业务应用层**（hma-web）独立管理，包含数据库、鉴权、消息通知等第三方依赖。hma-web 单向依赖框架层，框架层对 hma-web **零感知**。详见 [hma-web/README.md](hma-web/README.md)。
