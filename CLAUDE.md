# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HolyEval is an open-source virtual user evaluation framework for AI medical assistants. It synthesizes virtual users to have multi-turn conversations with the system under test, then automatically evaluates performance via pluggable evaluators.

## Commands

```bash
# Install dependencies (uses uv workspace)
uv sync

# Run benchmarks (target type 由 metadata.json 决定，--target-model 覆盖 editable 字段)
python -m benchmark.basic_runner healthbench sample --target-model gpt-4.1              # 100 条快速跑
python -m benchmark.basic_runner healthbench full --target-model gpt-4.1 --limit 50     # 全量取前 50
python -m benchmark.basic_runner healthbench hard --target-model gemini-3-pro -p 5     # 困难子集，5 并发
python -m benchmark.basic_runner medcalc sample --target-model gpt-4.1                  # MedCalc 快速跑
python -m benchmark.basic_runner agentclinic medqa --target-model gpt-4.1              # AgentClinic 临床诊断
python -m benchmark.basic_runner memoryarena sample --target-model gpt-4.1              # MemoryArena 快速跑
python -m benchmark.basic_runner healthbench sample --target-model gpt-4.1 --ids hb_abc  # 指定 ID
python -m benchmark.basic_runner healthbench sample --target-model gpt-4.1 --limit 10 -p 3 -v  # 组合选项
python -m benchmark.basic_runner healthbench sample --resume                            # 断点续跑（恢复上次中断的评测）

# Data conversion (外部数据集 → HolyEval)
python -m generator.healthbench.converter input.jsonl output.jsonl --target-model gpt-4.1
python -m generator.medcalc.converter                                                       # MedCalc CSV → JSONL
python -m generator.agentclinic.converter <input.jsonl> <output.jsonl>                      # AgentClinic → BenchItem
python -m generator.medhall.data_gen --count 15 --output generator/medhall/raw_data.jsonl  # 生成幻觉用例
python -m generator.medhall.converter generator/medhall/raw_data.jsonl benchmark/data/medhall/theta.jsonl  # 转换
python -m generator.memoryarena.converter                                                   # MemoryArena HuggingFace → JSONL

# Web UI
python -m web                              # uvicorn :8000, auto-reload

# Tests
pytest evaluator/tests/
pytest evaluator/tests/test_e2e.py          # single test file

# Lint
ruff check .
ruff format .
```

## Architecture

### Execution Flow

```
TestCase (JSON) → Orchestrator (do_single_test)
  1. Initialize agents from TestCase config via plugin registry
  2. Dialogue loop: TestAgent ↔ TargetAgent (until is_finished or max_turns)
  3. EvalAgent.run(memory_list, session_info) → EvalResult
  4. Return TestResult (score, pass/fail, feedback, cost)
```

All call paths (CLI, batch, API) funnel through `evaluator/core/orchestrator.py:do_single_test()`.

### Batch Execution & Observability

`BatchSession` 提供可观测、可取消的批量执行能力，适用于 server 场景：

```python
session = BatchSession(cases, max_concurrency=5, on_progress=callback)
report = await session.run()       # 执行并返回 TestReport
session.snapshot()                  # JSON 可序列化的进度快照
session.cancel()                    # 协作式取消

# 实时观测单个 case 的对话过程
ctx = session.contexts["case_id"]   # CaseContext
ctx.status                          # CaseStatus 枚举 (pending/init/dialogue/eval/completed/error/cancelled)
ctx.test_agent.memory_list          # 实时对话数据
ctx.turn                            # 当前轮次
```

`do_batch_test()` 为向后兼容的薄包装，内部委托给 `BatchSession`。

### Plugin System

Three agent types use `__init_subclass__` registration. Plugins register by inheriting with a `name` keyword:

```python
class CustomTestAgent(AbstractTestAgent, name="custom"):
    ...
# Lookup: AbstractTestAgent.get("custom")
```

Plugins are activated by import (in `evaluator/plugin/`), which triggers `__init_subclass__` registration. The `core/` layer depends only on abstract interfaces, never on concrete plugins.

| Agent Type | Interface | Built-in Implementations |
|---|---|---|
| **TestAgent** (virtual user) | `core/interfaces/abstract_test_agent.py` | `auto` — LLM-driven user simulation, `manual` — scripted sequential inputs |
| **TargetAgent** (system under test) | `core/interfaces/abstract_target_agent.py` | `llm_api` — generic LLM API (OpenAI/Gemini) |
| **EvalAgent** (evaluator) | `core/interfaces/abstract_eval_agent.py` | `semantic` (LLM-based), `keyword` (rule-based), `preset_answer` (exact/keyword matching), `healthbench` (rubric-based grading), `medcalc` (medical calculation), `hallucination` (LLM-as-Judge), `redteam_compliance` (red-team compliance), `memoryarena` (multi-subtask LLM judge) |

You can add custom agent plugins by inheriting from the abstract base classes. See `/add-eval-agent`, `/add-target-agent` skills for guided scaffolding.

#### Plugin 元数据约定

每个 plugin 类可声明以下类属性，`agent_inspector` 自动发现并暴露给 CLI / Web UI：

| 属性 | 适用 Agent | 说明 |
|---|---|---|
| `_display_meta` | 全部 | 展示元数据: `icon` (SVG path), `color` (CSS), `features` (标签列表) |
| `_cost_meta` | EvalAgent | 费用预估: `{"est_cost_per_case": float}` (USD/case) |
| `_cost_meta` | TargetAgent | Token 预估: `{"est_input_tokens": int, "est_output_tokens": int}` (单次调用) |
| `_config_model` | TestAgent | 配置模型类名: `"AutoUserInfo"` (schema.py 中的 Pydantic 模型名) |

### Key Modules

- **`evaluator/core/schema.py`** — All data structures (Pydantic v2): TestCase, UserInfo, TargetInfo, EvalInfo, TestResult, SessionInfo
- **`evaluator/core/orchestrator.py`** — `do_single_test()`, `do_batch_test()`, `BatchSession`（进度跟踪/取消/实时观测）, `CaseContext`, `CaseStatus`
- **`evaluator/utils/llm.py`** — Unified LLM interface `do_execute()` wrapping langchain's `create_agent`. Supports OpenAI and Google Gemini with optional thinking/reasoning
- **`evaluator/core/bench_schema.py`** — Benchmark 数据模型（BenchItem, BenchMark, BenchReport）+ 转换函数
- **`evaluator/utils/benchmark_reader.py`** — 读取 + 加载 benchmark/data/ 目录（CLI + Web 共享）
- **`evaluator/utils/report_reader.py`** — 读取 + 写入 benchmark/report/ 报告（CLI + Web 共享）
- **`evaluator/utils/agent_inspector.py`** — 反射 plugin registry 提取 agent 元数据（CLI + Web 共享）
- **`evaluator/utils/checkpoint.py`** — 检查点管理器（断点续跑）

### Workspace Structure

uv workspace monorepo with four members:
- **`evaluator/`** — Core evaluation engine (main package)
- **`benchmark/`** — Benchmark runner (basic_runner.py) + data/report 目录
- **`generator/`** — Test case generator and data conversion tools
- **`web/`** — Web UI for managing and visualizing evaluations

### Benchmark Data

按评测类型分目录管理，CLI 使用 `<benchmark> <dataset>` 两个位置参数：

```
benchmark/
├── data/
│   ├── healthbench/          # HealthBench 医疗 AI 评测
│   ├── medcalc/              # MedCalc-Bench 医疗计算评测
│   ├── agentclinic/          # AgentClinic 多专科临床诊断
│   ├── medhall/              # MedHall 医疗幻觉检测
│   ├── memoryarena/          # MemoryArena Agent 记忆评测
│   └── history_demo/         # History $ref 引用演示
├── report/                   # 报告输出（镜像 data/ 子目录结构）
└── basic_runner.py           # 跑分执行器
```

每个 benchmark 目录包含: `<dataset>.jsonl`（数据文件）+ `metadata.json`（套件元数据，Web UI 展示用）。报告文件名格式: `{dataset}_{target_label}_{YYYYMMDD_HHmmss}.json`，其中 `target_label = {type}[_{model}][_{agent}][_k{top_k}]`。

metadata.json 的 `target` 为 **TargetSpec 数组**，支持一个 benchmark 定义多个被测系统类型；`params` 为可选的共享参数字典，供 JSONL 条目通过 `$ref` 引用：

```json
{
  "description": "...",
  "target": [
    { "type": "llm_api", "fields": { "model": {"default": "gpt-4.1", "editable": true, "required": true} } }
  ],
  "params": {
    "shared_history": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
  }
}
```

- 单 target → CLI/Web UI 自动使用，type 锁定
- 多 target → CLI 通过 `--target-type` 指定，Web UI 显示类型选择器
- 每个 TargetSpec 的 `fields` 定义字段默认值、是否可编辑（`editable`）、是否必填（`required`）
- `params`（可选）: 共享数据字典。JSONL 中某字段值为 `{"$ref": "key"}` 时，加载时自动替换为 `params[key]`。仅扫描顶层字段

### Test Cases

JSONL files in `benchmark/data/<benchmark>/<dataset>.jsonl` (benchmark) and `evaluator/tests/fixtures/cases/` (dev). Each case specifies user config (goal, context, strict_inputs, max_turns), target config, eval config (evaluator type + threshold), and optional `history` (pre-evaluation conversation as `[{role, content}]` dicts, auto-converted to langchain `BaseMessage`).

Key fields:
- **`strict_inputs`** (`List[str]`): manual 模式逐条按序发送。单条 = 单轮问答；多条 = 逐轮向被测系统注入上下文再提问（如先发病历再问诊断），中间回复不影响评估
- **`history`** (`List[Dict]`): 评测前的预加载对话上下文（不经过对话循环），与 `strict_inputs` 可组合使用

### Data Conversion (generator/)

`generator/` 包含数据集转换工具，将外部评测数据集转为 HolyEval BenchItem 格式：

- **`generator/healthbench/converter.py`** — HealthBench JSONL → HolyEval BenchItem JSONL
- **`generator/medcalc/converter.py`** — MedCalc-Bench CSV → HolyEval BenchItem JSONL
- **`generator/agentclinic/converter.py`** — AgentClinic JSONL → HolyEval BenchItem JSONL
- **`generator/medhall/converter.py`** — MedHall 原始 JSONL → HolyEval BenchItem JSONL
- **`generator/memoryarena/converter.py`** — MemoryArena HuggingFace → HolyEval BenchItem JSONL

### Web UI

`web/` 提供可视化管理界面，基于 FastAPI + Jinja2 + htmx + Alpine.js + Tailwind CSS (CDN)。

```bash
python -m web                  # uvicorn :8000, auto-reload
```

| 模块 | 路由 | 功能 |
|------|------|------|
| 执行评测 | `/tasks` | 选择 benchmark/dataset，配置参数，发起任务，实时 SSE 进度 |
| 任务详情 | `/tasks/{task_id}` | 进度卡片 + 可展开用例列表（对话/反馈/agent 标签） |
| 报告查看 | `/reports/{benchmark}/{filename}` | 独立报告页（与任务详情相同 UI） |
| 指标数据 | `/benchmarks` | 按目录浏览 benchmark 数据集 |
| Agent 注册表 | `/agents/target`, `/agents/eval`, `/agents/test` | 反射插件 registry 展示元信息 |

## Environment Variables

关键环境变量（配置在 `.env` 文件中，从 `.env.example` 复制）：

| 变量 | 必填 | 说明 |
|------|------|------|
| `OPENAI_API_KEY` | 至少配一个 LLM | OpenAI API 密钥（gpt-4.1 等） |
| `GOOGLE_API_KEY` | 至少配一个 LLM | Google Gemini API 密钥 |
| `OPENROUTER_API_KEY` | 可选 | OpenRouter 统一多提供商访问 |
| `HOLYEVAL_PORT` | 可选 | Web UI 端口（默认 8000） |

## Code Style

- Python 3.11+, async/await throughout
- Ruff for linting/formatting, line-length 120
- Pydantic v2 for all data models
- Source language: Chinese comments and documentation
