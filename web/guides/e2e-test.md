# 端到端测试

> **在 Claude Code 中使用**: 输入 `/run-e2e-test`，后跟你的测试需求（如指定用例、测试范围），Claude 会自动运行端到端测试并分析结果。

## 概述

端到端测试覆盖完整的评测流程：加载用例 → 初始化 Agent → 虚拟用户与被测系统多轮对话 → 评估器打分 → 输出结果。测试通过即代表框架各组件（TestAgent、TargetAgent、EvalAgent、Orchestrator）协作正常。

## 运行 e2e 测试

### 前置条件

确保 `.env` 中已配置 `OPENAI_API_KEY`，e2e 测试需要调用 LLM。

### 运行全部

```bash
pytest evaluator/tests/test_e2e.py -v -s
```

### 运行单条用例

用 `-k` 过滤用例 ID：

```bash
pytest evaluator/tests/test_e2e.py -v -s -k "manual_headache_001"
pytest evaluator/tests/test_e2e.py -v -s -k "auto_history_llm_004"
```

### 运行批量测试（含进度跟踪）

```bash
pytest evaluator/tests/test_e2e.py -v -s -k "batch"
```

## 测试用例

e2e 测试用例在 `evaluator/tests/fixtures/test_cases.jsonl`，包含 4 条用例：

| ID | 用户模式 | 被测系统 | 评估器 |
|----|----------|----------|--------|
| `manual_headache_001` | manual | theta_api | semantic |
| `auto_cough_child_002` | auto | theta_api | semantic |
| `auto_history_sleep_003` | auto | theta_api | semantic |
| `auto_history_llm_004` | auto | llm_api | semantic |

覆盖了：
- **manual / auto** 两种 TestAgent 模式
- **theta_api / llm_api** 两种 TargetAgent
- 带 history 的预设对话场景

## 测试验证内容

每条用例通过后验证：

1. 评估结果为 `pass` 或 `fail`，分数在 `0.0~1.0` 范围内
2. 有非空的 feedback 评语
3. 执行时间有效（`start <= end`）
4. 带 history 的用例：trace 中保留了历史对话
5. llm_api 目标：有 target 成本记录

批量测试额外验证：
- `TestReport` 聚合指标（pass_count, fail_count, pass_rate, avg_score）
- `BatchSession` 进度跟踪和快照（snapshot）功能

## 运行全部单元测试

```bash
pytest evaluator/tests/ -v
```

> 注意：e2e 测试需要 API Key，未配置时会自动跳过（`skipif OPENAI_API_KEY`）。单元测试中的 mock 测试无需 API Key。
