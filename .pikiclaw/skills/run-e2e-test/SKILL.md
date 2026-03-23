---
name: run-e2e-test
description: 运行端到端测试，验证评测框架各组件协作正常。
---

# Run E2E Test

运行端到端集成测试，验证 TestAgent、TargetAgent、EvalAgent、Orchestrator 整体工作正常。

## Workflow

### Step 1: 检查前置条件

确认 `.env` 中配置了 `OPENAI_API_KEY`，e2e 测试需要调用 LLM：

```bash
grep OPENAI_API_KEY .env
```

如果未配置，提示用户先执行 `/quick-start` 完成环境设置。

### Step 2: 运行测试

根据用户需求选择运行方式：

```bash
# 运行全部 e2e 测试（4 条用例）
pytest evaluator/tests/test_e2e.py -v -s

# 运行单条用例（用 -k 过滤）
pytest evaluator/tests/test_e2e.py -v -s -k "manual_headache_001"
pytest evaluator/tests/test_e2e.py -v -s -k "auto_history_llm_004"

# 运行批量测试（含 BatchSession 进度跟踪验证）
pytest evaluator/tests/test_e2e.py -v -s -k "batch"
```

### Step 3: 解读结果

测试用例覆盖范围：

| ID | 用户模式 | 被测系统 | 评估器 |
|----|----------|----------|--------|
| `manual_headache_001` | manual | theta_api | semantic |
| `auto_cough_child_002` | auto | theta_api | semantic |
| `auto_history_sleep_003` | auto | theta_api | semantic |
| `auto_history_llm_004` | auto | llm_api | semantic |

全部通过 = 框架工作正常。如果有失败：
- 检查 API Key 是否有效
- 检查网络连接（theta_api 需要访问 Theta Health 服务）
- 查看错误日志中的具体原因

### Step 4: 运行全部单元测试（可选）

```bash
pytest evaluator/tests/ -v
```

单元测试使用 mock，不需要 API Key。
