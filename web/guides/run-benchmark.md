# Benchmark 跑分

> **在 Claude Code 中使用**: 输入 `/run-benchmark`，后跟你的跑分需求（如 `healthbench sample --target-model gpt-4.1`、指定并发数），Claude 会自动配置并执行跑分。

## 概述

Benchmark 跑分对数据集中的所有用例批量执行评测，生成包含通过率、分数分布、成本统计的结构化报告。支持 CLI 和 Web UI 两种方式。

## CLI 跑分

### 基本用法

```bash
# 语法: python -m benchmark.basic_runner <benchmark> <dataset> --target-type <type> --target-model <model> [options]

# HealthBench
python -m benchmark.basic_runner healthbench sample --target-type llm_api --target-model gpt-4.1
python -m benchmark.basic_runner healthbench hard --target-type llm_api --target-model gpt-4.1 -p 5

# MedCalc-Bench
python -m benchmark.basic_runner medcalc sample --target-type llm_api --target-model gpt-4.1
python -m benchmark.basic_runner medcalc full --target-type llm_api --target-model gpt-4.1 --limit 50

# AgentClinic — 多专科临床诊断
python -m benchmark.basic_runner agentclinic medqa --target-model gpt-4.1
python -m benchmark.basic_runner agentclinic nejm --target-model gpt-4.1

# MedHall — 医疗幻觉检测（默认 theta_api，email 从用例 target_overrides 读取）
python -m benchmark.basic_runner medhall sample
python -m benchmark.basic_runner medhall sample --target-model general   # 切换 Agent 类型

# MedHall — 对比 LLM（需将 metadata.json 中 target.type 改为 llm_api）
# python -m benchmark.basic_runner medhall sample --target-model gpt-4.1

# Extraction（target 由 metadata 锁定，不需要指定）
python -m benchmark.basic_runner extraction simple

# MemoryArena — Agent 多子任务记忆评测
python -m benchmark.basic_runner memoryarena sample --target-model gpt-4.1
```

### 选项

| 选项 | 说明 |
|------|------|
| `--target-type` | 被测系统类型（`llm_api` / `theta_api`），**多数 benchmark 必填** |
| `--target-model` | 模型名称（如 `gpt-4.1`、`gemini-3-pro`） |
| `--limit N` | 只跑前 N 条用例 |
| `--ids x,y,z` | 指定用例 ID |
| `-p N` | 并发数（默认 1） |
| `-v` | 详细日志 |

### 组合示例

```bash
# 指定 ID + 详细日志
python -m benchmark.basic_runner healthbench sample --target-type llm_api --target-model gpt-4.1 --ids hb_abc,hb_def -v

# 限制数量 + 并发
python -m benchmark.basic_runner medcalc full --target-type llm_api --target-model gpt-4.1 --limit 10 -p 3 -v
```

## Web UI 跑分

1. 启动 Web 服务: `python -m web`
2. 访问 http://localhost:8000
3. 在 **执行评测** 页面选择 benchmark 和 dataset
4. 配置 target 参数（类型、模型）和并发数
5. 点击创建任务，SSE 实时推送进度

Web UI 提供：
- 实时进度跟踪（SSE 推送）
- 单个用例对话过程查看
- 任务取消功能
- 完成后自动生成报告

## 报告

### 报告位置

```
benchmark/report/<benchmark>/<dataset>_<target>_<timestamp>.json
```

示例: `benchmark/report/healthbench/sample_gpt-4.1_20260213_183356.json`

### 报告内容

- 总通过率和平均分
- 每个用例的 pass/fail、score、feedback
- 对话轮次统计
- 成本统计（test / eval / target）

### 查看报告

- **CLI**: 报告在跑分结束后输出到终端
- **Web UI**: 在执行评测页面底部的「历史报告」列表中查看，支持按 benchmark 和文件名筛选

## 可用数据集

| 评测套件 | 数据集 | 用例数 | 评估器 | 说明 |
|----------|--------|--------|--------|------|
| healthbench | sample | 100 | healthbench | 按主题等比抽样，快速验证 |
| healthbench | hard | 1,000 | healthbench | 高难度子集 |
| healthbench | consensus | 3,671 | healthbench | 医生共识子集 |
| healthbench | full | 5,000 | healthbench | 完整数据集 |
| medcalc | sample | 5 | medcalc | 覆盖多种计算器类型 |
| medcalc | full | ~1,047 | medcalc | 完整测试集 |
| agentclinic | medqa | 107 | preset_answer | USMLE 多专科 OSCE 临床诊断 |
| agentclinic | nejm | 15 | preset_answer | NEJM 病例 MCQ 诊断 |
| medhall | sample | 30 | hallucination | 医疗幻觉检测（事实/上下文/引用），默认 theta_api |
| extraction | simple | 9 | preset_answer | 健康数据提取 |
| memoryarena | sample | 10 | memoryarena | 按领域等比抽样 |
| memoryarena | full | 701 | memoryarena | 5 领域全量（shopping/travel/search/math/physics） |
