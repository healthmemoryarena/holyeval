---
name: run-benchmark
description: Run all benchmark test cases or a filtered subset.
argument-hint: "<benchmark> <dataset> [options]"
---

# Run Benchmark

运行 benchmark 跑分，对数据集中的用例批量执行评测并生成报告。

## Usage

当用户要求运行 benchmark 跑分时使用此 skill。

## Commands

```bash
# 语法: python -m benchmark.basic_runner <benchmark> <dataset> --target-type <type> --target-model <model> [options]

# HealthBench
python -m benchmark.basic_runner healthbench sample --target-type llm_api --target-model gpt-4.1
python -m benchmark.basic_runner healthbench hard --target-type llm_api --target-model gpt-4.1 -p 5

# MedCalc-Bench
python -m benchmark.basic_runner medcalc sample --target-type llm_api --target-model gpt-4.1
python -m benchmark.basic_runner medcalc full --target-type llm_api --target-model gpt-4.1 --limit 50

# Extraction（target 由 metadata 锁定）
python -m benchmark.basic_runner extraction simple

# 组合选项
python -m benchmark.basic_runner healthbench sample --target-type llm_api --target-model gpt-4.1 --limit 10 -p 3 -v
```

## Options

| 选项 | 说明 |
|------|------|
| `--target-type` | 被测系统类型（`llm_api` / `theta_api`），**必填** |
| `--target-model` | 模型名称（如 `gpt-4.1`、`gemini-3-pro`） |
| `--limit N` | 只跑前 N 条用例 |
| `--ids x,y,z` | 指定用例 ID |
| `-p N` | 并发数（默认 1） |
| `-v` | 详细日志 |

## Directories

- Benchmark 数据集: `benchmark/data/<benchmark>/<dataset>.jsonl`
- 评测报告: `benchmark/report/<benchmark>/<dataset>_<target>_<timestamp>.json`

## Available Datasets

| 评测套件 | 数据集 | 数量 | 评估器 |
|----------|--------|------|--------|
| healthbench | sample / hard / consensus / full | 100 ~ 5,000 | healthbench |
| medcalc | sample / full | 5 ~ 1,047 | medcalc |
| extraction | simple | 9 | preset_answer |

## Web UI Alternative

也可通过 Web UI 执行跑分，支持实时进度查看：

```bash
python -m web
# 访问 http://localhost:8000/tasks 创建任务
```

## Output

命令输出：
1. 每条用例的执行进度
2. 汇总统计（通过率、平均分、成本）
3. 报告文件保存到 `benchmark/report/` 目录
