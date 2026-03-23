# Benchmark Scoring

> **Using in Claude Code**: Type `/run-benchmark`, followed by your benchmarking requirements (e.g., `healthbench sample --target-model gpt-4.1`, specify concurrency). Claude will automatically configure and execute the benchmark run.

## Overview

Benchmark scoring batch-executes evaluations for all cases in a dataset, generating structured reports containing pass rates, score distributions, and cost statistics. Both CLI and Web UI methods are supported.

## CLI Benchmarking

### Basic Usage

```bash
# Syntax: python -m benchmark.basic_runner <benchmark> <dataset> --target-type <type> --target-model <model> [options]

# HealthBench
python -m benchmark.basic_runner healthbench sample --target-type llm_api --target-model gpt-4.1
python -m benchmark.basic_runner healthbench hard --target-type llm_api --target-model gpt-4.1 -p 5

# MedCalc-Bench
python -m benchmark.basic_runner medcalc sample --target-type llm_api --target-model gpt-4.1
python -m benchmark.basic_runner medcalc full --target-type llm_api --target-model gpt-4.1 --limit 50

# AgentClinic — Multi-specialty clinical diagnosis
python -m benchmark.basic_runner agentclinic medqa --target-model gpt-4.1
python -m benchmark.basic_runner agentclinic nejm --target-model gpt-4.1

# MedHall — Medical hallucination detection (default theta_api, email read from case target_overrides)
python -m benchmark.basic_runner medhall sample
python -m benchmark.basic_runner medhall sample --target-model general   # Switch Agent type

# MedHall — Compare with LLM (requires changing target.type to llm_api in metadata.json)
# python -m benchmark.basic_runner medhall sample --target-model gpt-4.1

# Extraction (target is locked by metadata, no need to specify)
python -m benchmark.basic_runner extraction simple

# MemoryArena — Agent multi-subtask memory evaluation
python -m benchmark.basic_runner memoryarena sample --target-model gpt-4.1
```

### Options

| Option | Description |
|------|------|
| `--target-type` | System-under-test type (`llm_api` / `theta_api`), **required for most benchmarks** |
| `--target-model` | Model name (e.g., `gpt-4.1`, `gemini-3-pro`) |
| `--limit N` | Run only the first N cases |
| `--ids x,y,z` | Specify case IDs |
| `-p N` | Concurrency level (default 1) |
| `-v` | Verbose logging |

### Combined Examples

```bash
# Specify IDs + verbose logging
python -m benchmark.basic_runner healthbench sample --target-type llm_api --target-model gpt-4.1 --ids hb_abc,hb_def -v

# Limit count + concurrency
python -m benchmark.basic_runner medcalc full --target-type llm_api --target-model gpt-4.1 --limit 10 -p 3 -v
```

## Web UI Benchmarking

1. Start the web server: `python -m web`
2. Navigate to http://localhost:8000
3. On the **Run Evaluations** page, select a benchmark and dataset
4. Configure target parameters (type, model) and concurrency
5. Click to create the task; progress is pushed in real time via SSE

The Web UI provides:
- Real-time progress tracking (SSE push)
- Per-case conversation process viewing
- Task cancellation
- Automatic report generation upon completion

## Reports

### Report Location

```
benchmark/report/<benchmark>/<dataset>_<target>_<timestamp>.json
```

Example: `benchmark/report/healthbench/sample_gpt-4.1_20260213_183356.json`

### Report Contents

- Overall pass rate and average score
- Per-case pass/fail, score, and feedback
- Conversation turn statistics
- Cost statistics (test / eval / target)

### Viewing Reports

- **CLI**: Reports are printed to the terminal after benchmarking completes
- **Web UI**: View in the "Historical Reports" list at the bottom of the Run Evaluations page; supports filtering by benchmark and filename

## Available Datasets

| Benchmark Suite | Dataset | Case Count | Evaluator | Description |
|----------|--------|--------|--------|------|
| healthbench | sample | 100 | healthbench | Proportionally sampled by topic, quick validation |
| healthbench | hard | 1,000 | healthbench | High-difficulty subset |
| healthbench | consensus | 3,671 | healthbench | Physician consensus subset |
| healthbench | full | 5,000 | healthbench | Full dataset |
| medcalc | sample | 5 | medcalc | Covers multiple calculator types |
| medcalc | full | ~1,047 | medcalc | Full test set |
| agentclinic | medqa | 107 | preset_answer | USMLE multi-specialty OSCE clinical diagnosis |
| agentclinic | nejm | 15 | preset_answer | NEJM case MCQ diagnosis |
| medhall | sample | 30 | hallucination | Medical hallucination detection (factual/contextual/citation), default theta_api |
| extraction | simple | 9 | preset_answer | Health data extraction |
| memoryarena | sample | 10 | memoryarena | Proportionally sampled by domain |
| memoryarena | full | 701 | memoryarena | All 5 domains (shopping/travel/search/math/physics) |
