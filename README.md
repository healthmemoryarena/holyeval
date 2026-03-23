# HolyEval

**Reproduce any LLM benchmark with one command. Build your own with pluggable agents.**

HolyEval is an open-source evaluation framework for large language models. Drop in a benchmark dataset, run one command, get a scored report. Extend it with custom evaluators, target systems, and virtual users via a pluggable agent architecture.

```bash
# Reproduce HealthBench (medical AI, 100 cases) — one command
python -m benchmark.basic_runner healthbench sample --target-model gpt-4.1

# MedCalc-Bench (medical calculations)
python -m benchmark.basic_runner medcalc sample --target-model gpt-4.1

# MemoryArena (agent memory, 701 cases)
python -m benchmark.basic_runner memoryarena full --target-model gemini-3-pro
```

Each benchmark is a published paper you can reproduce in one line.

## Why HolyEval?

| | |
|---|---|
| **One-command reproduction** | Integrate a published benchmark once, reproduce it forever with a single CLI command |
| **Pluggable architecture** | Three agent types (TestAgent, TargetAgent, EvalAgent) — extend any of them with a single class |
| **Multi-turn dialogue** | Simulates real user conversations, not just single-turn Q&A |
| **Batch execution** | Concurrent runs with real-time progress, cancellation, and checkpoint resume |
| **Web UI** | Visual dashboard for running evaluations, viewing reports, and browsing datasets |

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- At least one LLM API key (OpenAI or Google Gemini)

### Install

```bash
git clone https://github.com/anthropics/holyeval.git
cd holyeval
uv sync

cp .env.example .env
# Edit .env — add your OPENAI_API_KEY or GOOGLE_API_KEY
```

### Run your first benchmark

```bash
# Quick smoke test: 10 HealthBench cases, 3 concurrent
python -m benchmark.basic_runner healthbench sample --target-model gpt-4.1 --limit 10 -p 3

# Full run
python -m benchmark.basic_runner healthbench sample --target-model gpt-4.1

# Resume if interrupted
python -m benchmark.basic_runner healthbench sample --resume
```

### Launch Web UI

```bash
python -m web    # http://localhost:8000
```

## Architecture

```
TestCase (JSON) → Orchestrator
  1. Initialize agents from config via plugin registry
  2. Dialogue loop: TestAgent ↔ TargetAgent (until finished or max turns)
  3. EvalAgent.run(conversation, session) → EvalResult
  4. Return TestResult (score, pass/fail, feedback, cost)
```

All execution paths (CLI, Web UI, programmatic) funnel through a single entry point: `do_single_test()`.

### Plugin System

Three agent types, each extensible via `__init_subclass__` auto-registration:

```python
# Define a custom evaluator — that's it, it's registered
class MyEvalAgent(AbstractEvalAgent, name="my_eval"):
    async def run(self, memory_list, session_info):
        # your evaluation logic
        return EvalResult(score=0.95, passed=True, feedback="...")
```

| Agent Type | Role | Built-in Plugins |
|---|---|---|
| **TestAgent** | Virtual user | `auto` (LLM-driven), `manual` (scripted) |
| **TargetAgent** | System under test | `llm_api` (OpenAI / Gemini) |
| **EvalAgent** | Evaluator | `semantic`, `keyword`, `preset_answer`, `healthbench`, `medcalc`, `hallucination`, `redteam_compliance`, `memoryarena` |

### Project Structure

```
holyeval/
├── evaluator/          # Core engine: schema, orchestrator, plugin interfaces
├── benchmark/          # Runner + datasets (JSONL) + reports
├── generator/          # Dataset converters (paper → HolyEval format)
└── web/                # Web UI (FastAPI + htmx)
```

## Benchmarks

| Benchmark | Paper / Source | Datasets | What it evaluates |
|---|---|---|---|
| **HealthBench** | OpenAI HealthBench | `sample` (100), `full`, `hard`, `consensus` | Medical AI quality |
| **MedCalc-Bench** | MedCalc-Bench | `sample`, `full` | Medical calculations |
| **AgentClinic** | AgentClinic | `medqa` (107), `nejm` (15) | Clinical diagnosis |
| **MedHall** | Custom | `theta` (30) | Hallucination detection |
| **MemoryArena** | MemoryArena | `sample` (10), `full` (701) | Agent memory |

### Add a new benchmark

Two ways:

**A) Use the Claude Code skill (recommended):**
```
/add-benchmark    # guided: research paper → convert data → validate
```

**B) Manual:**
1. Create `benchmark/data/<name>/metadata.json` with target config
2. Create `benchmark/data/<name>/<dataset>.jsonl` in BenchItem format
3. Run: `python -m benchmark.basic_runner <name> <dataset> --target-model gpt-4.1`

See [benchmark/data/history_demo/](benchmark/data/history_demo/) for a minimal example.

## Extending HolyEval

### Add an evaluator

```python
# evaluator/plugin/eval_agent/my_eval_agent.py
from evaluator.core.interfaces import AbstractEvalAgent, EvalResult

class MyEvalAgent(AbstractEvalAgent, name="my_eval"):
    async def run(self, memory_list, session_info):
        conversation = memory_list[-1].target_response
        score = your_scoring_logic(conversation)
        return EvalResult(score=score, passed=score > 0.8, feedback="...")
```

### Add a target system

```python
# evaluator/plugin/target_agent/my_target_agent.py
from evaluator.core.interfaces import AbstractTargetAgent

class MyTargetAgent(AbstractTargetAgent, name="my_target"):
    async def execute(self, message):
        response = await call_your_api(message)
        return response
```

Use `/add-eval-agent` or `/add-target-agent` Claude Code skills for guided scaffolding.

## Web UI

Launch with `python -m web`, then visit http://localhost:8000.

| Page | Route | Description |
|------|-------|-------------|
| Run evaluations | `/tasks` | Select benchmark, configure, launch with real-time SSE progress |
| Task details | `/tasks/{id}` | Progress cards, expandable cases with dialogue and feedback |
| Reports | `/reports/{benchmark}/{file}` | Scored report viewer |
| Datasets | `/benchmarks` | Browse benchmark datasets |
| Agent registry | `/agents/*` | Inspect registered plugins |

## CLI Reference

```bash
# Run benchmark
python -m benchmark.basic_runner <benchmark> <dataset> [options]
  --target-model MODEL    # LLM model to evaluate (e.g., gpt-4.1, gemini-3-pro)
  --target-type TYPE      # Target agent type (for multi-target benchmarks)
  --limit N               # Max cases to run
  --ids id1,id2           # Run specific case IDs
  -p N                    # Concurrency (default: 5)
  -v                      # Verbose output
  --resume                # Resume from last checkpoint

# Convert external datasets
python -m generator.healthbench.converter input.jsonl output.jsonl --target-model gpt-4.1
python -m generator.medcalc.converter
python -m generator.agentclinic.converter input.jsonl output.jsonl
python -m generator.memoryarena.converter

# Web UI
python -m web             # http://localhost:8000
```

## Configuration

Environment variables (in `.env`):

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | At least one | OpenAI API key |
| `GOOGLE_API_KEY` | At least one | Google Gemini API key |
| `OPENROUTER_API_KEY` | Optional | OpenRouter multi-provider access |
| `HOLYEVAL_PORT` | Optional | Web UI port (default: 8000) |

## Roadmap

### In Progress
- [ ] Generic HTTP API TargetAgent — evaluate your own product/API endpoints, not just raw LLMs
- [ ] Cross-model comparison & leaderboard — side-by-side scoring across models on the same benchmark

### Planned
- [ ] **Eval-driven optimization loop** — run benchmark → auto-analyze weaknesses → generate targeted improvements → re-run to verify
- [ ] **A/B evaluation mode** — compare two models/prompts case-by-case on the same dataset
- [ ] **Regression detection** — compare multiple runs over time, alert on score drops
- [ ] **Docker support** — `docker compose up` for one-command deployment
- [ ] **PyPI package** — `pip install holyeval`
- [ ] **SDK mode** — `holyeval.run("healthbench", model="gpt-4.1")` for CI pipeline integration
- [ ] **Human-in-the-loop review** — sample LLM-as-Judge results for human calibration
- [ ] **More benchmarks** — MMLU-Med, PubMedQA, MedQA-USMLE, BioASQ, and community contributions
- [ ] **Advanced visualization** — radar charts, error distribution, token efficiency analysis

## Development

```bash
# Run tests
pytest evaluator/tests/

# Lint & format
ruff check .
ruff format .
```

### Claude Code Skills

| Command | Description |
|---------|-------------|
| `/quick-start` | Project initialization guide |
| `/add-benchmark` | Integrate an external benchmark (paper → data → validate) |
| `/add-eval-agent` | Scaffold a new evaluator plugin |
| `/add-target-agent` | Scaffold a new target system plugin |
| `/run-benchmark` | Run benchmark tests |
| `/run-e2e-test` | Run end-to-end tests |
| `/review-architecture` | Review architecture health |

## Contributing

Contributions are welcome! Here's how to get started:

1. **Add a benchmark** — The fastest way to contribute. Use `/add-benchmark` or follow the manual steps above.
2. **Add an evaluator** — Implement a new `EvalAgent` plugin for a different evaluation methodology.
3. **Add a target** — Implement a new `TargetAgent` plugin to evaluate different systems.
4. **Improve existing benchmarks** — More test cases, better prompts, edge cases.

Please open an issue first to discuss significant changes.

## License

[MIT](LICENSE)
