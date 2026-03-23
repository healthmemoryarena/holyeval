# HolyEval

An open-source virtual user evaluation framework for AI medical assistants. HolyEval synthesizes virtual users to have multi-turn conversations with the system under test, then automatically evaluates performance via pluggable evaluators.

## Features

- **Multi-turn dialogue evaluation** — Simulates real user conversations, not just single-turn Q&A
- **Pluggable agent architecture** — TestAgent (virtual user), TargetAgent (system under test), EvalAgent (evaluator) are all extensible plugins
- **Built-in benchmarks** — HealthBench, MedCalc-Bench, AgentClinic, MedHall, MemoryArena and more
- **Batch execution with observability** — Real-time progress tracking, cancellation, and checkpoint resume
- **Web UI** — Visual dashboard for running evaluations, viewing reports, and browsing benchmarks
- **Data converters** — Transform external benchmark datasets into HolyEval format

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- At least one LLM API key (OpenAI or Google Gemini)

### Installation

```bash
git clone https://github.com/anthropics/holyeval.git
cd holyeval

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Run a benchmark

```bash
# HealthBench sample (100 cases)
python -m benchmark.basic_runner healthbench sample --target-model gpt-4.1

# With concurrency and verbosity
python -m benchmark.basic_runner healthbench sample --target-model gpt-4.1 --limit 10 -p 3 -v

# Resume interrupted run
python -m benchmark.basic_runner healthbench sample --resume
```

### Launch Web UI

```bash
python -m web    # http://localhost:8000
```

## Architecture

```
TestCase (JSON) → Orchestrator (do_single_test)
  1. Initialize agents from TestCase config via plugin registry
  2. Dialogue loop: TestAgent ↔ TargetAgent (until is_finished or max_turns)
  3. EvalAgent.run(memory_list, session_info) → EvalResult
  4. Return TestResult (score, pass/fail, feedback, cost)
```

All call paths (CLI, Web UI, API) funnel through `evaluator/core/orchestrator.py:do_single_test()`.

### Plugin System

Three agent types use `__init_subclass__` auto-registration:

```python
class CustomEvalAgent(AbstractEvalAgent, name="custom"):
    async def run(self, memory_list, session_info):
        ...

# Lookup: AbstractEvalAgent.get("custom")
```

| Agent Type | Role | Built-in Plugins |
|---|---|---|
| **TestAgent** | Virtual user | `auto` (LLM-driven), `manual` (scripted) |
| **TargetAgent** | System under test | `llm_api` (OpenAI / Gemini) |
| **EvalAgent** | Evaluator | `semantic`, `keyword`, `preset_answer`, `healthbench`, `medcalc`, `hallucination`, `redteam_compliance`, `memoryarena` |

### Project Structure

```
holyeval/
├── evaluator/          # Core evaluation engine
│   ├── core/           # Schema, orchestrator, interfaces
│   ├── plugin/         # Agent implementations
│   └── utils/          # LLM, config, readers
├── benchmark/          # Benchmark runner + data + reports
│   ├── data/           # Benchmark datasets (JSONL + metadata)
│   └── basic_runner.py # CLI runner
├── generator/          # Dataset converters
└── web/                # Web UI (FastAPI + htmx)
```

## Benchmarks

| Benchmark | Description | Datasets |
|---|---|---|
| **HealthBench** | Medical AI evaluation | `sample` (100), `full`, `hard`, `consensus` |
| **MedCalc-Bench** | Medical calculations | `sample`, `full` |
| **AgentClinic** | Multi-specialty clinical diagnosis | `medqa` (107), `nejm` (15) |
| **MedHall** | Medical hallucination detection | `theta` |
| **MemoryArena** | Agent memory evaluation | `sample`, `full` |

### Adding a benchmark

Each benchmark directory contains:
- `metadata.json` — Suite metadata (description, target configuration)
- `<dataset>.jsonl` — Test data in BenchItem format

Use the `/add-benchmark` Claude Code skill for guided scaffolding, or see `benchmark/data/history_demo/` for a minimal example.

## Web UI

Launch with `python -m web`, then visit http://localhost:8000.

| Page | Route | Description |
|------|-------|-------------|
| Run evaluations | `/tasks` | Select benchmark, configure params, launch with SSE progress |
| Task details | `/tasks/{id}` | Progress cards, expandable case list with dialogue/feedback |
| Reports | `/reports/{benchmark}/{filename}` | Full report viewer |
| Datasets | `/benchmarks` | Browse benchmark datasets |
| Agent registry | `/agents/*` | Inspect registered plugins |

## Configuration

Environment variables (configured in `.env`):

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | At least one LLM | OpenAI API key |
| `GOOGLE_API_KEY` | At least one LLM | Google Gemini API key |
| `OPENROUTER_API_KEY` | Optional | OpenRouter multi-provider access |
| `HOLYEVAL_PORT` | Optional | Web UI port (default: 8000) |

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
| `/add-benchmark` | Integrate an external benchmark (research → convert → validate) |
| `/add-eval-agent` | Scaffold a new evaluator plugin |
| `/add-target-agent` | Scaffold a new target system plugin |
| `/run-benchmark` | Run benchmark tests |
| `/run-e2e-test` | Run end-to-end tests |
| `/review-architecture` | Review architecture health |

## License

[MIT](LICENSE)
