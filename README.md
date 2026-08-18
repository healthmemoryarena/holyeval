<h1 align="center">
  <br>
  mirobody-eval
  <br>
</h1>

<p align="center">
  <strong>Reproduce any LLM benchmark with one command. Build your own with pluggable agents.<br>No code required — just talk to Claude Code.</strong>
</p>

<p align="center">
  <a href="https://github.com/thetahealth/mirobody-eval/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-3776AB.svg?logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="https://docs.anthropic.com/en/docs/claude-code"><img src="https://img.shields.io/badge/Claude_Code-native-cc785c.svg?logo=anthropic&logoColor=white" alt="Claude Code Native"></a>
  <a href="https://arxiv.org/abs/2604.02834"><img src="https://img.shields.io/badge/arXiv-2604.02834-b31b1b.svg" alt="arXiv Paper"></a>
  <a href="https://huggingface.co/datasets/healthmemoryarena/ESL-Bench"><img src="https://img.shields.io/badge/%F0%9F%A4%97_HuggingFace-ESL--Bench-FFD21E.svg" alt="HuggingFace Dataset"></a>
  <a href="http://healthmemoryarena.ai"><img src="https://img.shields.io/badge/%F0%9F%8C%90_Live-Health_Memory_Arena-black.svg" alt="Live Demo"></a>
  <a href="https://github.com/thetahealth/mirobody-eval/stargazers"><img src="https://img.shields.io/github/stars/thetahealth/mirobody-eval?style=social" alt="GitHub Stars"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#evaluate-your-own-mirobody-deployment">Evaluate mirobody</a> &middot;
  <a href="#ai-native-development-with-claude-code">Claude Code</a> &middot;
  <a href="#web-ui">Web UI</a> &middot;
  <a href="http://healthmemoryarena.ai">Live Demo</a> &middot;
  <a href="https://arxiv.org/abs/2604.02834">Paper</a> &middot;
  <a href="https://huggingface.co/datasets/healthmemoryarena/ESL-Bench">Dataset</a> &middot;
  <a href="#benchmarks">Benchmarks</a> &middot;
  <a href="#contributing">Contributing</a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2604.02834">
    <img src="docs/screenshots/eslbench_overview.png" alt="ESL-Bench: Event-Driven Longitudinal Health Agent Benchmark" width="80%">
  </a>
</p>

<p align="center">
  <em>ESL-Bench — an event-driven synthetic longitudinal benchmark for health agents.
  <br>100 synthetic users, 10,000 queries, 5 dimensions, programmatic ground truth.
  <br>Read the paper: <a href="https://arxiv.org/abs/2604.02834">arXiv:2604.02834</a></em>
</p>

---

mirobody-eval is the evaluation half of [mirobody](https://github.com/thetahealth/mirobody) — the open-source health data engine. It exists so mirobody's claims come with numbers attached: seed a synthetic user into your own deployment, run a benchmark against it, get a scored report. The framework underneath is general, so the same one command reproduces any published benchmark against any target you plug in. Drop in a benchmark dataset, run one command, get a scored report. Extend it with custom evaluators, target systems, and virtual users via a pluggable agent architecture.

Built from the ground up as a [Claude Code](https://docs.anthropic.com/en/docs/claude-code) native project — every workflow, from initial setup to integrating a new benchmark from a research paper, is an interactive slash command. You describe what you want in natural language, and Claude Code handles the rest. **You don't need to write a single line of code to use or extend this framework.**

### Integrate any benchmark — just paste the paper link

<p align="center">
  <img src="docs/screenshots/holyeval_add_benchmark.gif" alt="Add Benchmark Demo" width="80%">
</p>

> **From paper to scored report in one conversation.** Paste a link, Claude Code reads the paper, writes the converter, creates datasets, validates — done. No boilerplate, no manual file creation.

### Run any benchmark with one command

<p align="center">
  <img src="docs/screenshots/holyeval_run_benchmark.gif" alt="Run Benchmark Demo" width="80%">
</p>

```bash
# Try it out — each command costs < $0.05 with --limit 3
uv run python -m benchmark.basic_runner healthbench sample --target-model gpt-5.4-mini --limit 3
uv run python -m benchmark.basic_runner medcalc sample --target-model gpt-5.4-mini --limit 3
uv run python -m benchmark.basic_runner memoryarena sample --target-model gemini-3-pro --limit 3

# ESLBench requires data preparation first (see Quick Start below)
uv run python -m benchmark.basic_runner eslbench sample50-20260331 --target-model gpt-5.4-mini --limit 3

# Ready for a full run? Remove --limit to run the entire dataset
```

## Why mirobody-eval?

| | |
|---|---|
| **Paper → benchmark in one conversation** | Paste a paper link, Claude Code reads it, writes the converter, creates datasets, validates — done |
| **One-command reproduction** | Reproduce any integrated benchmark forever with a single CLI command |
| **Pluggable architecture** | Three agent types (TestAgent, TargetAgent, EvalAgent) — extend any of them with a single class |
| **Multi-turn dialogue** | Simulates real user conversations, not just single-turn Q&A |
| **Batch execution** | Concurrent runs with real-time progress, cancellation, and checkpoint resume |
| **Web UI** | Visual dashboard for running evaluations, viewing reports, and browsing datasets |
| **AI-native** | Built for [Claude Code](https://docs.anthropic.com/en/docs/claude-code) — set up, run, and extend the project through natural language, zero boilerplate |

## Quick Start

With [Claude Code](https://docs.anthropic.com/en/docs/claude-code): just run `/quick-start` — it handles everything automatically.

Or manually:

```bash
git clone https://github.com/thetahealth/mirobody-eval.git && cd mirobody-eval
uv sync
cp .env.example .env                    # add your OPENAI_API_KEY or GOOGLE_API_KEY

# Run your first benchmark (< $0.02)
uv run python -m benchmark.basic_runner healthbench sample --target-model gpt-5.4-mini --limit 2

# Launch Web UI
uv run python -m web                    # http://localhost:8000
```

> **Prerequisites:** Python 3.11+, [uv](https://docs.astral.sh/uv/), at least one LLM API key (OpenAI or Google Gemini).
>
> **ESLBench data prep:** ESLBench requires downloading data from HuggingFace first — run `uv run python -m generator.eslbench.prepare_data` (automatic via Web UI). Other benchmarks ship with data included.

## Evaluate your own mirobody deployment

This is what mirobody-eval is for. A fresh [mirobody](https://github.com/thetahealth/mirobody)
install has an empty database, so there is nothing to ask it about and no way to tell whether a
change you made helped. Three commands fix both:

```bash
# 1. Pull one synthetic user's five-year trajectory from HuggingFace (~20 MB)
uv run python -m generator.eslbench.prepare_data

# 2. Load it into your deployment's Postgres, and make the indicators searchable
uv run python -m generator.eslbench.seed_mirobody --users user5086@demo

# 3. Score your deployment on ESL-Bench
uv run python -m benchmark.basic_runner eslbench sample200-20260430 --target-type mirobody --limit 20
```

You now have a number for the five ESL-Bench reasoning dimensions — Lookup, Trend, Comparison,
Anomaly, Explanation. Change the model, switch agent type, edit a prompt, add a tool, re-run, and
see which dimensions moved. Point `--target-type llm_api` at the same questions for a
retrieval-only baseline to compare against.

> **Prerequisites for step 2:** `pip install mirobody` with config pointing at the deployment you
> want to seed, plus a working embedding-provider key. Seeding verifies afterwards that every
> indicator is actually reachable by the agent and **fails loudly if not** — the alternative is a
> database that looks full while the agent answers "I don't have your health data", with nothing in
> the logs to explain why.
>
> **Prerequisites for step 3:** the deployment's HTTP server running (`MIROBODY_BASE_URL`, default
> `http://localhost:18080`).

### The file-upload demo

`labreport` renders one of the synthetic user's lab panels as a PDF, so mirobody's ingest path has
something real to chew on. Hold that panel back when seeding and the upload contributes data the
database genuinely does not have yet — which is what turns "what is my lipid trend?" into a real
question instead of a single point:

```bash
uv run python -m generator.eslbench.seed_mirobody --users user5086@demo --hold-out-exams 1
uv run python -m generator.eslbench.labreport     --users user5086@demo -o samples/lab_report.pdf
```

`user5086@demo` is a generated 58-year-old with type 2 diabetes whose lipids improve and then drift
back across four panels. Every value is synthetic; the PDF says so on its front page.

## AI-Native Development with Claude Code

mirobody-eval is designed to be operated entirely through [Claude Code](https://docs.anthropic.com/en/docs/claude-code). Every common task has a dedicated slash command. You describe your intent in natural language; Claude Code reads the code, generates files, runs tests, and validates the result.

**You don't need to memorize CLI flags, read source code, or write boilerplate.** Just type the slash command and follow the conversation.

### Slash Command Reference

| What you want to do | Command | What Claude Code does for you |
|---|---|---|
| **Set up the project** | `/quick-start` | Checks Python/uv, installs dependencies, configures `.env` with your API keys, launches Web UI |
| **Run a benchmark** | `/run-benchmark` | Asks which benchmark & dataset, then executes with your chosen model and concurrency |
| **Integrate a new benchmark** | `/add-benchmark` | End-to-end: reads the paper/repo → analyzes data format → writes the converter → creates dataset → validates |
| **Add a custom evaluator** | `/add-eval-agent` | Scaffolds config model + plugin implementation + registration. Immediately available in CLI & Web UI |
| **Add a new target system** | `/add-target-agent` | Scaffolds connection handling, message processing, and cleanup for a new system under test |
| **Run end-to-end tests** | `/run-e2e-test` | Verifies all components (TestAgent ↔ TargetAgent ↔ EvalAgent ↔ Orchestrator) work together |
| **Audit architecture** | `/review-architecture` | Checks GitOps compliance, plugin isolation, shared-layer reuse. Reports violations with fix suggestions |

### Workflow Examples

**"I want to reproduce HealthBench on GPT-4.1"**
```
> /run-benchmark
# Claude asks: which benchmark? → healthbench
# Which dataset? → sample
# Which model? → gpt-5.4-mini
# How many cases? → 5 (start small!)
# Running... 5 cases → report saved
```

**"I need a custom evaluator that checks citation accuracy"**
```
> /add-eval-agent
# Claude asks: plugin name? → citation_accuracy
# What does it evaluate? → checks if AI responses cite valid sources
# Generates: evaluator/plugin/eval_agent/citation_accuracy_eval_agent.py
# Registered automatically via __init_subclass__ — ready to use
```

> **Tip:** You're not limited to slash commands. Claude Code understands the full codebase — ask it anything in natural language, like *"explain how the plugin system works"* or *"why did this test case fail?"*.

## Web UI

Launch with `uv run python -m web`, then visit http://localhost:8000.

<table>
<tr>
<td width="50%">

**Run Evaluations** — Select benchmark, configure parameters, launch tasks with real-time SSE progress tracking.

<img src="docs/screenshots/holyeval_tasks.jpg" alt="Run Evaluations" width="100%">
</td>
<td width="50%">

**Evaluation Report** — Scored results with expandable cases, dialogue history, and per-case feedback.

<img src="docs/screenshots/holyeval_report.jpg" alt="Evaluation Report" width="100%">
</td>
</tr>
<tr>
<td width="50%">

**Browse Benchmarks** — Overview of all benchmark datasets with case counts and statistics.

<img src="docs/screenshots/holyeval_benchmarks.jpg" alt="Browse Benchmarks" width="100%">
</td>
<td width="50%">

**Agent Registry** — Inspect all registered plugins with config schemas, features, and cost estimates.

<img src="docs/screenshots/holyeval_agents.jpg" alt="Agent Registry" width="100%">
</td>
</tr>
</table>

## Health Memory Arena — Live Evaluation Platform

[Health Memory Arena](http://healthmemoryarena.ai) (HMA) is the public evaluation platform powered by mirobody-eval. It hosts the ESL-Bench leaderboard where health AI agents compete on structured longitudinal reasoning tasks.

<table>
<tr>
<td width="33%">
<a href="http://healthmemoryarena.ai"><img src="docs/screenshots/hma_home.jpg" alt="HMA Home" width="100%"></a>
<p align="center"><em>Platform Home</em></p>
</td>
<td width="33%">
<a href="http://healthmemoryarena.ai/leaderboard"><img src="docs/screenshots/hma_leaderboard.jpg" alt="HMA Leaderboard" width="100%"></a>
<p align="center"><em>Agent Leaderboard</em></p>
</td>
<td width="33%">
<a href="http://healthmemoryarena.ai/dataset"><img src="docs/screenshots/hma_dataset.jpg" alt="HMA Dataset" width="100%"></a>
<p align="center"><em>Dataset Browser</em></p>
</td>
</tr>
</table>

## Architecture

```
TestCase (JSON) → Orchestrator
  1. Initialize agents from config via plugin registry
  2. Dialogue loop: TestAgent ↔ TargetAgent (until finished or max turns)
  3. EvalAgent.run(conversation, session) → EvalResult
  4. Return TestResult (score, pass/fail, feedback, cost)
```

All execution paths (CLI, Web UI, programmatic) funnel through a single entry point: `do_single_test()`.

<p align="center">
  <img src="docs/architecture.png" alt="mirobody-eval Architecture" width="80%">
</p>

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
| **EvalAgent** | Evaluator | `semantic`, `healthbench`, `medcalc`, `hallucination`, `kg_qa`, `memoryarena` |

### Project Structure

```
mirobody-eval/
├── evaluator/          # Core engine: schema, orchestrator, plugin interfaces
├── benchmark/          # Runner + datasets (JSONL) + reports
│   └── data/eslbench/  # ESLBench: data + tools (retrieve.py for JSON/DuckDB)
├── generator/          # Dataset converters + data preparation scripts
│   └── eslbench/       # ESLBench data downloader + DuckDB builder
└── web/                # Web UI (FastAPI + htmx)
```

## Benchmarks

| Benchmark | Paper / Source | Datasets | What it evaluates |
|---|---|---|---|
| **HealthBench** | [OpenAI HealthBench](https://arxiv.org/abs/2505.07469) | `sample` (100), `full`, `hard`, `consensus` | Medical AI quality |
| **MedCalc-Bench** | [MedCalc-Bench](https://arxiv.org/abs/2406.12036) | `sample`, `full` | Medical calculations |
| **ESLBench** | [arXiv:2604.02834](https://arxiv.org/abs/2604.02834) | `sample50-20260331` (50), `sample500-20260331` (500), `full-20260331` (4500) | Longitudinal health reasoning |
| **AgentClinic** | [AgentClinic](https://arxiv.org/abs/2405.07960) | `medqa` (107), `nejm` (15) | Clinical diagnosis |
| **MedHall** | Custom | `theta` (30) | Hallucination detection |
| **MemoryArena** | [MemoryArena](https://arxiv.org/abs/2501.13916) | `sample` (10), `full` (701) | Agent memory |

### ESLBench — Event-Driven Synthetic Longitudinal Benchmark

ESLBench ([arXiv:2604.02834](https://arxiv.org/abs/2604.02834)) evaluates longitudinal health reasoning — the ability to align, aggregate, and attribute across multi-source patient trajectories combining device streams, clinical exams, and life events. Built on an event-driven synthesis framework where each user trajectory is modeled as a baseline health state plus discrete events with explicit temporal kernels (sigmoid onset, exponential decay), making ground truth programmatically computable.

<p align="center">
  <img src="docs/screenshots/eslbench_trajectory.png" alt="ESL-Bench Trajectory Visualization" width="70%">
</p>

<p align="center"><em>Four-month trajectory excerpt — event-driven indicator dynamics with sigmoid onset and exponential decay.</em></p>

**100 synthetic users** with 1–5 year trajectories, **10,000 evaluation queries** across five dimensions and three difficulty tiers:

| Dimension | What it tests | Example |
|---|---|---|
| **Lookup** | Direct data retrieval | "What was resting heart rate on 2024-03-15?" |
| **Trend** | Temporal pattern analysis | "In which month was step count highest?" |
| **Comparison** | Cross-event/source comparisons | "How did mean steps change after jogging started?" |
| **Anomaly** | Abnormality detection | "Has glucose ever been abnormal?" |
| **Explanation** | Causal attribution | "Rank events by impact on glucose drop" |

<details>
<summary><strong>Benchmark results — 13 methods across 3 paradigms</strong></summary>
<br>
<p align="center">
  <img src="docs/screenshots/eslbench_results.png" alt="ESL-Bench Main Results" width="80%">
</p>

Key findings: DB agents (48–58%) substantially outperform memory RAG (30–38%), with the gap concentrated on Comparison and Explanation queries where multi-hop reasoning and evidence attribution are required.
</details>

**Data preparation required** — ESLBench downloads user data from HuggingFace and builds per-user DuckDB indexes:

```bash
# First time: prepare data (automatic via Web UI, manual for CLI)
uv run python -m generator.eslbench.prepare_data

# Quick test: 3 cases to verify setup (< $0.05)
uv run python -m benchmark.basic_runner eslbench sample50-20260331 --target-model gpt-5.4-mini --limit 3

# Sample datasets
uv run python -m benchmark.basic_runner eslbench sample50-20260331 --target-model gpt-5.4-mini      # 50 cases
uv run python -m benchmark.basic_runner eslbench sample500-20260331 --target-model gpt-5.4-mini -p 5 # 500 cases

# Full benchmark (4500 cases — significant API cost, review before running)
uv run python -m benchmark.basic_runner eslbench full-20260331 --target-model gpt-5.4-mini -p 5
```

The LLM target is equipped with a tool group (`eslbench/retrieve`) that provides JSON file reading, DuckDB queries, and indicator lookup — the LLM must use these tools to find answers in the user's health data.

### Add a new benchmark

Two ways:

**A) Use the Claude Code skill (recommended):**
```
/add-benchmark    # guided: research paper → convert data → validate
```

**B) Manual:**
1. Create `benchmark/data/<name>/metadata.json` with target config
2. Create `benchmark/data/<name>/<dataset>.jsonl` in BenchItem format
3. Run: `uv run python -m benchmark.basic_runner <name> <dataset> --target-model gpt-5.4-mini`

See [benchmark/data/history_demo/](benchmark/data/history_demo/) for a minimal example.

## Extending mirobody-eval

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

## CLI Reference

```bash
# Prepare benchmark data (required for ESLBench; other benchmarks ship with data)
uv run python -m generator.eslbench.prepare_data          # download HF data + build DuckDB
uv run python -m generator.eslbench.prepare_data --force   # force rebuild

# Run benchmark
uv run python -m benchmark.basic_runner <benchmark> <dataset> [options]
  --target-model MODEL    # LLM model to evaluate (e.g., gpt-5.4-mini, gemini-3-pro)
  --target-type TYPE      # Target agent type (for multi-target benchmarks)
  --limit N               # Max cases to run
  --ids id1,id2           # Run specific case IDs
  -p N                    # Concurrency (default: 5)
  -v                      # Verbose output
  --resume                # Resume from last checkpoint

# Convert external datasets
uv run python -m generator.healthbench.converter input.jsonl output.jsonl --target-model gpt-5.4-mini
uv run python -m generator.medcalc.converter
uv run python -m generator.agentclinic.converter input.jsonl output.jsonl
uv run python -m generator.memoryarena.converter

# Web UI
uv run python -m web             # http://localhost:8000
```

## Configuration

Environment variables (in `.env`):

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | At least one | OpenAI API key |
| `GOOGLE_API_KEY` | At least one | Google Gemini API key |
| `HF_TOKEN` | ESLBench | HuggingFace token for downloading benchmark data |
| `OPENROUTER_API_KEY` | Optional | OpenRouter multi-provider access |
| `HOLYEVAL_PORT` | Optional | Web UI port (default: 8000) |

## Roadmap

### In Progress
- [ ] **GUI TargetAgent** — evaluate real products through their web UI, not just API endpoints. Browser-based agent interacts with your app like a real user, enabling end-to-end evaluation of any product with a frontend

### Planned
- [ ] **Eval-driven optimization loop** — run benchmark → auto-analyze failure patterns → generate targeted prompt/system improvements → re-run to verify. Close the loop between evaluation and iteration
- [ ] **CI/CD integration** — `pip install mirobody-eval` + `mirobody_eval.run("healthbench", model="gpt-5.4-mini")` as a one-liner in your CI pipeline. Regression detection across runs, alerting on score drops before deployment
- [ ] **Industry agent & app deep evaluation** — comprehensive evaluation reports for mainstream AI agents and health apps (e.g. ChatGPT, Gemini, health assistants). Standardized scoring across safety, accuracy, and user experience, published as reproducible community benchmarks

## Development

```bash
# Run tests
uv run pytest evaluator/tests/

# Lint & format
uv run ruff check .
uv run ruff format .
```

## Contributing

Contributions are welcome! The easiest way to contribute is through Claude Code — every workflow below has a guided slash command:

| Contribution type | How to start | Difficulty |
|---|---|---|
| **Add a benchmark** | `/add-benchmark` — the fastest way to contribute | Easy |
| **Add an evaluator** | `/add-eval-agent` — scaffold a new scoring methodology | Medium |
| **Add a target system** | `/add-target-agent` — connect a new API/service to evaluate | Medium |
| **Improve existing benchmarks** | Add more test cases, edge cases, or better prompts | Easy |

Please open an issue first to discuss significant changes.

## Citation

If you use ESL-Bench or mirobody-eval in your research, please cite:

```bibtex
@article{li2026eslbench,
  title={ESL-Bench: An Event-Driven Synthetic Longitudinal Benchmark for Health Agents},
  author={Li, Chao and Liu, Cailiang and Gao, Ang and Deng, Kexin and Zhang, Shu and Xu, Langping and Shi, Xiaotong and Ding, Xionghao and Pei, Jian and Jiang, Xun},
  journal={arXiv preprint arXiv:2604.02834},
  year={2026}
}
```

## License

[MIT](LICENSE)
