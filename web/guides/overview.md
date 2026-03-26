<!-- Project Overview - HolyEval -->

## Introduction

HolyEval is a **virtual user evaluation framework** designed for AI medical assistants (Theta Health). It systematically evaluates AI assistant performance through automated multi-turn conversation testing and multi-dimensional assessment.

## Architecture Overview

![HolyEval Architecture Diagram](/static/images/architecture.png)

HolyEval adopts a modular design, consisting of six core components as shown above. Components 2-6 correspond to the three core directories of the project:

- **evaluator** — Corresponds to components ②③: Virtual user evaluation framework + system-under-test integration
- **generator** — Corresponds to components ④⑥: Benchmark Generator + Benchmark Wrapper
- **benchmark** — Corresponds to component ⑤: Benchmark Scheduler (batch scheduling center)

### ② Virtual User Evaluation Framework (evaluator/)

Provides core evaluation capabilities based on a three-layer Agent architecture:

| Agent Type | Responsibility | Capabilities |
|-----------|---------|---------|
| **TestAgent** | Simulates virtual user behavior | Supports LLM-driven natural conversation and script-driven exact replay |
| **TargetAgent** | Integrates with the system under test | Unified wrapper for real system APIs and simulated LLM calls |
| **EvalAgent** | Evaluates conversation quality | Offers semantic, indicator, rule-based, reference answer, and other multi-dimensional evaluation methods |

**Core Evaluation Flow**:
1. TestAgent initiates conversation (based on user goal and context)
2. TargetAgent calls the system under test to obtain responses
3. Conversation repeats until the goal is met or the turn limit is exceeded
4. EvalAgent performs multi-dimensional evaluation based on the complete conversation history
5. Generates TestResult (score, pass rate, detailed feedback, cost statistics)

### ③ System-Under-Test Integration (evaluator/plugin/target_agent/)

Supports two execution modes for integrating different types of systems under test:

- **Real system integration**: Interacts with the production environment via the Theta Health HTTP API to verify real-world performance
- **Simulated system integration**: Directly calls LLM APIs to simulate system behavior for quick validation of evaluation logic

### ④ Benchmark Generator (generator/)

**Capabilities**: Converts external evaluation datasets into HolyEval standard format

- **Data conversion**: Supports mainstream AI evaluation datasets including HealthBench, MedCalc-Bench, MemoryArena, and more
- **Business abstraction**: Maps raw evaluation scenarios (prompt + rubrics) to the BenchItem standard format, supporting multi-turn conversation context (`history` field)
- **Flexible configuration**: Supports custom evaluation criteria, conversation history, and virtual user configuration

### ⑤ Benchmark Scheduler (benchmark/)

**Capabilities**: Scheduling, execution, and progress management for batch evaluations

- **Batch scheduling**: Supports concurrent execution of thousands of test cases (configurable concurrency)
- **Real-time monitoring**: Tracks execution status of each case (pending/in-dialogue/evaluating/completed/cancelled)
- **Flexible control**: Supports task cancellation at any time and progress snapshot export
- **Data organization**: Manages test data in directories organized by evaluation type (healthbench/medcalc/extraction)

### ⑥ Benchmark Wrapper (shared with generator/)

**Capabilities**: Standardized packaging and persistence of evaluation results

- **Result aggregation**: Aggregates individual case results into complete evaluation reports (total score, pass rate, statistics)
- **Metadata tracking**: Records the Agent type used for each case, facilitating subsequent analysis
- **Report management**: Automatically saves reports to the `benchmark/report/` directory, supporting historical report queries

## Directory Structure and Responsibilities

```
holyeval/
├── evaluator/              # ②③ Virtual user evaluation framework + system integration
│   ├── core/              # Core evaluation engine (orchestrator, data models)
│   ├── plugin/            # Three-layer Agent plugin implementations
│   └── utils/             # Shared utilities (LLM calls, data reading)
│
├── generator/             # ④⑥ Data generation + result wrapping
│   ├── healthbench/       # HealthBench data converter
│   ├── medcalc/           # MedCalc-Bench data converter
│   ├── agentclinic/       # AgentClinic data converter
│   ├── medhall/           # MedHall hallucination case generation + conversion
│   └── memoryarena/       # MemoryArena data converter
│
├── benchmark/             # ⑤ Batch scheduling center
│   ├── data/              # Evaluation datasets (organized by type)
│   ├── report/            # Evaluation reports (auto-generated)
│   └── basic_runner.py    # CLI runner
│
└── web/                   # Web management interface
    ├── app/               # API services (task scheduling, report queries)
    ├── templates/         # Frontend pages
    └── guides/            # Developer guides
```

## Core Evaluation Capabilities

HolyEval provides six evaluation methods covering different evaluation scenarios:

| Evaluator | Use Case | Evaluation Method |
|--------|---------|---------|
| **semantic** | General semantic quality | LLM-based subjective quality assessment (safety, empathy, professionalism) |
| **healthbench** | HealthBench standard | Rubric-based multi-dimensional scoring (concurrent evaluation of all rubric items) |
| **medcalc** | Medical calculation | LLM answer extraction + typed numerical matching (decimal/integer/date/weeks_days) |
| **hallucination** | Medical hallucination detection | LLM-as-Judge detecting factual/contextual/citation hallucinations, scoring 0~1 |
| **indicator** | Health data accuracy | Validates whether health indicators (exercise, sleep) returned by the system are correct |
| **keyword** | Keyword rules | Quick validation that responses contain required keywords |
| **preset_answer** | Reference answer matching | Exact match or keyword match against preset answers |
| **redteam_compliance** | Red-team compliance testing | LLM-as-Judge evaluating medical AI response compliance (defense against adversarial prompts) |
| **memoryarena** | Agent memory evaluation | LLM-as-Judge per-subtask assessment + Progress Score (5 domains) |

## Execution Modes

Three execution methods are supported to fit different use cases:

- **CLI execution**: Command-line batch benchmarking, suitable for CI/CD integration and automated testing
- **Web UI execution**: Visual interface with real-time progress and results, suitable for manual validation and analysis
- **API execution**: HTTP interface integration, suitable for embedding in other systems or services

## Next Steps

| Action | Description |
|------|------|
| [Run Evaluations](/tasks) | Select a dataset, configure parameters, view progress in real time |
| [View Reports](/tasks) | Browse historical evaluation results |
| [Benchmark Data](/benchmarks) | Browse datasets by directory |
| [Agent Registry](/agents/test) | View registered plugins |

**Skills Quick Commands**:

- `/add-benchmark` — Integrate an external benchmark (research -> conversion -> validation end-to-end)
- `/add-eval-agent` — Scaffold: add a new evaluation logic plugin
- `/add-target-agent` — Scaffold: add a new system-under-test plugin
- `/run-e2e-test` — Run end-to-end tests
- `/run-benchmark` — Run benchmark scoring
