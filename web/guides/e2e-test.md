# End-to-End Testing

> **Using in Claude Code**: Type `/run-e2e-test`, followed by your testing requirements (e.g., specific cases, test scope). Claude will automatically run the end-to-end tests and analyze the results.

## Overview

End-to-end tests cover the complete evaluation flow: load test cases -> initialize Agents -> multi-turn conversation between virtual user and system under test -> evaluator scoring -> output results. A passing test confirms that all framework components (TestAgent, TargetAgent, EvalAgent, Orchestrator) are working together correctly.

## Running E2E Tests

### Prerequisites

Ensure `OPENAI_API_KEY` is configured in `.env`, as e2e tests require LLM calls.

### Run All

```bash
pytest evaluator/tests/test_e2e.py -v -s
```

### Run a Single Case

Use `-k` to filter by case ID:

```bash
pytest evaluator/tests/test_e2e.py -v -s -k "manual_headache_001"
pytest evaluator/tests/test_e2e.py -v -s -k "auto_history_llm_004"
```

### Run Batch Tests (with Progress Tracking)

```bash
pytest evaluator/tests/test_e2e.py -v -s -k "batch"
```

## Test Cases

E2E test cases are in `evaluator/tests/fixtures/test_cases.jsonl`, containing 4 cases:

| ID | User Mode | Target System | Evaluator |
|----|----------|----------|--------|
| `manual_headache_001` | manual | theta_api | semantic |
| `auto_cough_child_002` | auto | theta_api | semantic |
| `auto_history_sleep_003` | auto | theta_api | semantic |
| `auto_history_llm_004` | auto | llm_api | semantic |

Coverage includes:
- **manual / auto** — both TestAgent modes
- **theta_api / llm_api** — both TargetAgent types
- Preset conversation scenarios with history

## What Tests Verify

Each case validates upon passing:

1. Evaluation result is `pass` or `fail`, with a score in the `0.0~1.0` range
2. Non-empty feedback commentary
3. Valid execution timing (`start <= end`)
4. Cases with history: historical conversation is preserved in the trace
5. llm_api target: target cost records are present

Batch tests additionally verify:
- `TestReport` aggregate metrics (pass_count, fail_count, pass_rate, avg_score)
- `BatchSession` progress tracking and snapshot functionality

## Running All Unit Tests

```bash
pytest evaluator/tests/ -v
```

> Note: E2E tests require an API key. Without one configured, they are automatically skipped (`skipif OPENAI_API_KEY`). Mock-based unit tests do not require an API key.
